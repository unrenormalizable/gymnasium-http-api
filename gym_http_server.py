#!/usr/bin/env python3
import json
import uuid
import argparse
import logging
import base64
from flask import Flask, request, jsonify
import gymnasium as gym
import numpy as np

logger = logging.getLogger("werkzeug")


def _mapper(obj):
    ret = obj
    if isinstance(ret, tuple):
        ret = ret[0]
    if isinstance(ret, int):
        ret = np.array([ret])
    if isinstance(ret, np.float32):
        ret = np.array([ret])
    return ret


def _replace_inf(num):
    if np.isneginf(num):
        return np.finfo(np.float64).min
    if np.isposinf(num):
        return np.finfo(np.float64).max
    return num.item()


def _normalize_infs(nums):
    return [_replace_inf(x) for x in np.array(nums).flatten()]


########## Container for environments ##########
class Envs:
    """
    Container and manager for the environments instantiated
    on this server.

    When a new environment is created, such as with
    envs.create('CartPole-v1'), it is stored under a short
    identifier (such as '3c657dbc'). Future API calls make
    use of this instance_id to identify which environment
    should be manipulated.
    """

    def __init__(self):
        self.envs = {}
        self.id_len = 8

    def _lookup_env(self, instance_id):
        try:
            return self.envs[instance_id]
        except KeyError as e:
            raise InvalidUsage(f"Instance_id {instance_id} unknown") from e

    def _remove_env(self, instance_id):
        try:
            del self.envs[instance_id]
        except KeyError as e:
            raise InvalidUsage(f"Instance_id {instance_id} unknown") from e

    def _add_alpha_channel(self, rf):
        if isinstance(rf, np.ndarray) and rf.dtype == np.uint8:
            return np.dstack((rf, np.full((rf.shape[0], rf.shape[1]), 255, dtype=np.uint8)))
        return rf

    def _render_frame_jsonable(self, rf):
        jsonable = None
        if isinstance(rf, str):
            jsonable = rf
        elif isinstance(rf, np.ndarray) and rf.dtype == np.uint8:
            # Refer: https://stackoverflow.com/questions/53548127/post-numpy-array-with-json-to-flask-app-with-requests
            jsonable = {
                "rows": rf.shape[0],
                "cols": rf.shape[1],
                "data": base64.b64encode(rf.tobytes()).decode("utf-8"),
            }
        else:
            jsonable = rf
        return jsonable

    def create(self, env_id, max_episode_steps, auto_reset, disable_env_checker, kwargs):
        try:
            env = gym.make(env_id, max_episode_steps, auto_reset, False, disable_env_checker, **kwargs)
        except gym.error.Error as e:
            raise InvalidUsage(f"Attempted to look up malformed environment ID '{env_id}'") from e

        instance_id = str(uuid.uuid4().hex)[: self.id_len]
        self.envs[instance_id] = env
        return instance_id

    def list_all(self):
        return {instance_id: env.spec.id for (instance_id, env) in self.envs.items()}

    def reset(self, instance_id, seed):
        env = self._lookup_env(instance_id)
        seed = int(seed) if seed is not None else None
        obs = env.reset(seed=seed)
        obs = _mapper(obs)
        return env.observation_space.to_jsonable(obs)

    def get_id(self, instance_id):
        _id = self._lookup_env(instance_id).spec.id
        return _id

    def render(self, instance_id):
        env = self._lookup_env(instance_id)
        render_frame = env.render()
        render_frame = self._add_alpha_channel(render_frame)
        return self._render_frame_jsonable(render_frame)

    def step(self, instance_id, action):
        env = self._lookup_env(instance_id)
        if isinstance(action, int):
            nice_action = action
        else:
            nice_action = np.array(action)
        observation, reward, terminated, truncated, info = env.step(nice_action)
        observation = _mapper(observation)
        obs_jsonable = env.observation_space.to_jsonable(observation)
        return [obs_jsonable, reward, terminated, truncated, info]

    def get_action_space_contains(self, instance_id, x):
        env = self._lookup_env(instance_id)
        return env.action_space.contains(int(x))

    def get_action_space_info(self, instance_id):
        env = self._lookup_env(instance_id)
        return self._get_space_properties(env.action_space)

    def get_action_space_sample(self, instance_id):
        env = self._lookup_env(instance_id)
        action = env.action_space.sample()
        if isinstance(action, (list, tuple)) or ("numpy" in str(type(action))):
            try:
                action = action.tolist()
            except TypeError:
                print(type(action))
                print("TypeError")
        return action

    def get_observation_space_contains(self, instance_id, j):
        env = self._lookup_env(instance_id)
        info = self._get_space_properties(env.observation_space)
        for key, value in j.items():
            # Convert both values to json for compaibility
            if json.dumps(info[key]) != json.dumps(value):
                print(f'Values for "{key}" do not match. Passed "{value}", Observed "{info[key]}".')
                return False
        return True

    def get_observation_space_info(self, instance_id):
        env = self._lookup_env(instance_id)
        return self._get_space_properties(env.observation_space)

    def get_transitions(self, instance_id):
        env = self._lookup_env(instance_id)
        return env.unwrapped.P

    def get_episode_samples(self, instance_id, seed, count):
        seed = int(seed) if seed is not None else None
        count = int(count)
        eps = []
        for _ in range(count):
            obs = self.reset(instance_id, seed)
            ep = [{"s": obs, "r": 0.0}]
            eps.append(ep)
            while True:
                a = self.get_action_space_sample(instance_id)
                si = self.step(instance_id, a)
                ep.append({"s": si[0], "r": si[1]})
                if si[2]:
                    break
        return eps

    def _get_space_properties(self, space):
        info = {}
        info["name"] = space.__class__.__name__
        if info["name"] == "Discrete":
            info["n"] = space.n.item()
        elif info["name"] == "Box":
            info["shape"] = space.shape
            # It's not JSON compliant to have Infinity, -Infinity, NaN.
            # Many newer JSON parsers allow it, but many don't. Notably python json
            # module can read and write such floats. So we only here fix "export version",
            # also make it flat.
            info["low"] = _normalize_infs(space.low)
            info["high"] = _normalize_infs(space.high)
        elif info["name"] == "HighLow":
            info["num_rows"] = space.num_rows
            info["matrix"] = _normalize_infs(space.matrix)
        return info

    def env_close(self, instance_id):
        env = self._lookup_env(instance_id)
        env.close()
        self._remove_env(instance_id)


########## App setup ##########
app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
envs = Envs()


########## Error handling ##########
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


def get_required_param(json_, param):
    if json_ is None:
        logger.info("Request is not a valid json")
        raise InvalidUsage("Request is not a valid json")
    value = json_.get(param, None)
    if (value is None) or (value == "") or (value == []):
        logger.info("A required request parameter '%s' had value %s", param, value)
        raise InvalidUsage(f"A required request parameter '{param}' was not provided")
    return value


def get_optional_param(json_, param, default):
    if json_ is None:
        logger.info("Request is not a valid json")
        raise InvalidUsage("Request is not a valid json")
    value = json_.get(param, None)
    if (value is None) or (value == "") or (value == []):
        logger.info(
            "An optional request parameter '%s' had value %s and was replaced with default value %s",
            param,
            value,
            default,
        )
        value = default
    return value


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


########## API route definitions ##########
@app.route("/v1/envs/", methods=["POST"])
def env_create():
    """
    Create an instance of the specified environment

    Parameters:
        - Refer: https://gymnasium.farama.org/api/registry/#gymnasium.make
    Returns:
        - instance_id: a short identifier (such as '3c657dbc')
        for the created environment instance. The instance_id is
        used in future API calls to identify the environment to be
        manipulated
    """
    print(f"#### {request.get_json()}")
    env_id = get_required_param(request.get_json(), "env_id")
    max_episode_steps = get_optional_param(request.get_json(), "max_episode_steps", None)
    auto_reset = get_optional_param(request.get_json(), "auto_reset", None)
    disable_env_checker = get_optional_param(request.get_json(), "disable_env_checker", None)
    kwargs = get_optional_param(request.get_json(), "kwargs", {})
    instance_id = envs.create(env_id, max_episode_steps, auto_reset, disable_env_checker, kwargs)
    return jsonify(instance_id=instance_id)


@app.route("/v1/envs/", methods=["GET"])
def env_list_all():
    """
    List all environments running on the server

    Returns:
        - envs: dict mapping instance_id to env_id
        (e.g. {'3c657dbc': 'CartPole-v1'}) for every env
        on the server
    """
    all_envs = envs.list_all()
    return jsonify(all_envs=all_envs)


@app.route("/v1/envs/<instance_id>/", methods=["GET"])
def env_get_id(instance_id):
    """
    Get the env_id.

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
        - env_id: the id of the environment
    """
    _id = envs.get_id(instance_id)
    return jsonify(id=_id)


@app.route("/v1/envs/<instance_id>/reset/", methods=["POST"])
def env_reset(instance_id):
    """
    Reset the state of the environment and return an initial
    observation.

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
        - seed: set the seed for this env's random number generator(s).
    Returns:
        - observation: the initial observation of the space
    """
    seed = get_optional_param(request.get_json(), "seed", None)
    observation = envs.reset(instance_id, seed)
    return jsonify(observation=observation)


@app.route("/v1/envs/<instance_id>/render/", methods=["GET"])
def env_render(instance_id):
    """
    Compute the render frames as specified by render_mode during the initialization
    of the environment.

    Returns:
        - render_frame: the computed render_frame.
    """
    render_frame = envs.render(instance_id)
    return jsonify(render_frame=render_frame)


@app.route("/v1/envs/<instance_id>/step/", methods=["POST"])
def env_step(instance_id):
    """
    Run one timestep of the environment's dynamics.

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
        - action: an action to take in the environment
    Returns:
        - observation: agent's observation of the current
        environment
        - reward: amount of reward returned after previous action
        - done: whether the episode has ended
        - info: a dict containing auxiliary diagnostic information
    """
    json_ = request.get_json()
    action = get_required_param(json_, "action")
    [obs_jsonable, reward, terminated, truncated, info] = envs.step(instance_id, action)
    return jsonify(observation=obs_jsonable, reward=reward, terminated=terminated, truncated=truncated, info=info)


@app.route("/v1/envs/<instance_id>/action_space/", methods=["GET"])
def env_action_space_info(instance_id):
    """
    Get information (name and dimensions/bounds) of the env's
    action_space

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
    - info: a dict containing 'name' (such as 'Discrete'), and
    additional dimensional info (such as 'n') which varies from
    space to space
    """
    info = envs.get_action_space_info(instance_id)
    return jsonify(info=info)


@app.route("/v1/envs/<instance_id>/action_space/sample/", methods=["GET"])
def env_action_space_sample(instance_id):
    """
    Get a sample from the env's action_space

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:

        - action: a randomly sampled element belonging to the action_space
    """
    action = envs.get_action_space_sample(instance_id)
    return jsonify(action=action)


@app.route("/v1/envs/<instance_id>/action_space/contains/<action>/", methods=["GET"])
def env_action_space_contains(instance_id, action):
    """
    Assess that value is a member of the env's action_space

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
            - x: the value to be checked as member
    Returns:
        - member: whether the value passed as parameter belongs to the action_space
    """

    member = envs.get_action_space_contains(instance_id, action)
    return jsonify(member=member)


@app.route("/v1/envs/<instance_id>/observation_space/contains/", methods=["POST"])
def env_observation_space_contains(instance_id):
    """
    Assess that the parameters are members of the env's observation_space

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
        - member: whether all the values passed belong to the observation_space
    """
    j = request.get_json()
    member = envs.get_observation_space_contains(instance_id, j)
    return jsonify(member=member)


@app.route("/v1/envs/<instance_id>/observation_space/", methods=["GET"])
def env_observation_space_info(instance_id):
    """
    Get information (name and dimensions/bounds) of the env's
    observation_space

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
        - info: a dict containing 'name' (such as 'Discrete'),
        and additional dimensional info (such as 'n') which
        varies from space to space
    """
    info = envs.get_observation_space_info(instance_id)
    return jsonify(info=info)


@app.route("/v1/envs/<instance_id>/transitions/", methods=["GET"])
def env_get_transitions(instance_id):
    """
    Get transition probability from state given action.

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
        - transition: all transitions as tuple (probability of transition, next state, reward, done)
    """
    probs = envs.get_transitions(instance_id)
    return jsonify(transitions=probs)


@app.route("/v1/envs/<instance_id>/episodes/", methods=["POST"])
def env_episode_samples(instance_id):
    """
    Generates a given number of episodes.

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
        - seed: set the seed for this env's random number generator(s).
        - count: number of episodes to generate
    Returns:
        - episodes: generated episodes as list of { s: state, r: reward }
    """
    json_ = request.get_json()
    count = get_required_param(json_, "count")
    seed = get_optional_param(json_, "seed", None)
    episodes = envs.get_episode_samples(instance_id, seed, count)
    return jsonify(episodes=episodes)


@app.route("/v1/envs/<instance_id>/", methods=["DELETE"])
def env_close(instance_id):
    """
    Manually close an environment

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    """
    envs.env_close(instance_id)
    return ("", 200)


def run_main():
    parser = argparse.ArgumentParser(description="Start a Gym HTTP API server")
    parser.add_argument("-l", "--listen", help="interface to listen to", default="127.0.0.1")
    parser.add_argument("-p", "--port", default=40004, type=int, help="port to bind to")
    parser.add_argument("-g", "--log_level", default="ERROR", type=str, help="server log level")

    args = parser.parse_args()
    print(f"Server starting at:  http://{args.listen}:{args.port}. Loglevel: {args.log_level}.")
    logger.setLevel(args.log_level)
    app.run(host=args.listen, port=args.port)


if __name__ == "__main__":
    run_main()
