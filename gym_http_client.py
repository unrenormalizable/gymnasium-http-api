import json
import six.moves.urllib.parse as urlparse
import logging
import requests


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Client:
    """
    Gym client to interface with gym_http_server
    """

    def __init__(self, remote_base_):
        self.remote_base = remote_base_
        self.session = requests.Session()
        self.session.headers.update({"Content-type": "application/json"})

    def _parse_server_error_or_raise_for_status(self, resp):
        j = {}
        try:
            j = resp.json()
        except:
            # Most likely json parse failed because of network error, not server error (server
            # sends its errors in json). Don't let parse exception go up, but rather raise default
            # error.
            resp.raise_for_status()
        if resp.status_code != 200 and "message" in j:  # descriptive message from server side
            raise ServerError(message=j["message"], status_code=resp.status_code)
        resp.raise_for_status()
        return j

    def _delete_request(self, route):
        url = urlparse.urljoin(self.remote_base, route)
        logger.info("DELETE %s", url)
        resp = self.session.delete(urlparse.urljoin(self.remote_base, route))
        return self._parse_server_error_or_raise_for_status(resp)

    def _post_request(self, route, data):
        url = urlparse.urljoin(self.remote_base, route)
        logger.info("POST %s\n%s", url, json.dumps(data))
        resp = self.session.post(urlparse.urljoin(self.remote_base, route), data=json.dumps(data))
        return self._parse_server_error_or_raise_for_status(resp)

    def _get_request(self, route):
        url = urlparse.urljoin(self.remote_base, route)
        logger.info("GET %s", url)
        resp = self.session.get(url)
        return self._parse_server_error_or_raise_for_status(resp)

    def env_create(self, env_id_, max_episode_steps=None, auto_reset=None, disable_env_checker=None, kwargs=None):
        route = "/v1/envs/"
        data = {
            "env_id": env_id_,
            "max_episode_steps": max_episode_steps,
            "auto_reset": auto_reset,
            "disable_env_checker": disable_env_checker,
            "kwargs": kwargs or {},
        }
        resp = self._post_request(route, data)
        instance_id_ = resp["instance_id"]
        return instance_id_

    def env_list_all(self):
        route = "/v1/envs/"
        resp = self._get_request(route)
        all_envs_ = resp["all_envs"]
        return all_envs_

    def env_reset(self, instance_id_, seed=None):
        route = f"/v1/envs/{instance_id_}/reset/"
        data = {"seed": seed if seed is not None else ""}
        resp = self._post_request(route, data)
        observation_ = resp["observation"]
        return observation_

    def env_render(self, instance_id_):
        route = f"/v1/envs/{instance_id_}/render/"
        resp = self._get_request(route)
        render_frame = resp["render_frame"]
        return render_frame

    def env_step(self, instance_id_, action):
        route = f"/v1/envs/{instance_id_}/step/"
        data = {"action": action}
        resp = self._post_request(route, data)
        observation_ = resp["observation"]
        reward_ = resp["reward"]
        terminated_ = resp["terminated"]
        truncated_ = resp["truncated"]
        info_ = resp["info"]
        return [observation_, reward_, terminated_, truncated_, info_]

    def env_action_space_info(self, instance_id_):
        route = f"/v1/envs/{instance_id_}/action_space/"
        resp = self._get_request(route)
        _info = resp["info"]
        return _info

    def env_action_space_sample(self, instance_id_):
        route = f"/v1/envs/{instance_id_}/action_space/sample"
        resp = self._get_request(route)
        action = resp["action"]
        return action

    def env_action_space_contains(self, instance_id_, x):
        route = f"/v1/envs/{instance_id_}/action_space/contains/{x}"
        resp = self._get_request(route)
        member = resp["member"]
        return member

    def env_observation_space_info(self, instance_id_):
        route = f"/v1/envs/{instance_id_}/observation_space/"
        resp = self._get_request(route)
        info_ = resp["info"]
        return info_

    def env_get_transitions(self, instance_id_):
        route = f"/v1/envs/{instance_id_}/transitions/"
        resp = self._get_request(route)
        probs = resp["transitions"]
        return probs

    def env_observation_space_contains(self, instance_id_, params):
        route = f"/v1/envs/{instance_id_}/observation_space/contains"
        resp = self._post_request(route, params)
        member = resp["member"]
        return member

    def env_close(self, instance_id_):
        route = f"/v1/envs/{instance_id_}/"
        self._delete_request(route)


class ServerError(Exception):
    def __init__(self, message, status_code=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


if __name__ == "__main__":
    REMOTE_BASE = "http://127.0.0.1:5000"
    client = Client(REMOTE_BASE)

    # Create environment
    ENV_ID = "CartPole-v1"
    instance_id = client.env_create(
        ENV_ID, max_episode_steps=100, auto_reset=False, disable_env_checker=True, kwargs={"render_mode": "rgb_array"}
    )

    # Check properties
    all_envs = client.env_list_all()
    action_info = client.env_action_space_info(instance_id)
    obs_info = client.env_observation_space_info(instance_id)

    # Run a single step
    init_obs = client.env_reset(instance_id)
    [observation, reward, terminated, truncated, info] = client.env_step(instance_id, 1)
