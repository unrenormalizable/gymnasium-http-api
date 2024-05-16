import logging

import gym_http_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

########## CONFIGURATION ##########

HOST = "127.0.0.1"
PORT = "40004"
# pylint: disable=C0103
server_process = None


def get_remote_base():
    return f"http://{HOST}:{PORT}"


def with_server(fn):
    return fn


########## TESTS ##########

##### Valid use cases #####


@with_server
def test_create_destroy():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create("CartPole-v1")
    assert instance_id in client.env_list_all()
    client.env_close(instance_id)
    assert instance_id not in client.env_list_all()


@with_server
def test_get_env_id():
    client = gym_http_client.Client(get_remote_base())
    _id = client.env_create("CartPole-v1")
    _id = client.env_id(_id)
    assert _id == "CartPole-v1"


@with_server
def test_action_space_discrete():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create("CartPole-v1")
    action_info = client.env_action_space_info(instance_id)
    assert action_info["name"] == "Discrete"
    assert action_info["n"] == 2


@with_server
def test_action_space_sample():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create("CartPole-v1")
    action = client.env_action_space_sample(instance_id)
    assert 0 <= action < 2


@with_server
def test_episode_samples():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create("FrozenLake-v1", kwargs={"desc": ["SFFF", "FFFF", "FFFF", "HFFG"]})
    eps = client.env_episode_samples(instance_id, 10, 2718)
    assert len(eps) == 10


@with_server
def test_action_space_contains():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create("CartPole-v1")
    action_info = client.env_action_space_info(instance_id)
    assert action_info["n"] == 2
    assert client.env_action_space_contains(instance_id, 0) is True
    assert client.env_action_space_contains(instance_id, 1) is True
    assert client.env_action_space_contains(instance_id, 2) is False


@with_server
def test_observation_space_box():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create("CartPole-v1")
    obs_info = client.env_observation_space_info(instance_id)
    assert obs_info["name"] == "Box"
    assert len(obs_info["shape"]) == 1
    assert obs_info["shape"][0] == 4
    assert len(obs_info["low"]) == 4
    assert len(obs_info["high"]) == 4


@with_server
def test_observation_space_contains():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create("CartPole-v1")
    obs_info = client.env_observation_space_info(instance_id)
    assert obs_info["name"] == "Box"
    assert client.env_observation_space_contains(instance_id, {"name": "Box"})
    assert client.env_observation_space_contains(instance_id, {"shape": (4,)})
    assert client.env_observation_space_contains(instance_id, {"name": "Box", "shape": (4,)})


@with_server
def test_reset():
    client = gym_http_client.Client(get_remote_base())

    instance_id = client.env_create("CartPole-v1")
    init_obs = client.env_reset(instance_id)
    assert len(init_obs) == 4

    instance_id = client.env_create("FrozenLake-v1")
    init_obs = client.env_reset(instance_id)
    assert len(init_obs) == 1
    assert init_obs[0] == 0


@with_server
def test_step():
    client = gym_http_client.Client(get_remote_base())

    instance_id = client.env_create("CartPole-v1")
    client.env_reset(instance_id)
    action = client.env_action_space_sample(instance_id)
    observation, reward, terminated, truncated, info = client.env_step(instance_id, action)
    assert len(observation) == 4
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    instance_id = client.env_create("Ant-v4")
    client.env_reset(instance_id)
    action = client.env_action_space_sample(instance_id)
    observation, reward, terminated, truncated, info = client.env_step(instance_id, action)
    assert len(observation) == 27
    assert isinstance(observation[0], float)


@with_server
def test_render_ansi():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create("FrozenLake-v1", kwargs={"render_mode": "ansi"})
    client.env_reset(instance_id)
    rf = client.env_render(instance_id)
    assert rf == "\n\x1b[41mS\x1b[0mFFF\nFHFH\nFFFH\nHFFG\n"


@with_server
def test_render_rgb():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create("MountainCar-v0", kwargs={"render_mode": "rgb_array"})
    client.env_reset(instance_id)
    rf = client.env_render(instance_id)
    assert rf["rows"] == 400
    assert rf["cols"] == 600
    assert len(rf["data"]) == 1280000


@with_server
def test_get_transitions():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create("FrozenLake-v1")
    client.env_reset(instance_id)
    osi = client.env_observation_space_info(instance_id)
    transitions = client.env_get_transitions(instance_id)
    assert len(transitions) == osi["n"]


@with_server
def test_close_env():
    client = gym_http_client.Client(get_remote_base())
    instance_id = client.env_create("FrozenLake-v1")
    envs = client.env_list_all()
    assert instance_id in envs
    client.env_reset(instance_id)
    client.env_close(instance_id)
    envs = client.env_list_all()
    assert instance_id not in envs


##### API usage errors #####


@with_server
def test_bad_instance_id():
    """Test all methods that use instance_id with an invalid ID"""
    client = gym_http_client.Client(get_remote_base())
    try_these = [
        client.env_reset,
        lambda x: client.env_step(x, 1),
        client.env_action_space_info,
        client.env_observation_space_info,
        client.env_close,
    ]
    for call in try_these:
        try:
            call("bad_id")
        except gym_http_client.ServerError as e:
            assert "Instance_id" in e.message
            assert e.status_code == 400
        else:
            assert False


@with_server
def test_missing_param_env_id():
    """Test client failure to provide JSON param: env_id"""

    class BadClient(gym_http_client.Client):
        def env_create(self, env_id_, max_episode_steps=None, auto_reset=None, disable_env_checker=None, kwargs=None):
            route = "/v1/envs/"
            data = {}  # deliberately omit env_id
            resp = self._post_request(route, data)
            instance_id = resp.json()["instance_id"]
            return instance_id

    client = BadClient(get_remote_base())
    try:
        client.env_create("CartPole-v1")
    except gym_http_client.ServerError as e:
        assert "env_id" in e.message
        assert e.status_code == 400
    else:
        assert False


@with_server
def test_missing_param_action():
    """Test client failure to provide JSON param: action"""

    class BadClient(gym_http_client.Client):
        def env_step(self, instance_id_, action):
            route = f"/v1/envs/{instance_id_}/step/"
            data = {}  # deliberately omit action
            resp = self._post_request(route, data)
            observation = resp.json()["observation"]
            reward = resp.json()["reward"]
            done = resp.json()["done"]
            info = resp.json()["info"]
            return [observation, reward, done, info]

    client = BadClient(get_remote_base())

    instance_id = client.env_create("CartPole-v1")
    client.env_reset(instance_id)
    try:
        client.env_step(instance_id, 1)
    except gym_http_client.ServerError as e:
        assert "action" in e.message
        assert e.status_code == 400
    else:
        assert False


##### Gym-side errors #####


@with_server
def test_create_malformed():
    client = gym_http_client.Client(get_remote_base())
    try:
        client.env_create("bad string")
    except gym_http_client.ServerError as e:
        assert "malformed environment ID" in e.message
        assert e.status_code == 400
    else:
        assert False
