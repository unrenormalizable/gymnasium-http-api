import logging

from gym_http_client import Client


class RandomDiscreteAgent:
    def __init__(self, n):
        self.n = n


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Set up client
    REMOTE_BASE = "http://127.0.0.1:5000"
    client = Client(REMOTE_BASE)

    # Set up environment
    ENV_ID = "CartPole-v1"
    instance_id = client.env_create(ENV_ID)

    # Set up agent
    action_space_info = client.env_action_space_info(instance_id)
    agent = RandomDiscreteAgent(action_space_info["n"])

    # Run experiment, with monitor
    OUTDIR = "/tmp/random-agent-results"
    client.env_monitor_start(instance_id, OUTDIR, force=True, resume=False, video_callable=False)

    EPISODE_COUNT = 100
    MAX_STEPS = 200
    REWARD = 0
    DONE = False

    for i in range(EPISODE_COUNT):
        ob = client.env_reset(instance_id)

        for j in range(MAX_STEPS):
            action = client.env_action_space_sample(instance_id)
            observation, REWARD, terminated, truncated, info = client.env_step(instance_id, action, render=True)
            if DONE:
                break

    # Dump result info to disk
    client.env_monitor_close(instance_id)

    # Upload to the scoreboard. This expects the 'OPENAI_GYM_API_KEY'
    # environment variable to be set on the client side.
    logger.info(
        """Successfully ran example agent using
        gym_http_client. Now trying to upload results to the
        scoreboard. If this fails, you likely need to set
        os.environ['OPENAI_GYM_API_KEY']=<your_api_key>"""
    )

    client.upload(OUTDIR)
