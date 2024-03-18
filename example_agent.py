import logging
from pathlib import Path
from PIL import Image
import numpy as np
from gym_http_client import Client


class RandomDiscreteAgent:
    def __init__(self, n):
        self.n = n


def save_render_frame(render_frame):
    img = Image.fromarray(np.array(render_frame, dtype=np.uint8))
    img.save(f"{Path(__file__).resolve()}.{j}_{action}.jpg")


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

    EPISODE_COUNT = 1
    MAX_STEPS = 2000
    REWARD = 0
    DONE = False

    for i in range(EPISODE_COUNT):
        ob = client.env_reset(instance_id)
        for j in range(MAX_STEPS):
            action = client.env_action_space_sample(instance_id)
            observation, REWARD, terminated, truncated, info = client.env_step(instance_id, action)
            rf = client.env_render(instance_id)
            save_render_frame(rf)
            if DONE:
                break

    logger.info("Successfully ran example agent using gym_http_client.")
