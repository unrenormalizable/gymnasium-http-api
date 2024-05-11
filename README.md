# gymnasium-http-api - REST API for Gymnasium

[![Discord](https://img.shields.io/discord/1060697970426773584?color=5965F2&label=join%20the%20community)](https://discord.gg/jCMXm7Z34k) [![CDP](https://github.com/unrenormalizable/gymnasium-http-api/actions/workflows/cdp.yml/badge.svg)](https://github.com/unrenormalizable/gymnasium-http-api/actions/workflows/cdp.yml) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?label=license)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

> Stolen from [gym-http-api](https://github.com/openai/gym-http-api) and made to work with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

Contributions welcomed!

## Project

This project is divided into two parts.

### 1. REST API

This parts provides a local REST API to the Gymnasium library, allowing development in languages other than python.

A Rust client (including UI) is the focus along. Here is a demo:

<img src="https://github.com/unrenormalizable/gymnasium-http-api/assets/152241361/20cffcaf-61ba-4fa5-9efa-c39e56866c2c" width="30%" title="FrozenLake-v1 by a Policy Iteration agent written in Rust."/>

### 2. RL from Basics

This part implements the various RL algos, starting from the basics.

> Image from [Theory: Reinforcement Learning by Steve Brunton](https://www.youtube.com/watch?v=0MNVhXEX9to&list=PLMrJAkhIeNNQe1JXNvaFvURxGY4gE9k74)

<img src="https://github.com/unrenormalizable/gymnasium-http-api/assets/152241361/932a7306-2020-4f91-b9d1-de91c352e956" width="50%" title="Reinforcement Learning by Steve Brunton" />
<br>
<br>

So far:

- [x] Extractions for π\* \\ Q\* \\ V\*
- [ ] Model based
  - [x] Value iteration
  - [x] Policy iteration
- [ ] Model free
  - [ ] MC
  - [ ] TD
    - [ ] SARSA
    - [ ] Q-learning
  - [ ] Policy Gradients
- [ ] DQN

## Installation

To download the code and install the requirements, you can run the following shell commands:

    git clone https://github.com/unrenormalizable/gymnasium-http-api
    cd gymnasium-http-api
    pip install -r requirements.txt
    python gym_http_server.py
    # Ensure the python tests run from another terminal
    nose2


## Getting started

This code is intended to be run locally by a single user. The server runs in python. You can implement your own HTTP clients using any language; a demo client written in python is provided to demonstrate the idea.

To start the server from the command line, run this:

In a separate terminal, you can then try running the example rust agent:

    cd rust/client
    cargo run --example mountain_car_gui

## Testing

> For running the Python & Rust client tests, you need the gym_http_server.py started manually as a separate process.
> See cdp.yml on how to do it.

This repository contains integration tests, using the python client implementation to send requests to the local server. They can be run using the `nose2` framework. From a shell (such as bash) you can run nose2 directly:

    cd gymnasium-http-api
    nose2
    cd rust/client
    cargo test

## API specification

> This is not maintained, it is here just to give a rough idea. For the current API refer to the gym_http_server.py.

  * POST `/v1/envs/`
      * Create an instance of the specified environment
      * param: Refer https://gymnasium.farama.org/api/registry/#gymnasium.make
      * returns: `instance_id` -- a short identifier (such as '3c657dbc')
	    for the created environment instance. The instance_id is
        used in future API calls to identify the environment to be
        manipulated

  * GET `/v1/envs/`
      * List all environments running on the server
	  * returns: `envs` -- dict mapping `instance_id` to `env_id`
	    (e.g. `{'3c657dbc': 'CartPole-v1'}`) for every env on the server

  * POST `/v1/envs/<instance_id>/reset/`
      * Reset the state of the environment and return an initial
        observation.
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance
	    * param: `seed` -- the seed that is used to initialize the environment’s
        PRNG, if the environment wasn't already seeded
      * returns: `observation` -- the initial observation of the space

  * POST `/v1/envs/<instance_id>/step/`
      *  Step though an environment using an action.
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance
	    * param: `action` -- an action to take in the environment
      * returns: `observation` -- agent's observation of the current
        environment
      * returns: `reward` -- amount of reward returned after previous action
      * returns: `done` -- whether the episode has ended
      * returns: `info` -- a dict containing auxiliary diagnostic information

  * GET `/v1/envs/<instance_id>/action_space/`
      * Get information (name and dimensions/bounds) of the env's
        `action_space`
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance
      * returns: `info` -- a dict containing 'name' (such as 'Discrete'), and
    additional dimensional info (such as 'n') which varies from
    space to space

  * GET `/v1/envs/<instance_id>/observation_space/`
      * Get information (name and dimensions/bounds) of the env's
        `observation_space`
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance
      * returns: `info` -- a dict containing 'name' (such as 'Discrete'), and
    additional dimensional info (such as 'n') which varies from
    space to space

  * GET `/v1/envs/<instance_id>/transitions/`
      * Get transition probability from state given action.
      * param: `instance_id` -- a short identifier (such as '3c657dbc')
        for the environment instance
      * returns: `transitions` -- all transitions as the tuple (probability of transition, next state, reward, done)

  * DELETE `/v1/envs/<instance_id>/`
      * Removes an environment

## References

- [Theory: Reinforcement Learning: An Introduction by Sutton & Barto](https://lcalem.github.io/blog/2018/09/22/sutton-index)
- [Theory: Reinforcement Learning by Steve Brunton](https://www.youtube.com/watch?v=0MNVhXEX9to&list=PLMrJAkhIeNNQe1JXNvaFvURxGY4gE9k74)
- [Theory: CSCI 531 — Fall 2023](https://people.stfx.ca/jdelamer/courses/csci-531/index.html)
- [Theory: CIS 522 - Deep Learning - Reinforcement Learning](https://www.youtube.com/watch?v=oJo0jb_h2sM&list=PLYgyoWurxA_8ePNUuTLDtMvzyf-YW7im2)
- [Code: Deep Reinforcement Learning With Python](https://github.com/sudharsan13296/Deep-Reinforcement-Learning-With-Python)
- [Toy Example: Reinforcement Learning: an Easy Introduction to Value Iteration](https://towardsdatascience.com/reinforcement-learning-an-easy-introduction-to-value-iteration-e4cfe0731fd5)
- [Theory: Introduction to Reinforcement Learning](https://gibberblot.github.io/rl-notes/intro.html)
- [RL workspace](https://aka.ms/edge/workspaceslaunch?code=dHlwZT0xJmlkPWFIUjBjSE02THk5b2IyMWxMbTFwWTNKdmMyOW1kSEJsY25OdmJtRnNZMjl1ZEdWdWRDNWpiMjB2T25VNkwyY3ZZMjl1ZEdWdWRITjBiM0poWjJVdk5rTjZibWxvVGpOMk1IVjVNWHBMTlVaeVVrTmlkemd4T0RBM05qQmtZVGsxWm1FM05ERjNiM0pyYzNCaFkyVnpMMGxSUzE5SFRYZzVjMGhEY2xOTGN6ZE5kbTlGWVdsWmNFRmhWbXRoTldsTlQzRnlOVk5MWWpFeWFGbDBObXR6JnN0b3JlPTUmc291cmNlPVdvcmtzcGFjZXMmcmVkZWVtQ29kZT1kdW1teV9zZWVkJmFwcElkR3VpZD1iNmQ4MzNjZi1iNTRlLTRjYWItODE0My0xMzE4ZTBiYzUwZTE%3D&source=Workspaces)
