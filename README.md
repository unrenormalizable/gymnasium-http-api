# gymnasium-http-api - REST API for Gymnasium

[![Discord](https://img.shields.io/discord/1060697970426773584?color=5965F2&label=join%20the%20community)](https://discord.gg/jCMXm7Z34k) [![CDP](https://github.com/unrenormalizable/gymnasium-http-api/actions/workflows/cdp.yml/badge.svg)](https://github.com/unrenormalizable/gymnasium-http-api/actions/workflows/cdp.yml) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?label=license)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

> Stolen from [gym-http-api](https://github.com/openai/gym-http-api) and made to work with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium). Contributions welcomed!

This project provides a local REST API to the Gymnasium open-source library, allowing development in languages other than python.

A Rust client (including UI) is the focus along with various RL algorithms, written in Rust.

Contributions welcome!

## Demo

<img src="https://github.com/unrenormalizable/gymnasium-http-api/assets/152241361/20cffcaf-61ba-4fa5-9efa-c39e56866c2c" width="30%" title="FrozenLake-v1 by a Policy Iteration agent written in Rust."/>

## Installation

To download the code and install the requirements, you can run the following shell commands:

    git clone https://github.com/unrenormalizable/gymnasium-http-api
    cd gymnasium-http-api
    pip install -r requirements.txt

## Getting started

This code is intended to be run locally by a single user. The server runs in python. You can implement your own HTTP clients using any language; a demo client written in python is provided to demonstrate the idea.

To start the server from the command line, run this:

    python gym_http_server.py
    # Ensure the python tests run
    nose2

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

> This is not maintained, it is here just to give a rough idea
> For the current API refer to the gym_http_server.py.

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


Contributors
============

  * unrenormalizable
