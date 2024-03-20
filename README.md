# gymnasium-http-api - REST API for Gymnasium

[![CDP](https://github.com/unrenormalizable/gymnasium-http-api/actions/workflows/cdp.yml/badge.svg)](https://github.com/unrenormalizable/gymnasium-http-api/actions/workflows/cdp.yml) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?label=license)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

> Stolen from [gym-http-api](https://github.com/openai/gym-http-api) and made to work with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium). Contributions welcomed!

This project provides a local REST API to the Gymnasium open-source library, allowing development in languages other than python.

A python client is included, to demonstrate how to interact with the server.

Contributions for clients in other languages are welcomed!

## Installation

To download the code and install the requirements, you can run the following shell commands:

    git clone https://github.com/unrenormalizable/gymnasium-http-api
    cd gymnasium-http-api
    pip install -r requirements.txt

## Getting started

This code is intended to be run locally by a single user. The server runs in python. You can implement your own HTTP clients using any language; a demo client written in python is provided to demonstrate the idea.

To start the server from the command line, run this:

    python gym_http_server.py

In a separate terminal, you can then try running the example python agent and see what happens:

    python example_agent.py

The example rust agent behaves very similarly:

    cd binding-rust
    cargo run

You can also write code like this to create your own client, and test it out by creating a new environment. For example, in python:

    remote_base = 'http://127.0.0.1:5000'
    client = Client(remote_base)

    env_id = 'CartPole-v1'
    instance_id = client.env_create(env_id)
    client.env_step(instance_id, 0)


## Testing

This repository contains integration tests, using the python client implementation to send requests to the local server. They can be run using the `nose2` framework. From a shell (such as bash) you can run nose2 directly:

    cd gymnasium-http-api
    nose2

## API specification

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

  * DELETE `/v1/envs/<instance_id>`
      * Removes an environment


Contributors
============

  * unrenormalizable
