from functools import partial
import gym
import pogema


def env_fn(name='Pogema-v0', **kwargs):
    if 'grid_config' in kwargs and kwargs['grid_config'] is None:
        del kwargs['grid_config']
    print(kwargs)
    return gym.make(name, **kwargs).unwrapped


REGISTRY = {}

REGISTRY["pogema"] = partial(env_fn, integration='PyMARL')