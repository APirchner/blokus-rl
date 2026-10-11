import copy
import pickle
from functools import partial

import numpy as np

from pettingzoo.test import api_test, seed_test
from blokus_rl._blokus import PyBlokus
from blokus_rl.blokus_env import REWARD_MODES, BlokusEnv


def full_game_test(seed: int) -> None:
    # play a random game to the end, each agent collects its score exactly once
    env = BlokusEnv(seed=seed)
    env.reset(seed=seed)
    returns = {i: 0 for i in env.possible_agents}
    left_early = False
    for agent in env.agent_iter(max_iter=10_000):
        obs, reward, termination, truncation, _ = env.last()
        returns[agent] += reward
        if termination or truncation:
            # the score arrives as soon as the agent is out of moves
            assert reward == env._env.rewards[int(agent.split("_")[1])]
            left_early |= not all(env._env.terminations)
            action = None
        else:
            action = env.action_space(agent).sample(obs["action_mask"])
        env.step(action)
    assert not env.agents, "agent_iter() did not end"
    assert left_early, "no agent left before the game was over"
    scores = dict(zip(env.possible_agents, env._env.rewards))
    assert returns == scores, f"returns {returns} != scores {scores}"


# piece sizes in the order of env.piece_names
PIECE_SIZES = np.array([1, 2, 3, 3] + [4] * 5 + [5] * 12)


def observation_test(seed: int) -> None:
    # the observation's parts agree with each other at every ply
    env = BlokusEnv(seed=seed, reward_mode="win_share")
    env.reset(seed=seed)
    for agent in env.agent_iter(max_iter=10_000):
        obs, _, termination, truncation, _ = env.last()
        placed = obs["observation"].sum(axis=(1, 2))
        assert (placed == PIECE_SIZES @ ~obs["pieces"].T).all()
        assert not (obs["corners"] & obs["blocked"]).any()
        assert (obs["blocked"] >= obs["observation"].any(axis=0)).all()
        assert obs["done"][0] == (obs["action_mask"].sum() == 0)
        assert obs["seat"] == int(agent.split("_")[1])
        assert obs["ply"] == env._env.ply
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample(obs["action_mask"])
        env.step(action)


def competitive_game_test(seed: int, reward_mode: str) -> None:
    # all agents get the outcome exactly once and leave together at game end
    env = BlokusEnv(seed=seed, reward_mode=reward_mode)
    env.reset(seed=seed)
    returns = {i: 0 for i in env.possible_agents}
    for agent in env.agent_iter(max_iter=10_000):
        obs, reward, termination, truncation, info = env.last()
        returns[agent] += reward
        if termination or truncation:
            assert env._env.is_over, "an agent left before the game was over"
            scores = dict(zip(env.possible_agents, env._env.scores))
            assert info["scores"] == scores
            action = None
        else:
            assert reward == 0, "a reward arrived before the game was over"
            action = env.action_space(agent).sample(obs["action_mask"])
        env.step(action)
    assert not env.agents, "agent_iter() did not end"
    outcome = dict(zip(env.possible_agents, env._env.outcome(reward_mode)))
    assert returns == outcome, f"returns {returns} != outcome {outcome}"
    if reward_mode == "win_share":
        assert abs(sum(returns.values()) - 1) < 1e-6


def pickle_copy(obj):
    return pickle.loads(pickle.dumps(obj))


COPIES = (copy.copy, copy.deepcopy, pickle_copy)


def copy_test(seed: int, copy_env) -> None:
    # a copy plays on independently, with the original's action sampling
    env = BlokusEnv(seed=seed)
    env.reset(seed=seed)
    for agent in env.agent_iter(max_iter=20):
        obs, _, termination, truncation, _ = env.last()
        env.step(env.action_space(agent).sample(obs["action_mask"]))
    state = env._env.to_bytes()
    mask = env.observe(env.agent_selection)["action_mask"].copy()
    clone = copy_env(env)
    histories = []
    for e in (clone, env):
        history = []
        for agent in e.agent_iter(max_iter=10_000):
            obs, reward, termination, truncation, _ = e.last()
            if e is clone:
                # the original stays where it was copied
                assert env._env.to_bytes() == state
                assert (
                    env.observe(env.agent_selection)["action_mask"] == mask
                ).all()
            if termination or truncation:
                action = None
            else:
                action = e.action_space(agent).sample(obs["action_mask"])
            history.append((agent, reward, action))
            e.step(action)
        histories.append(history)
    assert histories[0] == histories[1]


def game_copy_test() -> None:
    # copies of the game play on independently
    game = PyBlokus()
    state = game.to_bytes()
    for clone in [game.copy()] + [copy_game(game) for copy_game in COPIES]:
        assert clone.to_bytes() == state
        clone.step(clone.observe(0).action_mask.index(1))
        assert clone.to_bytes() != state
        assert game.to_bytes() == state
    # the observation spaces never change, so copies share them
    env = BlokusEnv()
    assert copy.deepcopy(env).observation_spaces is env.observation_spaces


def bytes_test() -> None:
    env = BlokusEnv(seed=0)
    env.reset(seed=0)
    for agent in env.agent_iter(max_iter=30):
        obs, _, _, _, _ = env.last()
        env.step(env.action_space(agent).sample(obs["action_mask"]))
    state = env._env.to_bytes()
    restored = PyBlokus.from_bytes(state)
    assert restored.to_bytes() == state
    for i in range(4):
        assert (
            restored.observe(i).action_mask == env._env.observe(i).action_mask
        )
    # a truncated state, a state of another format version
    for invalid in (state[:-1], bytes([state[0] + 1]) + state[1:]):
        try:
            PyBlokus.from_bytes(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("from_bytes() accepted an invalid state")


def invalid_arguments_test() -> None:
    game = PyBlokus()
    for call in (partial(game.observe, 4), partial(game.outcome, "wins")):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError(f"{call} accepted an invalid argument")


if __name__ == "__main__":
    for mode in REWARD_MODES:
        api_test(
            BlokusEnv(reward_mode=mode), num_cycles=100, verbose_progress=True
        )
        seed_test(partial(BlokusEnv, reward_mode=mode), num_cycles=10)
    for seed in range(3):
        full_game_test(seed)
        for mode in REWARD_MODES[1:]:
            competitive_game_test(seed, mode)
        for copy_env in COPIES:
            copy_test(seed, copy_env)
        observation_test(seed)
    game_copy_test()
    bytes_test()
    invalid_arguments_test()
    print("All tests passed")
