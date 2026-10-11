import copy
from functools import partial

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


def deepcopy_test(seed: int) -> None:
    # a copy plays on independently, with the original's action sampling
    env = BlokusEnv(seed=seed)
    env.reset(seed=seed)
    for agent in env.agent_iter(max_iter=20):
        obs, _, termination, truncation, _ = env.last()
        env.step(env.action_space(agent).sample(obs["action_mask"]))
    state = env._env.to_bytes()
    mask = env.observe(env.agent_selection)["action_mask"].copy()
    clone = copy.deepcopy(env)
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
    try:
        PyBlokus.from_bytes(state[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("from_bytes() accepted a truncated state")


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
        deepcopy_test(seed)
    bytes_test()
    print("All tests passed")
