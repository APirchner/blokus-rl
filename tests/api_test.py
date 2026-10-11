from pettingzoo.test import api_test, seed_test
from blokus_rl.blokus_env import BlokusEnv


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


if __name__ == "__main__":
    api_test(BlokusEnv(), num_cycles=10, verbose_progress=True)
    seed_test(BlokusEnv, num_cycles=10)
    for seed in range(3):
        full_game_test(seed)
