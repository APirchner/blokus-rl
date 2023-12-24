from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from _blokus import PyBlokus


class BlokusEnv(AECEnv):
    def __init__(self):
        super().__init__()
        self._env = PyBlokus()
        self._observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(20, 20, self.num_agents), dtype=bool
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self._env.num_actions,), dtype=np.int8
                    ),
                }
            )
            for i in self.agents
        }
        self._action_spaces = {
            i: spaces.Discrete(self._env.num_actions) for i in self.agents
        }
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self._infos = {i: {} for i in self.agents}

    @property
    def agents(self) -> list[str]:
        return [f"agent_{i}" for i in self._env.agents]

    @property
    def possible_agents(self) -> list[str]:
        return self.agents

    @property
    def agent_selection(self) -> int:
        return f"agent_{self._env.agent_selection}"

    @property
    def terminations(self) -> dict[int, bool]:
        return {i: self._env.terminations[int(i.split("_")[1])] for i in self.agents}

    @property
    def truncations(self) -> dict[int, bool]:
        return {i: self._env.terminations[int(i.split("_")[1])] for i in self.agents}

    @property
    def rewards(self) -> dict[int, dict[str, Any]]:
        return {i: self._env.rewards[int(i.split("_")[1])] for i in self.agents}

    @property
    def infos(self) -> dict[int, bool]:
        return self._infos

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self._info = {i: {} for i in self.agents}
        self._env.reset()

    def step(self, action: int) -> None:
        self._env.step(action)

    def observe(self, agent: str) -> dict | None:
        obs = self._env.observe(int(agent.split("_")[1]))
        return {
            "observation": np.einsum("ijk->jki", np.array(obs.observation, dtype=bool)),
            "action_mask": np.array(obs.action_mask, dtype=np.int8),
        }

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    def render(self) -> None:
        self._env.render()

    def close(self) -> None:
        ...


if __name__ == "__main__":
    env = BlokusEnv()
    env.reset()
    for i, agent in enumerate(env.agent_iter()):
        observation, reward, termination, truncation, info = env.last()
        action = env.action_space(agent).sample(mask=observation["action_mask"])
        env.step(action)
        env.render()
        if all([t[1] for t in env.terminations.items()]):
            break
    print(env.rewards)
