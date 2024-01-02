from typing import Any
from functools import lru_cache

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
import pygame

from blokus_rl._blokus import PyBlokus


class BlokusEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 2,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        seed: int | None = None,
        screen_size: int = 512,
    ):
        super().__init__()
        self._env = PyBlokus()
        self._observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.num_agents, 20, 20),
                        dtype=bool,
                    ),
                    "action_mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self._env.num_actions,),
                        dtype=np.int8,
                    ),
                }
            )
            for i in self.agents
        }
        self._action_spaces = {
            i: spaces.Discrete(self._env.num_actions, seed=seed)
            for i in self.agents
        }
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self._infos = {i: {} for i in self.agents}
        self._render_mode = render_mode
        self._screen = None
        self._screen_size = screen_size
        self._clock = None

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
        return {
            i: self._env.terminations[int(i.split("_")[1])]
            for i in self.agents
        }

    @property
    def truncations(self) -> dict[int, bool]:
        return {
            i: self._env.terminations[int(i.split("_")[1])]
            for i in self.agents
        }

    @property
    def rewards(self) -> dict[int, dict[str, Any]]:
        return {
            i: self._env.rewards[int(i.split("_")[1])] for i in self.agents
        }

    @property
    def infos(self) -> dict[int, bool]:
        return self._infos

    def _render_setup(self):
        if self._render_mode is None:
            return

        if self._render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("blokus-rl")
            self._screen = pygame.display.set_mode(
                (self._screen_size, self._screen_size)
            )
            self._clock = pygame.time.Clock()
            return
        elif self._render_mode == "rgb_array":
            self.screen = pygame.Surface(
                (self._screen_size, self._screen_size)
            )
            return
        raise NotImplementedError

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> None:
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self._info = {i: {} for i in self.agents}
        for a in self._action_spaces:
            self._action_spaces[a].seed(seed)
        self._env.reset()
        self.observe.cache_clear()
        self._render_setup()

    def step(self, action: int) -> None:
        self.observe.cache_clear()
        self._env.step(action)
        self.observe(self.agent_selection)
        if self._render_mode is not None:
            self.render()

    @lru_cache(maxsize=4, typed=True)
    def observe(self, agent: str) -> dict:
        obs = self._env.observe(int(agent.split("_")[1]))
        return {
            "observation": np.array(obs.observation, dtype=bool),
            "action_mask": np.array(obs.action_mask, dtype=np.int8),
        }

    def observation_space(self, agent: str) -> spaces.Dict:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    def render(self) -> None:
        obs = self.observe(self.agents[0])["observation"]

        canvas = pygame.Surface((self._screen_size, self._screen_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self._screen_size / obs.shape[1]

        for i in range(obs.shape[1]):
            for j in range(obs.shape[2]):
                if obs[0, i, j]:
                    color = 85, 85, 255
                elif obs[1, i, j]:
                    color = (255, 255, 85)
                elif obs[2, i, j]:
                    color = (255, 85, 85)
                elif obs[3, i, j]:
                    color = (85, 255, 85)
                else:
                    color = (255, 255, 255)

                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        (pix_square_size * j, pix_square_size * i),
                        (pix_square_size, pix_square_size),
                    ),
                )
        for x in range(obs.shape[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self._screen_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self._screen_size),
                width=3,
            )

        if self._render_mode == "human":
            self._screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata["render_fps"])
        else:  # rgb_array or single_rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        ...
