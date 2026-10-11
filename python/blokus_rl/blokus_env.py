import copy

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
import pygame

from blokus_rl._blokus import PyBlokus

REWARD_MODES = PyBlokus.reward_modes


def _bool_array(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    # read-only view of the 0/1 bytes from Rust
    return np.frombuffer(data, dtype=bool).reshape(shape)


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
        reward_mode: str = "score",
    ):
        if reward_mode not in REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {REWARD_MODES}, "
                f"got {reward_mode!r}"
            )
        super().__init__()
        self._env = PyBlokus()
        self._reward_mode = reward_mode
        self.possible_agents = [f"agent_{i}" for i in self._env.agents]
        num_agents = len(self.possible_agents)
        num_pieces = len(self._env.piece_names)
        self._observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=1,
                        shape=(num_agents, 20, 20),
                        dtype=bool,
                    ),
                    "action_mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self._env.num_actions,),
                        dtype=np.int8,
                    ),
                    "pieces": spaces.Box(
                        low=0,
                        high=1,
                        shape=(num_agents, num_pieces),
                        dtype=bool,
                    ),
                    "corners": spaces.Box(
                        low=0, high=1, shape=(num_agents, 20, 20), dtype=bool
                    ),
                    "blocked": spaces.Box(
                        low=0, high=1, shape=(num_agents, 20, 20), dtype=bool
                    ),
                    "done": spaces.Box(
                        low=0, high=1, shape=(num_agents,), dtype=bool
                    ),
                    "seat": spaces.Discrete(num_agents),
                    "ply": spaces.Box(
                        low=0,
                        high=num_agents * num_pieces,
                        shape=(),
                        dtype=np.int16,
                    ),
                }
            )
            for i in self.possible_agents
        }
        self._action_spaces = {
            i: spaces.Discrete(self._env.num_actions)
            for i in self.possible_agents
        }
        self._seed_action_spaces(seed)
        self._reset_agents()
        self._observations = {}
        self._render_mode = render_mode
        self._screen = None
        self._screen_size = screen_size
        self._clock = None

    @property
    def action_grid_shape(self) -> tuple[int, int, int]:
        """Shape (orientation, row, col) of the action grid, in C order."""
        return self._env.action_grid_shape

    @property
    def piece_names(self) -> tuple[str, ...]:
        """Piece names, in the order of the observation's piece axis."""
        return self._env.piece_names

    @property
    def observation_spaces(self):
        return self._observation_spaces

    @property
    def action_spaces(self):
        return self._action_spaces

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    def _seed_action_spaces(self, seed: int | None) -> None:
        if seed is None:
            return
        # distinct seeds, otherwise all agents sample from the same stream
        for i, space in enumerate(self._action_spaces.values()):
            space.seed(seed + i)

    def _render_setup(self):
        if self._render_mode is None or self._screen is not None:
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
            # render() draws onto its own canvas
            return
        raise NotImplementedError

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> None:
        self._seed_action_spaces(seed)
        self._env.reset()
        self._reset_agents()
        self._observations.clear()
        self._render_setup()

    def _reset_agents(self) -> None:
        self.agents = self.possible_agents[:]
        self.agent_selection = f"agent_{self._env.agent_selection}"
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            # the agent leaves, the game keeps passing its turn
            self._was_dead_step(action)
            self._env.step(None)
        else:
            self._env.step(action)
            self._cumulative_rewards[agent] = 0
            self._clear_rewards()
        self._select_next_agent()
        if self._render_mode == "human":
            self.render()
        self._accumulate_rewards()

    def _select_next_agent(self) -> None:
        # agents that left pass until the game is over
        self._observations.clear()
        while self.agents:
            idx = self._env.agent_selection
            agent = f"agent_{idx}"
            if agent not in self.agents:
                self._env.step(None)
                continue
            if self._env.terminations[idx] and not self.terminations[agent]:
                if self._reward_mode == "score":
                    # an agent never regains moves, its score is final
                    self.terminations[agent] = True
                    self.rewards[agent] = self._env.rewards[idx]
                elif self._env.is_over:
                    # the outcome is only known now, all agents get it
                    outcome = self._env.outcome(self._reward_mode)
                    for a in self.agents:
                        self.terminations[a] = True
                        self.rewards[a] = outcome[int(a.split("_")[1])]
                else:
                    # out of moves before game end, its turns are skipped
                    self._env.step(None)
                    continue
            if self._env.is_over:
                scores = dict(zip(self.possible_agents, self._env.scores))
                for a in self.agents:
                    self.infos[a]["scores"] = dict(scores)
            self.agent_selection = agent
            return

    def observe(self, agent: str) -> dict:
        # cached until the next step, so the arrays are read-only
        if agent not in self._observations:
            idx = int(agent.split("_")[1])
            obs = self._env.observe(idx)
            num_agents = len(self.possible_agents)
            ply = np.array(self._env.ply, dtype=np.int16)
            ply.flags.writeable = False
            self._observations[agent] = {
                "observation": _bool_array(
                    obs.observation, (num_agents, 20, 20)
                ),
                # read-only view of the bytes from Rust
                "action_mask": np.frombuffer(obs.action_mask, dtype=np.int8),
                "pieces": _bool_array(obs.pieces, (num_agents, -1)),
                "corners": _bool_array(obs.corners, (num_agents, 20, 20)),
                "blocked": _bool_array(obs.blocked, (num_agents, 20, 20)),
                "done": _bool_array(obs.done, (num_agents,)),
                "seat": idx,
                "ply": ply,
            }
        return dict(self._observations[agent])

    def __getstate__(self) -> dict:
        # observations are recomputed; a copy does not render, e.g. when it
        # branches off a tree search
        state = self.__dict__.copy()
        state.update(
            _observations={}, _render_mode=None, _screen=None, _clock=None
        )
        return state

    def __copy__(self) -> "BlokusEnv":
        # a shallow copy would step the same game
        return copy.deepcopy(self)

    def __deepcopy__(self, memo: dict) -> "BlokusEnv":
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        # the observation spaces never change, so the copy shares them
        memo[id(self._observation_spaces)] = self._observation_spaces
        clone.__dict__.update(copy.deepcopy(self.__getstate__(), memo))
        return clone

    def render(self) -> np.ndarray | None:
        obs = self.observe(self.possible_agents[0])["observation"]

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
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
