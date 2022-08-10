from typing import (
    Dict, Optional, Union, Set, Tuple
)
import bisect

import numpy as np
import scipy.signal as signal
import gym
from gym.core import ObsType
from gym.utils.renderer import Renderer

from blokus_rl.components.pieces import (
    build_piece_set, total_score
)
from blokus_rl.envs.constants import (
    BOARD_SIZE, NUM_PIECE_SETS
)


class BlokusEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'single_rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode: Optional[str] = None):
        assert render_mode is None or render_mode in self.metadata['render_modes']
        super(BlokusEnv, self).__init__()
        self.render_mode = render_mode
        self.window_size = 512
        self.piece_set, self._num_actions = build_piece_set()
        self._piece_indices = sorted(self.piece_set.keys())
        self._board = self._reset_board()
        self._scores = self._reset_scores()
        self._pieces_played = self._reset_pieces_played()
        self._action_masks = self._reset_action_masks()
        self._obs_space = gym.spaces.Dict({
            'board': gym.spaces.Box(low=0, high=4, shape=self._board.shape, dtype=int),
            'pieces_played': gym.spaces.MultiBinary(self._pieces_played.shape),
            'action_masks': gym.spaces.MultiBinary(self._action_masks.shape),
            'scores': gym.spaces.MultiDiscrete(self._scores)
        })
        self.action_space = gym.spaces.Dict({
            'piece_set': gym.spaces.Discrete(NUM_PIECE_SETS, start=1),
            'action_idx': gym.spaces.Discrete(self._num_actions)
        })

        if self.render_mode == 'human':
            import pygame
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.render_mode, self._render_frame)

    @classmethod
    def _reset_board(self) -> np.array:
        return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.dtype('u1'))

    @classmethod
    def _reset_scores(cls) -> np.array:
        return np.array(NUM_PIECE_SETS * [total_score()], dtype='u1')

    def _reset_pieces_played(self) -> np.array:
        return np.zeros((NUM_PIECE_SETS, len(self.piece_set)), dtype='u1')

    def _reset_action_masks(self) -> np.array:
        return np.ones((NUM_PIECE_SETS, self._num_actions), dtype=np.int8)

    def _init_action_masks(self):
        init_moves = []
        for i in range(self._num_actions):
            piece_idx = bisect.bisect(self._piece_indices, i) - 1
            x, y, op = self.piece_set[self._piece_indices[piece_idx]].map_action_idx_to_action(i)
            piece_shape = self.piece_set[self._piece_indices[piece_idx]].shape
            piece_shape = op(piece_shape) if op else piece_shape
            if x == 0 and y == 0 and piece_shape[0, 0] == 1:
                init_moves.append(1)
            elif x == 0 and y == BOARD_SIZE - piece_shape.shape[1] and piece_shape[0, piece_shape.shape[1] - 1] == 1:
                init_moves.append(2)
            elif x == BOARD_SIZE - piece_shape.shape[0] and y == BOARD_SIZE - piece_shape.shape[1] and piece_shape[
                piece_shape.shape[0] - 1, piece_shape.shape[1] - 1] == 1:
                init_moves.append(3)
            elif x == BOARD_SIZE - piece_shape.shape[0] and y == 0 and piece_shape[piece_shape.shape[0] - 1, 0] == 1:
                init_moves.append(4)
            else:
                init_moves.append(0)
        init_moves = np.array(init_moves, dtype=np.int8)
        for i in range(NUM_PIECE_SETS):
            self._action_masks[i] = np.where(init_moves == i + 1, 1, 0)

    def _detect_valid_positions(self, piece_set_mask_self: np.array) -> Tuple[np.array, np.array]:
        shift_up = np.roll(piece_set_mask_self, -1, axis=0)
        shift_up[BOARD_SIZE - 1] = 1
        shift_down = np.roll(piece_set_mask_self, 1, axis=0)
        shift_down[0] = 1
        shift_left = np.roll(piece_set_mask_self, -1, axis=1)
        shift_left[:, BOARD_SIZE - 1] = 1
        shift_right = np.roll(piece_set_mask_self, 1, axis=1)
        shift_right[:, 0] = 1
        shift_sum = piece_set_mask_self & shift_up & shift_down & shift_left & shift_right
        conv_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        shift_neigh_sums = signal.convolve2d(shift_sum + piece_set_mask_self - 1, conv_kernel, fillvalue=1)[1:-1, 1:-1]
        valid_positions = np.where((shift_neigh_sums <= 4) & (shift_sum > 0), 1, 0)
        return np.where(valid_positions == 1), shift_sum

    def _update_action_masks(self):
        for i in range(NUM_PIECE_SETS):
            for j, p in enumerate(self._pieces_played[i]):
                if p == 1:
                    self._action_masks[i, self._piece_indices[j]:(
                        self._piece_indices[j + 1] if j < self._pieces_played.shape[1] - 1 else self._num_actions
                    )] = 0
            piece_set_mask_self = np.where(self._board == i + 1, 0, 1)
            piece_set_mask_other = np.where((self._board > 0) & (self._board != i + 1), 0, 1)
            valid_positions, shift_mask = self._detect_valid_positions(piece_set_mask_self)
            for a_idx in np.where(self._action_masks[i] == 1)[0]:
                piece_idx = bisect.bisect(self._piece_indices, a_idx) - 1
                action_board = self.piece_set[self._piece_indices[piece_idx]].map_action_idx_to_board(a_idx)
                if not (action_board[valid_positions] == 1).any() or ((1 - shift_mask + action_board) > 1).any():
                    self._action_masks[i, a_idx] = 0
                elif not ((action_board | piece_set_mask_other) == piece_set_mask_other).all():
                    self._action_masks[i, a_idx] = 0
                else:
                    self._action_masks[i, a_idx] = 1


    def _get_obs(self) -> Dict[str, np.array]:
        return {
            'board': self._board,
            'pieces_played': self._pieces_played,
            'action_masks': self._action_masks,
            'scores': self._scores
        }

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        self._board = self._reset_board()
        self._scores = self._reset_scores()
        self._pieces_played = self._reset_pieces_played()
        self._action_masks = self._reset_action_masks()
        self._init_action_masks()
        return (self._get_obs(), None) if return_info else self._get_obs()

    def step(self, action):
        piece_set_idx = action['piece_set'] - 1
        piece_idx = bisect.bisect(self._piece_indices, action['action_idx']) - 1
        self._board += self.piece_set[
                           self._piece_indices[piece_idx]
                       ].map_action_idx_to_board(action['action_idx']) * action['piece_set']

        self._pieces_played[piece_set_idx, piece_idx] = 1
        reward = self.piece_set[self._piece_indices[piece_idx]].shape.sum(axis=(0, 1))
        self._scores[piece_set_idx] -= reward

        self.renderer.render_step()

        if (self._pieces_played.sum(axis=1) > 0).all():
            # inital round done
            self._action_masks = self._reset_action_masks()
            self._update_action_masks()

        if (self._pieces_played[piece_set_idx] == 1).all():
            done = True
        elif (self._action_masks[piece_set_idx] == 1).all():
            done = True
            reward = -self._scores[piece_set_idx]
        else:
            done = False

        return self._get_obs(), reward, done, None

    def render(self):
        # Just return the list of render frames collected by the Renderer.
        return self.renderer.get_renders()

    def _render_frame(self, mode: str):
        # This will be the function called by the Renderer to collect a single frame.
        assert mode is not None  # The renderer will not call this function with no-rendering.

        import pygame  # avoid global pygame dependency. This method is not called with no-render.

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / BOARD_SIZE
        )  # The size of a single grid square in pixels

        # # First we draw the target
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self._board[i, j] == 1:
                    color = (0, 0, 255)
                elif self._board[i, j] == 2:
                    color = (255, 255, 0)
                elif self._board[i, j] == 3:
                    color = (255, 0, 0)
                elif self._board[i, j] == 4:
                    color = (0, 255, 0)
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

        # Finally, add some gridlines
        for x in range(BOARD_SIZE + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if mode == 'human':
            assert self.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata['render_fps'])
        else:  # rgb_array or single_rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


if __name__ == '__main__':
    env = BlokusEnv(render_mode='human')

    obs = env.reset()
    done = False
    while not done:
        for i in range(4):
            player_mask = np.zeros(4, dtype=np.int8)
            player_mask[i] = 1
            act = env.action_space.sample(
                mask={
                    'piece_set': player_mask,
                    'action_idx': obs['action_masks'][i]
                }
            )
            obs, reward, done, _ = env.step(act)