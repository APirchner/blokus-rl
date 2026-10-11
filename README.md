# blokus-rl
A [PettingZoo-API-compatible](https://pettingzoo.farama.org) reinforcement learning (RL) environment for the strategy board game [Blokus](https://en.wikipedia.org/wiki/Blokus).

![blokus-rl](blokus_episode.gif)

## Installation

### pip
`pip install -U blokus-rl`
### Poetry
`poetry add blokus-rl`
### Source
```
git clone https://github.com/APirchner/blokus-rl.git
cd blokus-rl
pip install -U maturin
maturin build [--release]
pip install {path/to/wheel}.whl
```

## Example usage

```python
from blokus_rl import BlokusEnv

env = BlokusEnv(render_mode="human")
env.reset()
scores = {}
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        # the agent is out of moves, its score is final and it leaves the game
        scores[agent] = reward
        action = None
    else:
        action = env.action_space(agent).sample(mask=observation["action_mask"])
    env.step(action)
print(scores)
```

## Rewards
All rewards build on the official Blokus score: minus the squares of the remaining pieces, +15 for placing all pieces, +20 if the monomino was placed last.
`BlokusEnv(reward_mode=...)` selects the reward:

| `reward_mode` | Reward | When |
|---|---|---|
| `"score"` (default) | The agent's own score | As soon as the agent is out of moves, since it can never get moves back. It then leaves the game. |
| `"win_share"` | 1/k if among the k agents tied for the top score, else 0; rewards sum to 1 | At game end, to all agents at once |
| `"rank"` | Place points 1, 2/3, 1/3, 0; tied agents share the average of their places' points | At game end, to all agents at once |

In `win_share` and `rank` modes, an agent without moves is not terminated before the game is over; its turns are skipped.
At game end, `infos[agent]["scores"]` holds the final scores of all agents, for every agent still in the game. In `score` mode, agents that ran out of moves earlier have already left.

## Observation
The observation is a dict, in the frame of the observing agent.
The agent axis is agent-relative: the agent itself, the next to move, the one after, and the previous one.

| Key | Shape, dtype | Content |
|---|---|---|
| `observation` | (4, 20, 20) bool | Occupied cells per agent |
| `action_mask` | (36400,) int8 | Legal actions, see [Action space](#action-space) |
| `pieces` | (4, 21) bool | Pieces each agent still has, in the order of `env.piece_names` |
| `corners` | (4, 20, 20) bool | Free cells diagonal but not edge-adjacent to the agent's pieces, or its start cell before its first piece |
| `blocked` | (4, 20, 20) bool | Cells the agent can never fill: occupied by anyone, or edge-adjacent to its own pieces |
| `done` | (4,) bool | Agent is out of moves |
| `seat` | Discrete(4) | The observing agent's place in turn order; 0 moves first |
| `ply` | () int16 | Placements made so far, by all agents |

## Action space
Every action places one of the 91 piece orientations (21 pieces in all their distinct rotations and flips) on the 20x20 board.
Actions are laid out as a grid of shape `(orientation, row, col)` = `env.action_grid_shape` = `(91, 20, 20)` and flattened in C order into `Discrete(36400)`:
`(row, col)` is the top-left corner of the piece's bounding box, in the frame of the acting agent.
Positions where the piece would not fit on the board are always masked.

```python
action_mask = observation["action_mask"].reshape(env.action_grid_shape)
orientation, row, col = np.unravel_index(action, env.action_grid_shape)
```

## Copying and saving states
`copy.deepcopy(env)` copies an environment, e.g. to branch off a tree search; the copy plays on independently and does not render.
`env._env.copy()` copies only the game, which is much cheaper.
The game state can also be saved as bytes and restored:

```python
from blokus_rl._blokus import PyBlokus

state = env._env.to_bytes()
game = PyBlokus.from_bytes(state)
```

## Rust library
Search code, such as a tree search for self-play, can use the game directly from Rust without Python:

```toml
[dependencies]
blokus-rl = { git = "https://github.com/APirchner/blokus-rl", default-features = false }
```

```rust
use blokus_rl::game::{ACTION_GRID_SHAPE, ActionSet, GLOBALS_LEN, Game, PLANES_LEN, RewardMode};

let mut game = Game::new();
let mut moves = Vec::new();
let (mut planes, mut globals) = (vec![0; PLANES_LEN], vec![0.0; GLOBALS_LEN]);
while !game.is_over() {
    // the observation of the agent to move, as in Python
    game.encode(&mut planes, &mut globals);
    game.legal_moves(&mut moves);
    // an action's index in the policy grid of shape ACTION_GRID_SHAPE = (91, 20, 20)
    let policy_idx = ActionSet::grid_idx(moves[0]);
    let mut branch = game; // a copy is a few hundred bytes
    branch.apply(moves[0]);
    game.apply(moves[0]);
    assert_eq!(branch.hash(), game.hash());
}
println!("{:?}", game.outcome(RewardMode::WinShare));
```

Actions are indices into the action set, in the frame of the agent that plays them.
`Game::hash` is a Zobrist hash of the position, e.g. to cache network evaluations.
`blokus_rl::game::pentobi` converts actions to and from the moves of the [Pentobi](https://pentobi.sourceforge.io/) engine, and `Game::from_moves` replays Pentobi games.

Benchmarks over positions from random games run with `cargo bench`.

## Motivation
I'm not a passionate gamer and get frustrated quite easily when a board game does not go my way. For some reason, I always enjoy playing Blokus - wheter I'm winning or not. And being a computer scientist by training, I was always wondering what weird/effective strategies a powerful RL agent would uncover.

While there are a few comprehensive RL libraries such as [RLlib](https://docs.ray.io/en/latest/rllib/index.html) and environments for common classical boardgames and Atari games, I was not able to find an environment for Blokus that implements with the typical APIs. This repo should fill this gap.

Working mostly in Python, I started out with a inefficient implementation and soon found that masking invalid moves did not scale well.
It never hurts to have some lower-level language in your toolbox, so I started from scratch in Rust - both as a Rust-learning experience and a way to speed up Blokus episodes.

## Implementation details
Internally, the game is implemented with [bitboards](https://www.chessprogramming.org/Bitboards). A set of 4 u128 integers represents the board of 20x20 tiles and an additional column of separating bits. All game logic is built on bit operations - from generating all possible actions to finding the valid subsets of actions for each player at each turn of the game.
While doing research on how to speed up the masking of invalid actions, I found the smart Blokus [implementation](https://github.com/nikohass/rust-socha2021) by [nicohass](https://github.com/nikohass). From this repo I picked up the idea of using bitboards. The basics on how to rotate boards come from the [Chess Programming Wiki](https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating).

The internal logic of the game is implemented in Rust, while the API is exposed through a Python class. The Python bindings for the Rust code are built with [PyO3](https://pyo3.rs/) and [maturin](https://www.maturin.rs).
