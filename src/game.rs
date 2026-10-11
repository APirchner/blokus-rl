mod actions;
mod agents;
pub mod bitboard;
pub mod errors;
mod pieces;

use self::actions::{Action, ActionSet};
use self::agents::{ALL_PIECES, Agent, Color};
pub use self::bitboard::BOARD_SIZE;
use self::bitboard::{Bitboard, MAX_IDX, SEPARATING_BIT_MASK};
use self::errors::InvalidAction;
pub use self::pieces::PieceType;
use std::fmt;
use std::fmt::Write as _;
use std::sync::OnceLock;

/// The action set is identical for every game, so it is built once and shared across resets.
static ACTION_SET: OnceLock<ActionSet> = OnceLock::new();

/// The agents in turn order, starting from each agent.
const AGENT_ORDER: [[Color; 4]; 4] = [
    [Color::Blue, Color::Yellow, Color::Red, Color::Green],
    [Color::Yellow, Color::Red, Color::Green, Color::Blue],
    [Color::Red, Color::Green, Color::Blue, Color::Yellow],
    [Color::Green, Color::Blue, Color::Yellow, Color::Red],
];

/// The start cell of the agent `k` seats after another one, in that other agent's frame.
pub const START_CELLS: [Bitboard; 4] = [
    Bitboard::cell(0, 0),
    Bitboard::cell(0, BOARD_SIZE - 1),
    Bitboard::cell(BOARD_SIZE - 1, BOARD_SIZE - 1),
    Bitboard::cell(BOARD_SIZE - 1, 0),
];

/// Bytes per agent written by `Game::to_bytes`: its board, pieces and last piece.
const AGENT_BYTES: usize = 64 + 4 + 1;

/// What the final rewards are based on, see `Game::outcome`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RewardMode {
    /// Each agent's official score.
    Score,
    /// 1/k for each of the k agents tied for the top score, 0 for the others.
    WinShare,
    /// Place points 1, 2/3, 1/3 and 0; tied agents share the average of their places' points.
    Rank,
}

/// What an agent observes, in its frame and with the agents in agent-relative order: itself, the
/// next to move, the one after, and the previous one.
#[derive(PartialEq, Debug)]
pub struct Observation {
    /// The cells each agent occupies.
    pub boards: [Bitboard; 4],
    /// The free cells each agent's next piece can attach to: diagonal but not edge-adjacent to its
    /// pieces, or its start cell before its first piece.
    pub corners: [Bitboard; 4],
    /// The cells each agent can never fill: occupied, or edge-adjacent to its pieces.
    pub blocked: [Bitboard; 4],
    /// The pieces each agent still has, bit `i` for `PieceType::ALL[i]`.
    pub pieces: [u32; 4],
    /// Whether each agent is out of moves.
    pub done: [bool; 4],
    /// The legal actions over the flattened (orientation, row, col) action grid, as bytes of 0/1.
    pub action_mask: Vec<u8>,
}

/// The game state. It holds no heap allocations, so it is cheap to copy.
#[derive(Clone, Copy)]
pub struct Game {
    pub action_set: &'static ActionSet,
    pub agent_selection: usize,
    pub num_agents: usize,
    pub agents: [Agent; 4],
}

impl Game {
    pub fn new() -> Self {
        let action_set = ACTION_SET.get_or_init(|| ActionSet::new(&pieces::generate()));
        let agents = [
            Agent::new(Color::Blue),
            Agent::new(Color::Yellow),
            Agent::new(Color::Red),
            Agent::new(Color::Green),
        ];
        let num_agents: usize = 4;
        let agent_selection: usize = 0;
        Self {
            action_set,
            agent_selection,
            num_agents,
            agents,
        }
    }

    /// Returns what the agent observes, see `Observation`.
    pub fn observe(&self, agent_idx: usize) -> Observation {
        let order = AGENT_ORDER[agent_idx];
        let boards = self.align_boards(agent_idx);
        let occupied = boards[0] | boards[1] | boards[2] | boards[3];
        let blocked = boards.map(|own| own.dilate_ortho() | occupied);
        let corners = std::array::from_fn(|k| corners(boards[k], blocked[k], START_CELLS[k]));
        let pieces = order.map(|c| self.agents[c as usize].pieces);
        let action_mask = self
            .action_set
            .grid_mask(self.fitting_actions(pieces[0], blocked[0], corners[0]));
        Observation {
            boards,
            corners,
            blocked,
            pieces,
            done: order.map(|c| self.agents[c as usize].done),
            action_mask,
        }
    }

    /// The indices of the actions the agent can play.
    pub fn legal_actions(&self, agent_idx: usize) -> impl Iterator<Item = usize> + '_ {
        let (blocked, corners) = self.placement_masks(agent_idx);
        self.fitting_actions(self.agents[agent_idx].pieces, blocked, corners)
    }

    pub fn terminations(&self) -> [bool; 4] {
        [
            self.agents[0].done,
            self.agents[1].done,
            self.agents[2].done,
            self.agents[3].done,
        ]
    }

    /// Each agent's official score as if the game ended now: minus the squares of its remaining
    /// pieces, +15 for placing all pieces, +20 if the monomino was placed last.
    pub fn scores(&self) -> [i16; 4] {
        self.agents.map(|a| {
            let remaining = PieceType::iter()
                .enumerate()
                .filter(|(i, _)| a.pieces & (1 << i) != 0)
                .fold(0u8, |acc, (_, p)| acc + p.size());
            match remaining {
                // +5 bonus if the monomino was placed last
                0 if a.last_piece == Some(PieceType::I1) => 20,
                0 => 15,
                x => -i16::from(x),
            }
        })
    }

    /// The score of every agent that is done, it is final since an agent never regains moves.
    pub fn rewards(&self) -> Vec<Option<i16>> {
        self.agents
            .iter()
            .zip(self.scores())
            .map(|(a, score)| a.done.then_some(score))
            .collect()
    }

    /// The game is over once every agent is out of moves.
    pub fn is_over(&self) -> bool {
        self.agents.iter().all(|a| a.done)
    }

    /// The final reward of every agent in `mode`, None until the game is over.
    pub fn outcome(&self, mode: RewardMode) -> Option<[f32; 4]> {
        self.is_over()
            .then(|| outcome_from_scores(self.scores(), mode))
    }

    /// The number of placements made so far, by all agents.
    pub fn ply(&self) -> u16 {
        self.agents.iter().map(|a| u16::from(a.turn)).sum()
    }

    /// Plays the action at `grid_idx` of the flattened (orientation, row, col) action grid.
    /// Agents that are done pass their turn regardless of the action, e.g. with `None`.
    pub fn step(&mut self, grid_idx: Option<usize>) -> Result<(), InvalidAction> {
        let agent = &self.agents[self.agent_selection];
        if !agent.done {
            let (blocked, corners) = self.placement_masks(self.agent_selection);
            let action_idx = grid_idx
                .and_then(|i| self.action_set.grid_map.get(i).copied().flatten())
                .filter(|i| {
                    let action = &self.action_set[*i];
                    agent.pieces & (1 << action.piece_type.index()) != 0
                        && check_action_valid(action.bitboard, blocked, corners)
                })
                .ok_or_else(|| InvalidAction::new("Action index not available".to_owned()))?;
            let action = self.action_set[action_idx];
            self.execute_action(&action);
            self.update_done();
        }
        // switch control to next agent
        self.agent_selection = (self.agent_selection + 1) % 4;
        Ok(())
    }

    /// The state as bytes: per agent its board, pieces and last piece, then the agent selection.
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 * AGENT_BYTES + 1);
        for a in &self.agents {
            for word in [a.board.0, a.board.1, a.board.2, a.board.3] {
                bytes.extend(word.to_le_bytes());
            }
            bytes.extend(a.pieces.to_le_bytes());
            // piece types are numbered from 1, so 0 stands for no piece
            bytes.push(a.last_piece.map_or(0, |p| p as u8));
        }
        bytes.push(self.agent_selection as u8);
        bytes
    }

    /// The state written by `to_bytes`, None if `bytes` is not one. It does not check that the
    /// position can be reached in a game.
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let (agent_selection, agents) = bytes.split_last()?;
        if agents.len() != 4 * AGENT_BYTES || *agent_selection >= 4 {
            return None;
        }
        let mut game = Self::new();
        game.agent_selection = usize::from(*agent_selection);
        for (a, chunk) in game
            .agents
            .iter_mut()
            .zip(agents.as_chunks::<AGENT_BYTES>().0)
        {
            let word =
                |i: usize| u128::from_le_bytes(chunk[16 * i..16 * (i + 1)].try_into().unwrap());
            a.board = Bitboard(word(0), word(1), word(2), word(3));
            a.pieces = u32::from_le_bytes(chunk[64..68].try_into().unwrap());
            a.last_piece = match chunk[68] {
                0 => None,
                x => Some(PieceType::iter().find(|p| *p as u8 == x)?),
            };
            if a.board & !SEPARATING_BIT_MASK != Bitboard::default() || a.pieces & !ALL_PIECES != 0
            {
                return None;
            }
            a.turn = (ALL_PIECES & !a.pieces).count_ones() as u8;
        }
        game.update_done();
        Some(game)
    }

    fn align_boards(&self, agent_idx: usize) -> [Bitboard; 4] {
        let order = AGENT_ORDER[agent_idx];
        [
            self.agents[order[0] as usize].board,
            self.agents[order[1] as usize].board.rotate_clock(),
            self.agents[order[2] as usize]
                .board
                .rotate_anticlock()
                .rotate_anticlock(),
            self.agents[order[3] as usize].board.rotate_anticlock(),
        ]
    }

    /// The cells the agent can not fill and the corners its next piece must cover, in its frame.
    fn placement_masks(&self, agent_idx: usize) -> (Bitboard, Bitboard) {
        let boards = self.align_boards(agent_idx);
        let blocked = boards[0].dilate_ortho() | boards[1] | boards[2] | boards[3];
        (blocked, corners(boards[0], blocked, START_CELLS[0]))
    }

    /// The placements of the `pieces` that avoid the `blocked` cells and cover one of the `corners`.
    fn fitting_actions(
        &self,
        pieces: u32,
        blocked: Bitboard,
        corners: Bitboard,
    ) -> impl Iterator<Item = usize> + '_ {
        PieceType::iter()
            .enumerate()
            .filter(move |(i, _)| pieces & (1 << i) != 0)
            .flat_map(|(_, p)| self.action_set.piece_map[&p].clone())
            .filter(move |a| check_action_valid(self.action_set[*a].bitboard, blocked, corners))
    }

    fn execute_action(&mut self, action: &Action) {
        let agent = &mut self.agents[self.agent_selection];
        // execute action, the agent's own board is already in its frame
        agent.board |= action.bitboard;
        agent.turn += 1;
        agent.pieces &= !(1 << action.piece_type.index());
        agent.last_piece = Some(action.piece_type);
    }

    /// Flags the agents that are out of moves. An agent never regains moves, so only the agents
    /// that still had moves need a check.
    fn update_done(&mut self) {
        for i in 0..4 {
            if !self.agents[i].done {
                let done = self.legal_actions(i).next().is_none();
                self.agents[i].done = done;
            }
        }
    }

    pub fn render(&self) {
        println!("{self}");
    }
}

fn cell_occupier(boards: &[Bitboard; 4], idx: usize) -> Option<Color> {
    if boards[Color::Blue as usize].bit_lookup(idx) {
        Some(Color::Blue)
    } else if boards[Color::Yellow as usize].bit_lookup(idx) {
        Some(Color::Yellow)
    } else if boards[Color::Red as usize].bit_lookup(idx) {
        Some(Color::Red)
    } else if boards[Color::Green as usize].bit_lookup(idx) {
        Some(Color::Green)
    } else {
        None
    }
}

/// An action is valid if it avoids the `blocked` cells (own pieces and their edges, other pieces)
/// and covers one of the `corners`.
fn check_action_valid(action_board: Bitboard, blocked: Bitboard, corners: Bitboard) -> bool {
    (action_board & blocked).is_empty() && !(action_board & corners).is_empty()
}

/// The free cells an agent's next piece can attach to, given its `own` cells, the cells it can not
/// fill and its `start` cell.
fn corners(own: Bitboard, blocked: Bitboard, start: Bitboard) -> Bitboard {
    let attach = if own.is_empty() {
        start
    } else {
        own.dilate_diag()
    };
    attach & !blocked
}

/// The final rewards for the final `scores`, see `RewardMode`.
#[allow(clippy::cast_precision_loss)]
fn outcome_from_scores(scores: [i16; 4], mode: RewardMode) -> [f32; 4] {
    scores.map(|score| {
        let above = scores.iter().filter(|s| **s > score).count();
        let tied = scores.iter().filter(|s| **s == score).count();
        match mode {
            RewardMode::Score => f32::from(score),
            RewardMode::WinShare if above == 0 => 1.0 / tied as f32,
            RewardMode::WinShare => 0.0,
            // the average points of the shared places, place p (from 0) gets (3 - p) / 3
            RewardMode::Rank => {
                (above..above + tied)
                    .map(|p| (3 - p) as f32 / 3.0)
                    .sum::<f32>()
                    / tied as f32
            }
        }
    })
}

impl fmt::Display for Game {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let boards = self.align_boards(0);

        let horizontal = "-".repeat(3 * BOARD_SIZE + 43);
        let mut board = String::from("  | ");
        let header = (0..BOARD_SIZE)
            .map(|x| format!("{:>2}", x.to_string()))
            .collect::<Vec<String>>()
            .join(" | ");
        let _ = write!(board, "{header}");
        let _ = write!(board, " |\n{}\n{:>2}", horizontal, 0);

        let mut line_break: usize = BOARD_SIZE;
        for i in 0..MAX_IDX {
            // the separating bit ends the row instead of being drawn as a cell
            if i == line_break {
                let _ = write!(board, "|\n{}\n{:>2}", horizontal, i / BOARD_SIZE);
                line_break += BOARD_SIZE + 1;
                continue;
            }
            match cell_occupier(&boards, i) {
                Some(Color::Blue) => board.push_str("|  \x1b[44mx\x1b[0m "),
                Some(Color::Yellow) => board.push_str("|  \x1b[43mx\x1b[0m "),
                Some(Color::Red) => board.push_str("|  \x1b[41mx\x1b[0m "),
                Some(Color::Green) => board.push_str("|  \x1b[42mx\x1b[0m "),
                None => board.push_str("|  . "),
            }
        }
        write!(f, "{board}|")
    }
}

impl fmt::Debug for Game {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Game")
            .field("agents", &self.agents)
            .field("num_agents", &self.num_agents)
            .field("agent_selection", &self.agent_selection)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: [f32; 4], expected: [f32; 4]) {
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn test_win_share_splits_ties() {
        let third = 1.0 / 3.0;
        for (scores, expected) in [
            ([-10, 15, -3, -40], [0.0, 1.0, 0.0, 0.0]),
            ([-10, -3, -3, -40], [0.0, 0.5, 0.5, 0.0]),
            ([-3, -3, -40, -3], [third, third, 0.0, third]),
            ([-3, -3, -3, -3], [0.25; 4]),
        ] {
            let rewards = outcome_from_scores(scores, RewardMode::WinShare);
            assert_close(rewards, expected);
            assert!((rewards.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rank_averages_tied_places() {
        let third = 1.0 / 3.0;
        for (scores, expected) in [
            ([-10, 15, -3, -40], [third, 1.0, 2.0 * third, 0.0]),
            ([-10, -3, -3, -40], [third, 5.0 / 6.0, 5.0 / 6.0, 0.0]),
            ([-10, -3, -10, -40], [0.5, 1.0, 0.5, 0.0]),
            (
                [-3, -3, -40, -3],
                [2.0 * third, 2.0 * third, 0.0, 2.0 * third],
            ),
            ([-3, -3, -3, -3], [0.5; 4]),
        ] {
            assert_close(outcome_from_scores(scores, RewardMode::Rank), expected);
        }
    }

    #[test]
    fn test_outcome_only_at_game_end() {
        let mut game = Game::new();
        assert_eq!(game.outcome(RewardMode::Score), None);
        for a in &mut game.agents {
            a.done = true;
        }
        assert_eq!(game.outcome(RewardMode::Score), Some([-89.0; 4]));
        assert_eq!(game.outcome(RewardMode::WinShare), Some([0.25; 4]));
    }
}
