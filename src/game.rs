mod actions;
mod agents;
pub(crate) mod bitboard;
pub mod errors;
pub mod pentobi;
mod pieces;

use self::actions::Action;
pub use self::actions::ActionSet;
use self::agents::{ALL_PIECES, Agent, Color};
pub use self::bitboard::BOARD_SIZE;
use self::bitboard::{Bitboard, MAX_IDX, SEPARATING_BIT_MASK};
use self::errors::InvalidAction;
pub use self::pieces::PieceType;
use std::fmt;
use std::fmt::Write as _;
use std::ops::ControlFlow;

/// The agents in turn order, starting from each agent.
const AGENT_ORDER: [[Color; 4]; 4] = [
    [Color::Blue, Color::Yellow, Color::Red, Color::Green],
    [Color::Yellow, Color::Red, Color::Green, Color::Blue],
    [Color::Red, Color::Green, Color::Blue, Color::Yellow],
    [Color::Green, Color::Blue, Color::Yellow, Color::Red],
];

/// The start cell of the agent `k` seats after another one, in that other agent's frame. Boards
/// are stored in the frame of agent 0, so these are the agents' start cells there.
pub(crate) const START_CELLS: [Bitboard; 4] = [
    Bitboard::cell(0, 0),
    Bitboard::cell(0, BOARD_SIZE - 1),
    Bitboard::cell(BOARD_SIZE - 1, BOARD_SIZE - 1),
    Bitboard::cell(BOARD_SIZE - 1, 0),
];

/// Bytes per agent written by `Game::to_bytes`: its board, pieces and last piece.
const AGENT_BYTES: usize = 64 + 4 + 1;

/// Version of the format written by `Game::to_bytes`, its first byte. It changes whenever the
/// meaning of the bytes does, e.g. the order of `PieceType::ALL` or the frame of the boards, so
/// that `Game::from_bytes` rejects states of another version.
const BYTES_VERSION: u8 = 1;

/// Length of the `planes` written by `Game::encode`: the occupied cells, corners and blocked
/// cells of each agent, see `Observation`, as (plane, row, col) bytes of 0/1.
pub const PLANES_LEN: usize = 3 * 4 * BOARD_SIZE * BOARD_SIZE;

/// Length of the `globals` written by `Game::encode`: the pieces of each agent as 0/1 in
/// (agent, piece) order, the done flag of each agent as 0/1, the seat and the ply.
pub const GLOBALS_LEN: usize = 4 * PieceType::ALL.len() + 4 + 2;

/// Zobrist keys for every agent and bit of the board, see `Game::hash`.
static CELL_KEYS: [[u64; MAX_IDX + 1]; 4] = cell_keys();

/// Zobrist keys for the agent to move.
const TURN_KEYS: [u64; 4] = agent_keys(4 * (MAX_IDX as u64 + 1));

/// Zobrist keys for the agents that placed all pieces with the monomino last, which scores +5.
const BONUS_KEYS: [u64; 4] = agent_keys(4 * (MAX_IDX as u64 + 1) + 4);

/// Shape (orientation, row, col) of the action grid, see `ActionSet::grid_idx`.
pub const ACTION_GRID_SHAPE: (usize, usize, usize) = (91, BOARD_SIZE, BOARD_SIZE);

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
pub(crate) struct Observation {
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
}

/// The game state. It holds no heap allocations, so it is cheap to copy.
///
/// Agents are numbered in turn order, 0 moves first. Actions are indices into the action set,
/// each in the frame of the agent that plays it; see `ActionSet::grid_idx` for the policy grid.
#[derive(Clone, Copy)]
pub struct Game {
    pub(crate) action_set: &'static ActionSet,
    pub(crate) agent_selection: usize,
    pub(crate) num_agents: usize,
    /// The agents, with their boards in the frame of agent 0.
    pub(crate) agents: [Agent; 4],
    /// The Zobrist keys of the agents' cells, see `hash`.
    zobrist: u64,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    #[must_use]
    pub fn new() -> Self {
        let agents = [
            Agent::new(Color::Blue),
            Agent::new(Color::Yellow),
            Agent::new(Color::Red),
            Agent::new(Color::Green),
        ];
        let num_agents: usize = 4;
        let agent_selection: usize = 0;
        Self {
            action_set: ActionSet::get(),
            agent_selection,
            num_agents,
            agents,
            zobrist: 0,
        }
    }

    /// The agent to move. Agents that are out of moves are skipped; once the game is over, it is
    /// the agent that would move next.
    #[must_use]
    pub fn current_player(&self) -> usize {
        (0..4)
            .map(|k| (self.agent_selection + k) % 4)
            .find(|a| !self.agents[*a].done)
            .unwrap_or(self.agent_selection)
    }

    /// Writes the legal actions of the agent to move into `out`, which is cleared first.
    pub fn legal_moves(&self, out: &mut Vec<u16>) {
        self.legal_moves_for(self.current_player(), out);
    }

    /// Plays the action of the agent to move. The action must be legal, which is only checked in
    /// debug builds.
    ///
    /// # Panics
    ///
    /// If the game is over, or `action` is not below 30,433, the number of actions.
    pub fn apply(&mut self, action: u16) {
        let agent = self.current_player();
        assert!(!self.agents[agent].done, "The game is over");
        debug_assert!(
            self.is_legal(agent, usize::from(action)),
            "Illegal action {action} for agent {agent}"
        );
        self.place(agent, usize::from(action));
        self.agent_selection = (agent + 1) % 4;
    }

    /// The Zobrist hash of the position: the cells of every agent, the agent to move until the
    /// game is over, and the agents that earned the monomino bonus.
    #[must_use]
    pub fn hash(&self) -> u64 {
        let bonus = (0..4)
            .filter(|a| {
                self.agents[*a].pieces == 0 && self.agents[*a].last_piece == Some(PieceType::I1)
            })
            .fold(0, |h, a| h ^ BONUS_KEYS[a]);
        // a finished game has no agent to move, however it was reached
        let turn = if self.is_over() {
            0
        } else {
            TURN_KEYS[self.current_player()]
        };
        self.zobrist ^ turn ^ bonus
    }

    /// Writes the observation of the agent to move into `planes` and `globals`, see `PLANES_LEN`
    /// and `GLOBALS_LEN` for the layout. It is the observation of the Python environment.
    ///
    /// # Panics
    ///
    /// If `planes` is shorter than `PLANES_LEN` or `globals` is shorter than `GLOBALS_LEN`.
    pub fn encode(&self, planes: &mut [u8], globals: &mut [f32]) {
        self.encode_for(self.current_player(), planes, globals);
    }

    /// Returns what the agent observes, see `Observation`.
    pub(crate) fn observe(&self, agent_idx: usize) -> Observation {
        let order = AGENT_ORDER[agent_idx];
        let boards = order.map(|c| into_frame(self.agents[c as usize].board, agent_idx));
        let occupied = boards[0] | boards[1] | boards[2] | boards[3];
        let masks: [_; 4] =
            std::array::from_fn(|k| blocked_and_corners(boards[k], occupied, START_CELLS[k]));
        Observation {
            boards,
            corners: masks.map(|(_, corners)| corners),
            blocked: masks.map(|(blocked, _)| blocked),
            pieces: order.map(|c| self.agents[c as usize].pieces),
            done: order.map(|c| self.agents[c as usize].done),
        }
    }

    #[must_use]
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
    #[must_use]
    pub fn scores(&self) -> [i16; 4] {
        self.agents.map(|a| {
            let remaining = PieceType::iter()
                .filter(|p| a.pieces & p.bit() != 0)
                .fold(0u8, |acc, p| acc + p.size());
            match remaining {
                // +5 bonus if the monomino was placed last
                0 if a.last_piece == Some(PieceType::I1) => 20,
                0 => 15,
                x => -i16::from(x),
            }
        })
    }

    /// The score of every agent that is done, it is final since an agent never regains moves.
    #[must_use]
    pub fn rewards(&self) -> Vec<Option<i16>> {
        self.agents
            .iter()
            .zip(self.scores())
            .map(|(a, score)| a.done.then_some(score))
            .collect()
    }

    /// The game is over once every agent is out of moves.
    #[must_use]
    pub fn is_over(&self) -> bool {
        self.agents.iter().all(|a| a.done)
    }

    /// The final reward of every agent in `mode`, None until the game is over.
    #[must_use]
    pub fn outcome(&self, mode: RewardMode) -> Option<[f32; 4]> {
        self.is_over()
            .then(|| outcome_from_scores(self.scores(), mode))
    }

    /// The number of placements made so far, by all agents.
    #[must_use]
    pub fn ply(&self) -> u16 {
        self.agents.iter().map(|a| u16::from(a.turn)).sum()
    }

    /// Plays the action at `grid_idx` of the flattened (orientation, row, col) action grid for the
    /// selected agent, then selects the next agent. Agents that are done pass their turn
    /// regardless of the action, e.g. with `None`.
    ///
    /// # Errors
    ///
    /// If the selected agent has moves and `grid_idx` is not one of them.
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn step(&mut self, grid_idx: Option<usize>) -> Result<(), InvalidAction> {
        let agent = self.agent_selection;
        if !self.agents[agent].done {
            let action = grid_idx
                .and_then(|i| self.action_set.grid_map.get(i).copied().flatten())
                .filter(|a| self.is_legal(agent, *a))
                .ok_or_else(|| InvalidAction::new("Action index not available".to_owned()))?;
            self.place(agent, action);
        }
        // switch control to next agent
        self.agent_selection = (agent + 1) % 4;
        Ok(())
    }

    /// The state as bytes: the format version, per agent its board in the frame of agent 0, its
    /// pieces and its last piece, then the selected agent.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn to_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 * AGENT_BYTES + 2);
        bytes.push(BYTES_VERSION);
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

    /// The state written by `to_bytes`, None if `bytes` is not one or of another format version.
    /// It does not check that the position can be reached in a game.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let [BYTES_VERSION, agents @ .., agent_selection] = bytes else {
            return None;
        };
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
            let words = chunk[..64].as_chunks::<16>().0;
            let word = |i: usize| u128::from_le_bytes(words[i]);
            a.board = Bitboard(word(0), word(1), word(2), word(3));
            a.pieces = u32::from_le_bytes([chunk[64], chunk[65], chunk[66], chunk[67]]);
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
        game.zobrist = game.full_zobrist();
        game.update_done();
        Some(game)
    }

    /// Writes the legal actions of the agent into `out`, in its frame.
    pub(crate) fn legal_moves_for(&self, agent: usize, out: &mut Vec<u16>) {
        out.clear();
        let (blocked, corners) = self.placement_masks(agent);
        let into_agent_frame = &self.action_set.rotated[(4 - agent) % 4];
        let pieces = self.agents[agent].pieces;
        let _ = self.for_each_placement(pieces, blocked, corners, |global| {
            out.push(into_agent_frame[usize::from(global)]);
            ControlFlow::Continue(())
        });
    }

    /// Writes the legal actions of the agent that observed `obs` into `out`, in its frame, as
    /// `legal_moves_for` but from the masks of the observation.
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn observed_moves(&self, obs: &Observation, out: &mut Vec<u16>) {
        out.clear();
        // the masks are in the agent's frame, and so are the placements found from them
        let _ = self.for_each_placement(obs.pieces[0], obs.blocked[0], obs.corners[0], |action| {
            out.push(action);
            ControlFlow::Continue(())
        });
    }

    /// See `encode`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub(crate) fn encode_for(&self, agent: usize, planes: &mut [u8], globals: &mut [f32]) {
        let obs = self.observe(agent);
        let boards = [obs.boards, obs.corners, obs.blocked];
        write_planes(boards.as_flattened(), &mut planes[..PLANES_LEN]);
        let pieces = obs
            .pieces
            .iter()
            .flat_map(|p| (0..PieceType::ALL.len()).map(move |i| p >> i & 1));
        let done = obs.done.map(u32::from);
        let values = pieces
            .chain(done)
            .chain([agent as u32, u32::from(self.ply())]);
        for (g, v) in globals[..GLOBALS_LEN].iter_mut().zip(values) {
            *g = v as f32;
        }
    }

    /// The Zobrist keys of the agents' cells, computed from scratch.
    pub(crate) fn full_zobrist(&self) -> u64 {
        (0..4).fold(0, |h, a| h ^ cells_key(a, self.agents[a].board))
    }

    /// Whether the agent can play the action, in its frame.
    pub(crate) fn is_legal(&self, agent: usize, action: usize) -> bool {
        let action = self.action_set[usize::from(self.action_set.rotated[agent][action])];
        let (blocked, corners) = self.placement_masks(agent);
        self.agents[agent].pieces & action.piece_type.bit() != 0
            && check_action_valid(action.bitboard, blocked, corners)
    }

    /// The cells the agent can not fill and the corners its next piece must cover, in the frame of
    /// agent 0.
    fn placement_masks(&self, agent: usize) -> (Bitboard, Bitboard) {
        let occupied = self
            .agents
            .iter()
            .fold(Bitboard::default(), |acc, a| acc | a.board);
        blocked_and_corners(self.agents[agent].board, occupied, START_CELLS[agent])
    }

    /// Calls `visit` with every placement of the `pieces` that avoids the `blocked` cells and covers
    /// one of the `corners`, in the frame of the masks, until it breaks. Only the placements that
    /// cover a corner are tested.
    fn for_each_placement(
        &self,
        pieces: u32,
        blocked: Bitboard,
        corners: Bitboard,
        mut visit: impl FnMut(u16) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        let mut skip = blocked;
        for corner in corners.ones() {
            for piece in (0..PieceType::ALL.len()).filter(|i| pieces & (1 << i) != 0) {
                for &action in self.action_set.covering(corner, piece) {
                    if (self.action_set[usize::from(action)].bitboard & skip).is_empty() {
                        visit(action)?;
                    }
                }
            }
            // a placement that covers this corner was visited from it
            skip |= Bitboard::bit(corner);
        }
        ControlFlow::Continue(())
    }

    /// Places the action of the agent, in its frame.
    fn place(&mut self, agent: usize, action: usize) {
        let action: Action = self.action_set[usize::from(self.action_set.rotated[agent][action])];
        let a = &mut self.agents[agent];
        a.board |= action.bitboard;
        a.turn += 1;
        a.pieces &= !action.piece_type.bit();
        a.last_piece = Some(action.piece_type);
        self.zobrist ^= cells_key(agent, action.bitboard);
        self.update_done();
    }

    /// Flags the agents that are out of moves. An agent never regains moves, so only the agents
    /// that still had moves need a check.
    fn update_done(&mut self) {
        for i in 0..4 {
            if !self.agents[i].done {
                let (blocked, corners) = self.placement_masks(i);
                let pieces = self.agents[i].pieces;
                let done =
                    self.for_each_placement(pieces, blocked, corners, |_| ControlFlow::Break(()));
                self.agents[i].done = done.is_continue();
            }
        }
    }

    pub fn render(&self) {
        println!("{self}");
    }
}

/// Rotates a board from the frame of agent 0 into the frame of the agent.
fn into_frame(board: Bitboard, agent: usize) -> Bitboard {
    match agent {
        0 => board,
        1 => board.rotate_anticlock(),
        2 => board.rotate_clock().rotate_clock(),
        _ => board.rotate_clock(),
    }
}

/// Writes the `boards` into `planes` as (board, row, col) bytes of 0/1.
pub(crate) fn write_planes(boards: &[Bitboard], planes: &mut [u8]) {
    planes.fill(0);
    for (plane, board) in boards.iter().enumerate() {
        for idx in board.ones() {
            // the separating bit ends every row of the board
            planes[plane * BOARD_SIZE * BOARD_SIZE + idx - idx / (BOARD_SIZE + 1)] = 1;
        }
    }
}

/// The XOR of the agent's Zobrist keys for the cells of `board`.
fn cells_key(agent: usize, board: Bitboard) -> u64 {
    board.ones().fold(0, |h, idx| h ^ CELL_KEYS[agent][idx])
}

/// The `i`-th output of the `SplitMix64` generator.
const fn zobrist_key(i: u64) -> u64 {
    let mut z = (i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const fn agent_keys(first: u64) -> [u64; 4] {
    [
        zobrist_key(first),
        zobrist_key(first + 1),
        zobrist_key(first + 2),
        zobrist_key(first + 3),
    ]
}

const fn cell_keys() -> [[u64; MAX_IDX + 1]; 4] {
    let mut keys = [[0; MAX_IDX + 1]; 4];
    let mut i = 0;
    while i < 4 * (MAX_IDX + 1) {
        keys[i / (MAX_IDX + 1)][i % (MAX_IDX + 1)] = zobrist_key(i as u64);
        i += 1;
    }
    keys
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

/// The cells an agent can not fill and the free cells its next piece can attach to, see
/// `Observation`, given its `own` cells, all `occupied` cells and its `start` cell. It is the
/// placement rule of both the observation and the legal moves.
fn blocked_and_corners(own: Bitboard, occupied: Bitboard, start: Bitboard) -> (Bitboard, Bitboard) {
    let blocked = own.dilate_ortho() | occupied;
    let attach = if own.is_empty() {
        start
    } else {
        own.dilate_diag()
    };
    (blocked, attach & !blocked)
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
        let boards = self.agents.map(|a| a.board);

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
impl Game {
    /// The agent's legal actions in its frame, found by testing every placement of its pieces as
    /// before the corner-anchored generation. It is the oracle for `legal_moves`.
    pub(crate) fn oracle_moves(&self, agent: usize) -> impl Iterator<Item = u16> + '_ {
        let obs = self.observe(agent);
        (0..self.action_set.actions.len())
            .filter(move |a| {
                let action = self.action_set[*a];
                obs.pieces[0] & action.piece_type.bit() != 0
                    && check_action_valid(action.bitboard, obs.blocked[0], obs.corners[0])
            })
            .map(|a| u16::try_from(a).unwrap())
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
