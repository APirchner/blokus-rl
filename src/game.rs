mod actions;
mod agents;
pub mod bitboard;
pub mod errors;
mod pieces;

use self::actions::{Action, ActionSet};
use self::agents::{Agent, Color};
pub use self::bitboard::BOARD_SIZE;
use self::bitboard::{Bitboard, MAX_IDX};
use self::errors::InvalidAction;
use std::fmt;
use std::fmt::Write as _;
use std::sync::OnceLock;

/// The action set is identical for every game, so it is built once and shared across resets.
static ACTION_SET: OnceLock<ActionSet> = OnceLock::new();

pub struct Game {
    pub action_set: &'static ActionSet,
    pub agent_selection: usize,
    pub num_agents: usize,
    pub agents: [Agent; 4],
    pub agent_order: [[Color; 4]; 4],
}

impl Game {
    pub fn new() -> Self {
        let action_set = ACTION_SET.get_or_init(|| ActionSet::new(&pieces::generate()));
        let agents = [
            Agent::new(Color::Blue, action_set.initial_actions()),
            Agent::new(Color::Yellow, action_set.initial_actions()),
            Agent::new(Color::Red, action_set.initial_actions()),
            Agent::new(Color::Green, action_set.initial_actions()),
        ];
        let num_agents: usize = 4;
        let agent_selection: usize = 0;
        let agent_order = [
            [Color::Blue, Color::Yellow, Color::Red, Color::Green],
            [Color::Yellow, Color::Red, Color::Green, Color::Blue],
            [Color::Red, Color::Green, Color::Blue, Color::Yellow],
            [Color::Green, Color::Blue, Color::Yellow, Color::Red],
        ];
        Self {
            action_set,
            agent_selection,
            num_agents,
            agents,
            agent_order,
        }
    }

    /// Returns the boards aligned to the agent's frame and its action mask over the
    /// flattened (orientation, row, col) action grid.
    pub fn observe(
        &mut self,
        agent_idx: usize,
    ) -> ([[[bool; BOARD_SIZE]; BOARD_SIZE]; 4], Vec<bool>) {
        let boards = self.align_boards(agent_idx);
        if self.agents[agent_idx].turn > 0 && !self.agents[agent_idx].done {
            let blocked = boards[0].dilate_ortho() | boards[1] | boards[2] | boards[3];
            let diagonals = boards[0].dilate_diag();
            for (p, _) in self.agents[agent_idx].pieces.iter().filter(|p| *p.1) {
                let piece_range = &self.action_set.piece_map[p];
                for r in piece_range.clone() {
                    self.agents[agent_idx].action_mask[r] =
                        check_action_valid(self.action_set.actions[r].bitboard, blocked, diagonals);
                }
            }
            self.agents[agent_idx].done = !self.agents[agent_idx].action_mask.iter().any(|x| *x);
        }

        self.agents[agent_idx].action_mask_stale = false;
        (
            [
                boards[0].into_vecs(),
                boards[1].into_vecs(),
                boards[2].into_vecs(),
                boards[3].into_vecs(),
            ],
            self.action_set
                .grid_mask(&self.agents[agent_idx].action_mask),
        )
    }

    pub fn terminations(&self) -> [bool; 4] {
        [
            self.agents[0].done,
            self.agents[1].done,
            self.agents[2].done,
            self.agents[3].done,
        ]
    }

    pub fn rewards(&self) -> Option<Vec<i16>> {
        let game_done = self.terminations().iter().all(|x| *x);
        if !game_done {
            return None;
        }
        let scores_raw: Vec<u8> = self
            .agents
            .iter()
            .map(|x| {
                x.pieces
                    .iter()
                    .filter(|y| *y.1)
                    .fold(0u8, |acc, p| acc + p.0.size())
            })
            .collect();
        let scores = scores_raw
            .iter()
            .map(|x| match x {
                0 => 15,
                x => -i16::from(*x),
            })
            .collect();
        Some(scores)
    }

    /// Plays the action at `grid_idx` of the flattened (orientation, row, col) action grid.
    /// Agents that are done pass their turn regardless of the action, e.g. with `None`.
    pub fn step(&mut self, grid_idx: Option<usize>) -> Result<(), InvalidAction> {
        let agent = &self.agents[self.agent_selection];
        if !agent.done {
            if agent.action_mask_stale && agent.turn > 0 {
                return Err(InvalidAction::new(
                    "Action mask is stale, call observe() first!".to_owned(),
                ));
            }
            let action_idx = grid_idx
                .and_then(|i| self.action_set.grid_map.get(i).copied().flatten())
                .filter(|i| agent.action_mask[*i])
                .ok_or_else(|| InvalidAction::new("Action index not available".to_owned()))?;
            let action = self.action_set[action_idx];
            self.execute_action(&action);

            for a in &mut self.agents {
                a.action_mask_stale = true;
            }
        }
        // switch control to next agent
        self.agent_selection = (self.agent_selection + 1) % 4;
        Ok(())
    }

    fn align_boards(&self, agent_idx: usize) -> [Bitboard; 4] {
        let order = self.agent_order[agent_idx];
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

    fn execute_action(&mut self, action: &Action) {
        let agent = &mut self.agents[self.agent_selection];
        // execute action, the agent's own board is already in its frame
        agent.board |= action.bitboard;
        agent.turn += 1;
        agent.pieces.insert(action.piece_type, false);
        // mask the played piece
        let piece_range = self.action_set.piece_map[&action.piece_type].clone();
        agent.action_mask[piece_range].fill(false);
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
/// and touches the `diagonals` of the own pieces.
fn check_action_valid(action_board: Bitboard, blocked: Bitboard, diagonals: Bitboard) -> bool {
    (action_board & blocked).is_empty() && !(action_board & diagonals).is_empty()
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
