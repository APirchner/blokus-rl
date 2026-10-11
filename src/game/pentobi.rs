//! Conversions between actions and the moves of the [Pentobi](https://pentobi.sourceforge.io/)
//! engine, which talks [GTP](https://github.com/enz/pentobi/blob/main/pentobi_gtp/Pentobi-GTP.md).
//!
//! A Pentobi move lists its cells, such as `f9,e10,f10,g10,f11`. Pentobi's colors 1 to 4 are the
//! agents 0 to 3, and its board is the frame of agent 0: columns `a` to `t` from the left, rows
//! `1` to `20` from the bottom.

use super::bitboard::Bitboard;
use super::errors::InvalidAction;
use super::{ActionSet, BOARD_SIZE, Game};

/// The agent's action, in its frame, as a Pentobi move with its cells in Pentobi's order: `a1`,
/// `b1`, ..., `a2`.
///
/// # Panics
///
/// If `agent` is not 0 to 3, or `action` is not below 30,433, the number of actions.
#[must_use]
pub fn to_pentobi(agent: usize, action: u16) -> String {
    assert!(agent < 4, "No agent {agent}");
    let action_set = ActionSet::get();
    let global = action_set.rotated[agent][usize::from(action)];
    let mut cells: Vec<(usize, usize)> = action_set.actions[usize::from(global)]
        .bitboard
        .ones()
        .map(|idx| (BOARD_SIZE - idx / (BOARD_SIZE + 1), idx % (BOARD_SIZE + 1)))
        .collect();
    cells.sort_unstable();
    cells
        .iter()
        .map(|(row, col)| format!("{}{row}", char::from(b"abcdefghijklmnopqrst"[*col])))
        .collect::<Vec<_>>()
        .join(",")
}

/// The agent's action, in its frame, for a Pentobi move. None if the move is not a placement.
///
/// # Panics
///
/// If `agent` is not 0 to 3.
#[must_use]
pub fn from_pentobi(agent: usize, mv: &str) -> Option<u16> {
    assert!(agent < 4, "No agent {agent}");
    let mut bitboard = Bitboard::default();
    let mut cells = 0;
    for cell in mv.split(',') {
        let (col, row) = cell.split_at_checked(1)?;
        let col = usize::from(col.as_bytes()[0].to_ascii_lowercase().checked_sub(b'a')?);
        // as Pentobi, no sign or leading zeros
        let row = row.parse::<usize>().ok().filter(|r| r.to_string() == row)?;
        if col >= BOARD_SIZE || !(1..=BOARD_SIZE).contains(&row) {
            return None;
        }
        bitboard |= Bitboard::cell(BOARD_SIZE - row, col);
        cells += 1;
    }
    if bitboard.ones().count() != cells {
        return None;
    }
    let action_set = ActionSet::get();
    let global = action_set.find(bitboard)?;
    Some(action_set.rotated[(4 - agent) % 4][usize::from(global)])
}

impl Game {
    /// The game after `moves`, each an agent and its Pentobi move, such as a game played by
    /// Pentobi. `pass` is accepted from an agent that is out of moves, on its turn in the turn
    /// order.
    ///
    /// # Errors
    ///
    /// If a move is not legal, or not played by the agent to move.
    pub fn from_moves<'a>(
        moves: impl IntoIterator<Item = (usize, &'a str)>,
    ) -> Result<Self, InvalidAction> {
        let mut game = Self::new();
        for (agent, mv) in moves {
            let error = || InvalidAction::new(format!("Agent {agent} can not play {mv}"));
            if mv == "pass" {
                // the turn of the agent after the last move, even if it is out of moves
                if agent != game.agent_selection || !game.agents[agent].done {
                    return Err(error());
                }
                game.agent_selection = (agent + 1) % 4;
                continue;
            }
            if agent != game.current_player() {
                return Err(error());
            }
            let action = from_pentobi(agent, mv)
                .filter(|a| game.is_legal(agent, usize::from(*a)))
                .ok_or_else(error)?;
            game.apply(action);
        }
        Ok(game)
    }
}
