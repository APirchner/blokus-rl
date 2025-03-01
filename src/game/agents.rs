use std::collections::HashMap;
use std::fmt;
use std::iter;

use super::bitboard::Bitboard;
use super::pieces::PieceType;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Color {
    Blue = 0,
    Yellow = 1,
    Red = 2,
    Green = 3,
}

pub struct Agent {
    pub color: Color,
    pub board: Bitboard,
    pub turn: u8,
    pub pieces: HashMap<PieceType, bool>,
    pub action_mask: Vec<bool>,
    pub action_mask_stale: bool,
    pub done: bool,
}

impl Agent {
    pub fn new(color: Color, initial_actions: Vec<bool>) -> Self {
        let board = Bitboard::default();
        let turn = 0;
        let done = false;
        let action_mask_stale = false;
        let action_mask = initial_actions;
        let pieces: HashMap<PieceType, bool> =
            iter::zip(PieceType::iter(), iter::repeat(true)).collect();
        Self {
            color,
            board,
            turn,
            pieces,
            action_mask,
            action_mask_stale,
            done,
        }
    }
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("color", &self.color)
            .field("board", &self.board)
            .field("turn", &self.turn)
            .field("pieces", &self.pieces)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}
