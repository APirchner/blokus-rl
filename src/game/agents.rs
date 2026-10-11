use std::fmt;

use super::bitboard::Bitboard;
use super::pieces::PieceType;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Color {
    Blue = 0,
    Yellow = 1,
    Red = 2,
    Green = 3,
}

/// All pieces available, bit `i` stands for `PieceType::ALL[i]`.
pub const ALL_PIECES: u32 = (1 << PieceType::ALL.len()) - 1;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Agent {
    pub color: Color,
    pub board: Bitboard,
    pub turn: u8,
    /// Bit `i` is set while the agent still has the piece `PieceType::ALL[i]`.
    pub pieces: u32,
    pub last_piece: Option<PieceType>,
    pub done: bool,
}

impl Agent {
    pub fn new(color: Color) -> Self {
        let board = Bitboard::default();
        let turn = 0;
        let done = false;
        Self {
            color,
            board,
            turn,
            pieces: ALL_PIECES,
            last_piece: None,
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
