use std::cmp::Ordering;

use super::bitboard::{Bitboard, BOARD_SIZE};

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Debug)]
pub enum PieceType {
    I1 = 1,
    I2 = 2,
    V3 = 3,
    I3 = 9,
    T4 = 10,
    O = 16,
    L4 = 22,
    I4 = 28,
    Z4 = 34,
    F = 35,
    X = 41,
    P = 47,
    W = 53,
    Z5 = 59,
    Y = 65,
    L5 = 71,
    U = 77,
    T5 = 83,
    V5 = 89,
    N = 95,
    I5 = 101,
}

impl PieceType {
    pub fn size(self) -> u8 {
        self as u8 % 6
    }

    pub fn iter() -> impl Iterator<Item = PieceType> {
        [
            PieceType::I1,
            PieceType::I2,
            PieceType::V3,
            PieceType::I3,
            PieceType::T4,
            PieceType::O,
            PieceType::L4,
            PieceType::I4,
            PieceType::Z4,
            PieceType::F,
            PieceType::X,
            PieceType::P,
            PieceType::W,
            PieceType::Z5,
            PieceType::Y,
            PieceType::L5,
            PieceType::U,
            PieceType::T5,
            PieceType::V5,
            PieceType::N,
            PieceType::I5,
        ]
        .iter()
        .copied()
    }
}

struct RowBuilder(usize, usize);

impl RowBuilder {
    fn build(self) -> RowEncoding {
        RowEncoding(self.0, self.1, None)
    }
    fn build_skip(self, skip_at: usize) -> RowEncoding {
        RowEncoding(self.0, self.1, Some(skip_at))
    }
}

#[derive(Clone)]
struct RowEncoding(usize, usize, Option<usize>);

impl RowEncoding {
    fn decode_row(&self) -> Vec<u8> {
        let mut row = match self.0 {
            0 => [vec![1; self.1], vec![0; BOARD_SIZE - self.1]].concat(),
            _ => [
                vec![0; self.0],
                vec![1; self.1],
                vec![0; BOARD_SIZE - self.0 - self.1],
            ]
            .concat(),
        };
        if let Some(x) = self.2 {
            row.insert(x, 0);
            row.pop();
        }
        row
    }
}

#[derive(Clone)]
pub struct Piece {
    pub name: PieceType,
    encoding: Vec<RowEncoding>,
    rotation: usize,
    max_row: usize,
    max_col: usize,
}

impl Piece {
    fn decode(&self) -> u128 {
        let bits: Vec<u8> = self.encoding.iter().fold(Vec::new(), |acc, enc| {
            [acc, enc.decode_row(), vec![0; 1]].concat()
        });
        let parse_bits = |acc: u128, bit: &u8| (acc << 1) + u128::from(*bit);
        bits.iter().rev().fold(0, parse_bits)
    }

    fn to_bitboard(&self) -> Bitboard {
        Bitboard(0, 0, 0, self.decode())
    }

    pub fn build_translations(&self) -> Vec<Bitboard> {
        let mut trans: Vec<Bitboard> = Vec::with_capacity(self.max_row * self.max_col);
        for i in 0..=self.max_row {
            for j in 0..=self.max_col {
                trans.push(self.to_bitboard().translate_origin(j, i));
            }
        }
        trans
    }
}

impl PartialEq for Piece {
    fn eq(&self, other: &Self) -> bool {
        (self.name, self.rotation) == (other.name, other.rotation)
    }
}

impl Ord for Piece {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.name, self.rotation).cmp(&(other.name, other.rotation))
    }
}

impl Eq for Piece {}

impl PartialOrd for Piece {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[allow(clippy::too_many_lines)]
pub fn generate() -> Vec<Piece> {
    let mut pieces = vec![
        // I1, 1x
        Piece {
            name: PieceType::I1,
            encoding: vec![RowBuilder(0, 1).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 1,
            max_col: BOARD_SIZE - 1,
        },
        // I2, 2x
        Piece {
            name: PieceType::I2,
            encoding: vec![RowBuilder(0, 2).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 1,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::I2,
            encoding: vec![RowBuilder(0, 1).build(); 2],
            rotation: 1,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 1,
        },
        // V3, 4x
        Piece {
            name: PieceType::V3,
            encoding: vec![RowBuilder(0, 2).build(), RowBuilder(0, 1).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::V3,
            encoding: vec![RowBuilder(0, 1).build(), RowBuilder(0, 2).build()],
            rotation: 1,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::V3,
            encoding: vec![RowBuilder(1, 1).build(), RowBuilder(0, 2).build()],
            rotation: 2,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::V3,
            encoding: vec![RowBuilder(0, 2).build(), RowBuilder(1, 1).build()],
            rotation: 3,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 2,
        },
        // I3, 2x
        Piece {
            name: PieceType::I3,
            encoding: vec![RowBuilder(0, 3).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 1,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::I3,
            encoding: vec![RowBuilder(0, 1).build(); 3],
            rotation: 1,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 1,
        },
        // T4, 4x
        Piece {
            name: PieceType::T4,
            encoding: vec![RowBuilder(1, 1).build(), RowBuilder(0, 3).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::T4,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 1,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::T4,
            encoding: vec![RowBuilder(0, 3).build(), RowBuilder(1, 1).build()],
            rotation: 2,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::T4,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 3,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        // O, 1x
        Piece {
            name: PieceType::O,
            encoding: vec![RowBuilder(0, 2).build(), RowBuilder(0, 2).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 2,
        },
        // L4, 8x
        Piece {
            name: PieceType::L4,
            encoding: vec![RowBuilder(0, 1).build(), RowBuilder(0, 3).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::L4,
            encoding: vec![RowBuilder(0, 3).build(), RowBuilder(0, 1).build()],
            rotation: 1,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::L4,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 2).build(),
            ],
            rotation: 2,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::L4,
            encoding: vec![
                RowBuilder(0, 2).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 3,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::L4,
            encoding: vec![
                RowBuilder(0, 2).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 4,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::L4,
            encoding: vec![RowBuilder(0, 3).build(), RowBuilder(2, 1).build()],
            rotation: 5,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::L4,
            encoding: vec![RowBuilder(2, 1).build(), RowBuilder(0, 3).build()],
            rotation: 6,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::L4,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
            ],
            rotation: 7,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        // I4, 2x
        Piece {
            name: PieceType::I4,
            encoding: vec![RowBuilder(0, 4).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 1,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::I4,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 1,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 1,
        },
        // Z4, 4x
        Piece {
            name: PieceType::Z4,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 0,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::Z4,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 1,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::Z4,
            encoding: vec![RowBuilder(1, 2).build(), RowBuilder(0, 2).build()],
            rotation: 2,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::Z4,
            encoding: vec![RowBuilder(0, 2).build(), RowBuilder(1, 2).build()],
            rotation: 3,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        // F, 8x
        Piece {
            name: PieceType::F,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(0, 3).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 0,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::F,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 3).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 1,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::F,
            encoding: vec![
                RowBuilder(0, 2).build(),
                RowBuilder(1, 2).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 2,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::F,
            encoding: vec![
                RowBuilder(1, 2).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 3,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::F,
            encoding: vec![
                RowBuilder(2, 1).build(),
                RowBuilder(0, 3).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 4,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::F,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(1, 2).build(),
            ],
            rotation: 5,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::F,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(0, 3).build(),
                RowBuilder(2, 1).build(),
            ],
            rotation: 6,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::F,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(1, 2).build(),
                RowBuilder(0, 2).build(),
            ],
            rotation: 7,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        // X, 1x
        Piece {
            name: PieceType::X,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(0, 3).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 0,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        // P, 8x
        Piece {
            name: PieceType::P,
            encoding: vec![
                RowBuilder(0, 2).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 0,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::P,
            encoding: vec![
                RowBuilder(0, 2).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 1,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::P,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(0, 2).build(),
            ],
            rotation: 2,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::P,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(0, 2).build(),
            ],
            rotation: 3,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::P,
            encoding: vec![RowBuilder(0, 3).build(), RowBuilder(1, 2).build()],
            rotation: 4,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::P,
            encoding: vec![RowBuilder(1, 2).build(), RowBuilder(0, 3).build()],
            rotation: 5,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::P,
            encoding: vec![RowBuilder(0, 3).build(), RowBuilder(0, 2).build()],
            rotation: 6,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::P,
            encoding: vec![RowBuilder(0, 2).build(), RowBuilder(0, 3).build()],
            rotation: 7,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        // W, 4x
        Piece {
            name: PieceType::W,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(1, 2).build(),
            ],
            rotation: 0,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::W,
            encoding: vec![
                RowBuilder(1, 2).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 1,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::W,
            encoding: vec![
                RowBuilder(2, 1).build(),
                RowBuilder(1, 2).build(),
                RowBuilder(0, 2).build(),
            ],
            rotation: 2,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::W,
            encoding: vec![
                RowBuilder(0, 2).build(),
                RowBuilder(1, 2).build(),
                RowBuilder(2, 1).build(),
            ],
            rotation: 3,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        // Z5, 4x
        Piece {
            name: PieceType::Z5,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 3).build(),
                RowBuilder(2, 1).build(),
            ],
            rotation: 0,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::Z5,
            encoding: vec![
                RowBuilder(2, 1).build(),
                RowBuilder(0, 3).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 1,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::Z5,
            encoding: vec![
                RowBuilder(1, 2).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
            ],
            rotation: 2,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::Z5,
            encoding: vec![
                RowBuilder(0, 2).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(1, 2).build(),
            ],
            rotation: 3,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        // Y, 8x
        Piece {
            name: PieceType::Y,
            encoding: vec![RowBuilder(0, 4).build(), RowBuilder(1, 1).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::Y,
            encoding: vec![RowBuilder(1, 1).build(), RowBuilder(0, 4).build()],
            rotation: 1,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::Y,
            encoding: vec![RowBuilder(0, 4).build(), RowBuilder(2, 1).build()],
            rotation: 2,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::Y,
            encoding: vec![RowBuilder(2, 1).build(), RowBuilder(0, 4).build()],
            rotation: 3,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::Y,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 4,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::Y,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 5,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::Y,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 6,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::Y,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 7,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 2,
        },
        // L5, 8x
        Piece {
            name: PieceType::L5,
            encoding: vec![RowBuilder(0, 4).build(), RowBuilder(0, 1).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::L5,
            encoding: vec![RowBuilder(0, 1).build(), RowBuilder(0, 4).build()],
            rotation: 1,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::L5,
            encoding: vec![RowBuilder(0, 4).build(), RowBuilder(3, 1).build()],
            rotation: 2,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::L5,
            encoding: vec![RowBuilder(3, 1).build(), RowBuilder(0, 4).build()],
            rotation: 3,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::L5,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 2).build(),
            ],
            rotation: 4,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::L5,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
            ],
            rotation: 5,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::L5,
            encoding: vec![
                RowBuilder(0, 2).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 6,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::L5,
            encoding: vec![
                RowBuilder(0, 2).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 7,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 2,
        },
        // U, 4x
        Piece {
            name: PieceType::U,
            encoding: vec![RowBuilder(0, 2).build_skip(1), RowBuilder(0, 3).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::U,
            encoding: vec![RowBuilder(0, 3).build(), RowBuilder(0, 2).build_skip(1)],
            rotation: 1,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::U,
            encoding: vec![
                RowBuilder(0, 2).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 2).build(),
            ],
            rotation: 2,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::U,
            encoding: vec![
                RowBuilder(0, 2).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
            ],
            rotation: 3,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 2,
        },
        // T5, 4x
        Piece {
            name: PieceType::T5,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(0, 3).build(),
            ],
            rotation: 0,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::T5,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 3).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 1,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::T5,
            encoding: vec![
                RowBuilder(0, 3).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 2,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::T5,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(0, 3).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 3,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        // V5, 4x
        Piece {
            name: PieceType::V5,
            encoding: vec![
                RowBuilder(0, 3).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 0,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::V5,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 3).build(),
            ],
            rotation: 1,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::V5,
            encoding: vec![
                RowBuilder(2, 1).build(),
                RowBuilder(2, 1).build(),
                RowBuilder(0, 3).build(),
            ],
            rotation: 2,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        Piece {
            name: PieceType::V5,
            encoding: vec![
                RowBuilder(0, 3).build(),
                RowBuilder(2, 1).build(),
                RowBuilder(2, 1).build(),
            ],
            rotation: 3,
            max_row: BOARD_SIZE - 3,
            max_col: BOARD_SIZE - 3,
        },
        // N, 8x
        Piece {
            name: PieceType::N,
            encoding: vec![RowBuilder(1, 3).build(), RowBuilder(0, 2).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::N,
            encoding: vec![RowBuilder(0, 2).build(), RowBuilder(1, 3).build()],
            rotation: 1,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::N,
            encoding: vec![RowBuilder(0, 3).build(), RowBuilder(2, 2).build()],
            rotation: 2,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::N,
            encoding: vec![RowBuilder(2, 2).build(), RowBuilder(0, 3).build()],
            rotation: 3,
            max_row: BOARD_SIZE - 2,
            max_col: BOARD_SIZE - 4,
        },
        Piece {
            name: PieceType::N,
            encoding: vec![
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(1, 1).build(),
            ],
            rotation: 4,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::N,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 5,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::N,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 6,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 2,
        },
        Piece {
            name: PieceType::N,
            encoding: vec![
                RowBuilder(1, 1).build(),
                RowBuilder(0, 2).build(),
                RowBuilder(0, 1).build(),
                RowBuilder(0, 1).build(),
            ],
            rotation: 7,
            max_row: BOARD_SIZE - 4,
            max_col: BOARD_SIZE - 2,
        },
        // I5, 2x
        Piece {
            name: PieceType::I5,
            encoding: vec![RowBuilder(0, 5).build()],
            rotation: 0,
            max_row: BOARD_SIZE - 1,
            max_col: BOARD_SIZE - 5,
        },
        Piece {
            name: PieceType::I5,
            encoding: vec![RowBuilder(0, 1).build(); 5],
            rotation: 1,
            max_row: BOARD_SIZE - 5,
            max_col: BOARD_SIZE - 1,
        },
    ];
    pieces.sort();
    pieces
}
