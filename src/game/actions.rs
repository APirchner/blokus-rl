use std::collections::HashMap;
use std::ops::{Index, Range};

use crate::game::bitboard::{BOARD_SIZE, Bitboard};
use crate::game::pieces::{Piece, PieceType};

#[derive(Clone, Copy, Debug)]
pub struct Action {
    pub piece_type: PieceType,
    pub bitboard: Bitboard,
    /// Index into the (orientation, row, col) action grid, see `ActionSet::grid_map`.
    pub grid_idx: usize,
}

pub struct ActionSet {
    pub piece_map: HashMap<PieceType, Range<usize>>,
    pub actions: Vec<Action>,
    /// Maps each index of the flattened (orientation, row, col) grid to its action index, if the
    /// orientation fits on the board at that (row, col).
    pub grid_map: Vec<Option<usize>>,
}

impl ActionSet {
    pub fn new(pieces: &[Piece]) -> Self {
        let mut piece_map: HashMap<PieceType, Range<usize>> = HashMap::new();
        let mut actions: Vec<Action> = Vec::new();
        let mut action_idx = 0usize;
        for (orientation, p) in pieces.iter().enumerate() {
            let piece_placements = p.build_translations();
            let piece_actions: Vec<Action> = piece_placements
                .into_iter()
                .map(|(cell, bitboard)| Action {
                    piece_type: p.name,
                    bitboard,
                    grid_idx: orientation * BOARD_SIZE * BOARD_SIZE + cell,
                })
                .collect();
            let piece_tmp = piece_map.get(&p.name);
            match piece_tmp {
                Some(x) => {
                    piece_map.insert(
                        p.name,
                        Range {
                            start: x.start,
                            end: action_idx + piece_actions.len(),
                        },
                    );
                }
                None => {
                    piece_map.insert(
                        p.name,
                        Range {
                            start: action_idx,
                            end: action_idx + piece_actions.len(),
                        },
                    );
                }
            }
            action_idx += piece_actions.len();
            actions.extend(piece_actions);
        }
        let mut grid_map = vec![None; pieces.len() * BOARD_SIZE * BOARD_SIZE];
        for (i, a) in actions.iter().enumerate() {
            grid_map[a.grid_idx] = Some(i);
        }
        Self {
            piece_map,
            actions,
            grid_map,
        }
    }

    /// Scatters the indices of the legal actions into the flattened (orientation, row, col) grid,
    /// as bytes of 0/1.
    pub fn grid_mask(&self, legal: impl Iterator<Item = usize>) -> Vec<u8> {
        let mut grid = vec![0; self.grid_map.len()];
        for a in legal {
            grid[self.actions[a].grid_idx] = 1;
        }
        grid
    }
}

impl Index<usize> for ActionSet {
    type Output = Action;

    fn index(&self, index: usize) -> &Self::Output {
        &self.actions[index]
    }
}
