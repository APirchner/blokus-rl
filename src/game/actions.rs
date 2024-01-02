use std::collections::HashMap;
use std::iter;
use std::ops::{Index, Range};

use crate::game::bitboard::Bitboard;
use crate::game::pieces::{Piece, PieceType};

#[derive(Clone, Copy, Debug)]
pub struct Action {
    pub piece_type: PieceType,
    pub bitboard: Bitboard,
}

pub struct ActionSet {
    pub piece_map: HashMap<PieceType, Range<usize>>,
    pub actions: Vec<Action>,
}

impl ActionSet {
    pub fn new(pieces: &[Piece]) -> Self {
        let mut piece_map: HashMap<PieceType, Range<usize>> = HashMap::new();
        let mut actions: Vec<Action> = Vec::new();
        let mut action_idx = 0usize;
        for p in pieces {
            let piece_placements = p.build_translations();
            let piece_actions: Vec<Action> = iter::zip(piece_placements, iter::repeat(p))
                .enumerate()
                .map(|x| Action {
                    piece_type: x.1 .1.name,
                    bitboard: x.1 .0,
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
        Self { piece_map, actions }
    }

    pub fn initial_actions(&self) -> Vec<bool> {
        let mut init_actions = Vec::with_capacity(self.actions.len());
        for p in &self.actions {
            init_actions.push(!(p.bitboard & Bitboard(0, 0, 0, 1)).is_empty());
        }
        init_actions
    }
}

impl Index<PieceType> for ActionSet {
    type Output = [Action];

    fn index(&self, index: PieceType) -> &Self::Output {
        &self.actions[self.piece_map[&index].start..self.piece_map[&index].end]
    }
}

impl Index<usize> for ActionSet {
    type Output = Action;

    fn index(&self, index: usize) -> &Self::Output {
        &self.actions[index]
    }
}
