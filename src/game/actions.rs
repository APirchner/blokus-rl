use std::collections::HashMap;
use std::ops::Index;
use std::sync::OnceLock;

use crate::game::bitboard::{BOARD_SIZE, Bitboard, MAX_IDX};
use crate::game::pieces::{self, Piece, PieceType};

/// The action set is identical for every game, so it is built once and shared across resets.
static ACTION_SET: OnceLock<ActionSet> = OnceLock::new();

#[derive(Clone, Copy, Debug)]
pub struct Action {
    pub piece_type: PieceType,
    pub bitboard: Bitboard,
    /// Index into the (orientation, row, col) action grid, see `ActionSet::grid_map`.
    pub grid_idx: usize,
}

/// All placements of all pieces on the board. Actions are indexed the same in every agent's
/// frame, and the indices fit in a `u16`.
pub struct ActionSet {
    pub(crate) actions: Vec<Action>,
    /// Maps each index of the flattened (orientation, row, col) grid to its action index, if the
    /// orientation fits on the board at that (row, col).
    pub(crate) grid_map: Vec<Option<usize>>,
    /// `rotated[k][a]` is the action `a` rotated by `k` quarter turns clockwise, which takes it
    /// from the frame of agent `k` to the frame of agent 0.
    pub(crate) rotated: [Vec<u16>; 4],
    /// The actions covering each bit of the board, grouped by piece, see `covering`.
    cover: Vec<u16>,
    /// Where the group of each (bit, piece) starts in `cover`, it ends where the next one starts.
    cover_start: Vec<usize>,
}

impl ActionSet {
    pub(crate) fn get() -> &'static Self {
        ACTION_SET.get_or_init(|| ActionSet::new(&pieces::generate()))
    }

    /// Index of the action in the (orientation, row, col) policy grid of shape
    /// `ACTION_GRID_SHAPE`, flattened in C order.
    ///
    /// # Panics
    ///
    /// If `action` is not below 30,433, the number of actions.
    #[must_use]
    pub fn grid_idx(action: u16) -> usize {
        Self::get().actions[usize::from(action)].grid_idx
    }

    /// The actions of the piece `PieceType::ALL[piece]` that cover the bit `idx` of the board.
    pub(crate) fn covering(&self, idx: usize, piece: usize) -> &[u16] {
        let group = idx * PieceType::ALL.len() + piece;
        &self.cover[self.cover_start[group]..self.cover_start[group + 1]]
    }

    /// The action that covers exactly the cells of `bitboard`, if there is one.
    pub(crate) fn find(&self, bitboard: Bitboard) -> Option<u16> {
        let idx = bitboard.ones().next()?;
        (0..PieceType::ALL.len())
            .flat_map(|piece| self.covering(idx, piece))
            .copied()
            .find(|a| self.actions[usize::from(*a)].bitboard == bitboard)
    }

    fn new(pieces: &[Piece]) -> Self {
        let mut actions: Vec<Action> = Vec::new();
        for (orientation, p) in pieces.iter().enumerate() {
            let piece_placements = p.build_translations();
            let piece_actions = piece_placements.into_iter().map(|(cell, bitboard)| Action {
                piece_type: p.name,
                bitboard,
                grid_idx: orientation * BOARD_SIZE * BOARD_SIZE + cell,
            });
            actions.extend(piece_actions);
        }
        let mut grid_map = vec![None; pieces.len() * BOARD_SIZE * BOARD_SIZE];
        for (i, a) in actions.iter().enumerate() {
            grid_map[a.grid_idx] = Some(i);
        }
        let ids: Vec<u16> = (0..actions.len())
            .map(|i| u16::try_from(i).unwrap())
            .collect();
        let by_bitboard: HashMap<Bitboard, u16> = actions
            .iter()
            .map(|a| a.bitboard)
            .zip(ids.iter().copied())
            .collect();
        // a rotated placement of a piece is another placement of it
        let clockwise: Vec<u16> = actions
            .iter()
            .map(|a| by_bitboard[&a.bitboard.rotate_clock()])
            .collect();
        let mut rotated = [ids.clone(), vec![], vec![], vec![]];
        for k in 1..4 {
            rotated[k] = rotated[k - 1]
                .iter()
                .map(|a| clockwise[usize::from(*a)])
                .collect();
        }
        let mut groups = vec![Vec::new(); (MAX_IDX + 1) * PieceType::ALL.len()];
        for (i, a) in actions.iter().enumerate() {
            for idx in a.bitboard.ones() {
                groups[idx * PieceType::ALL.len() + a.piece_type.index()].push(ids[i]);
            }
        }
        let mut cover_start = vec![0];
        cover_start.extend(groups.iter().scan(0, |end, g| {
            *end += g.len();
            Some(*end)
        }));
        Self {
            actions,
            grid_map,
            rotated,
            cover: groups.concat(),
            cover_start,
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
