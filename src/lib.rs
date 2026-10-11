use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod game;

use game::errors::InvalidAction;
use game::{BOARD_SIZE, Game};

#[pyclass(unsendable)]
struct PyBlokus(Game);

#[pymethods]
impl PyBlokus {
    #[new]
    pub fn new() -> Self {
        Self(Game::new())
    }

    pub fn reset(&mut self) {
        self.0 = Game::new();
    }

    #[getter(agents)]
    pub fn agents(&self) -> Vec<usize> {
        (0usize..self.0.num_agents).collect()
    }

    #[getter(agent_selection)]
    pub fn agent_selection(&self) -> usize {
        self.0.agent_selection
    }

    /// Size of the flattened (orientation, row, col) action grid.
    #[getter(num_actions)]
    pub fn num_actions(&self) -> usize {
        self.0.action_set.grid_map.len()
    }

    /// Shape (orientation, row, col) of the action grid, in C order.
    #[getter(action_grid_shape)]
    pub fn action_grid_shape(&self) -> (usize, usize, usize) {
        (
            self.0.action_set.grid_map.len() / (BOARD_SIZE * BOARD_SIZE),
            BOARD_SIZE,
            BOARD_SIZE,
        )
    }

    #[getter(terminations)]
    pub fn terminations(&self) -> [bool; 4] {
        self.0.terminations()
    }

    /// Blokus has no turn limit, so agents are never truncated.
    #[getter(truncations)]
    #[allow(clippy::unused_self)]
    pub fn truncations(&self) -> [bool; 4] {
        [false; 4]
    }

    #[getter(rewards)]
    pub fn rewards(&self) -> Vec<i16> {
        self.0
            .rewards()
            .unwrap_or_else(|| vec![0i16; self.0.num_agents])
    }

    pub fn observe(&mut self, action_idx: usize) -> PyObservation {
        let (observation, action_mask) = self.0.observe(action_idx);
        PyObservation {
            observation,
            action_mask: action_mask.into_iter().map(u8::from).collect(),
        }
    }

    pub fn step(&mut self, action_idx: Option<usize>) -> Result<(), InvalidAction> {
        self.0.step(action_idx)
    }

    pub fn render(&self) {
        self.0.render();
    }
}

#[pyclass(unsendable)]
struct PyObservation {
    observation: [[[bool; BOARD_SIZE]; BOARD_SIZE]; 4],
    action_mask: Vec<u8>,
}

#[pymethods]
impl PyObservation {
    #[getter(observation)]
    pub fn observation(&self) -> [[[bool; BOARD_SIZE]; BOARD_SIZE]; 4] {
        [
            self.observation[0],
            self.observation[1],
            self.observation[2],
            self.observation[3],
        ]
    }

    /// The action mask as bytes of 0/1, cheap to wrap with `numpy.frombuffer`.
    #[getter(action_mask)]
    pub fn action_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.action_mask)
    }
}

impl From<InvalidAction> for PyErr {
    fn from(error: InvalidAction) -> Self {
        PyAssertionError::new_err(error.to_string())
    }
}

#[pymodule]
mod _blokus {
    #[pymodule_export]
    use super::PyBlokus;
    #[pymodule_export]
    use super::PyObservation;
}

#[cfg(test)]
mod tests {
    use crate::game::bitboard::{Bitboard, SEPARATING_BIT_MASK};
    use crate::game::{BOARD_SIZE, Game};

    #[test]
    fn test_action_generation_valid() {
        let game = Game::new();
        assert_eq!(game.action_set.actions.len(), 30433);
        for a in &game.action_set.actions {
            assert!(
                (a.bitboard & !SEPARATING_BIT_MASK) == Bitboard::default(),
                "Invalid action!"
            );
        }
    }
    #[test]
    fn test_rotation_clock_valid() {
        let game = Game::new();
        for a in &game.action_set.actions {
            let board_rot = a
                .bitboard
                .rotate_clock()
                .rotate_clock()
                .rotate_clock()
                .rotate_clock();
            assert!(board_rot == a.bitboard, "Invalid rotation!");
        }
    }
    #[test]
    fn test_rotation_anticlock_valid() {
        let game = Game::new();
        for a in &game.action_set.actions {
            let board_rot = a
                .bitboard
                .rotate_anticlock()
                .rotate_anticlock()
                .rotate_anticlock()
                .rotate_anticlock();
            assert!(board_rot == a.bitboard, "Invalid rotation!");
        }
    }
    #[test]
    fn test_agent_selection() {
        let mut game = Game::new();
        let a400 = Some(game.action_set[400].grid_idx);
        let a0 = Some(game.action_set[0].grid_idx);
        assert_eq!(game.agent_selection, 0);
        let _ = game.step(a400);
        assert_eq!(game.agent_selection, 1);
        let _ = game.step(a400);
        assert_eq!(game.agent_selection, 2);
        let _ = game.step(a400);
        assert_eq!(game.agent_selection, 3);
        let _ = game.step(a0);
        assert_eq!(game.agent_selection, 0);
    }

    #[test]
    fn test_scoring() {
        let mut game = Game::new();
        let a400 = Some(game.action_set[400].grid_idx);
        let a22 = Some(game.action_set[22].grid_idx);
        game.step(a400).unwrap();
        game.step(a400).unwrap();
        game.step(a400).unwrap();
        game.step(a400).unwrap();
        game.observe(0);
        game.step(a22).unwrap();
        game.observe(1);
        game.step(a22).unwrap();
        game.observe(2);
        assert!(game.rewards().is_none());
        game.agents[0].done = true;
        game.agents[1].done = true;
        game.agents[2].done = true;
        game.agents[3].done = true;
        assert_eq!(game.rewards().unwrap()[0], -86);
        assert_eq!(game.rewards().unwrap()[1], -86);
        assert_eq!(game.rewards().unwrap()[2], -87);
        assert_eq!(game.rewards().unwrap()[3], -87);
    }

    #[test]
    fn test_grid_map_bijective() {
        let game = Game::new();
        let action_set = &game.action_set;
        assert_eq!(action_set.grid_map.len(), 91 * BOARD_SIZE * BOARD_SIZE);
        assert_eq!(
            action_set.grid_map.iter().flatten().count(),
            action_set.actions.len()
        );
        for (i, a) in action_set.actions.iter().enumerate() {
            assert_eq!(action_set.grid_map[a.grid_idx], Some(i));
        }
    }

    #[test]
    fn test_grid_idx_is_translation() {
        // the action at (orientation, row, col) is the one at (orientation, 0, 0) shifted by (row, col)
        let game = Game::new();
        let action_set = &game.action_set;
        for a in &action_set.actions {
            let orientation = a.grid_idx / (BOARD_SIZE * BOARD_SIZE);
            let cell = a.grid_idx % (BOARD_SIZE * BOARD_SIZE);
            let origin_idx = action_set.grid_map[orientation * BOARD_SIZE * BOARD_SIZE].unwrap();
            let origin = action_set.actions[origin_idx];
            assert_eq!(origin.piece_type, a.piece_type);
            assert_eq!(
                origin
                    .bitboard
                    .translate_origin(cell % BOARD_SIZE, cell / BOARD_SIZE),
                a.bitboard
            );
        }
    }

    #[test]
    fn test_grid_mask_matches_action_mask() {
        let mut game = Game::new();
        for turn in 0..40 {
            let agent = game.agent_selection;
            let grid = game.observe(agent).1;
            let mask = &game.agents[agent].action_mask;
            assert_eq!(
                grid.iter().filter(|x| **x).count(),
                mask.iter().filter(|x| **x).count()
            );
            for (a, valid) in game.action_set.actions.iter().zip(mask) {
                assert_eq!(grid[a.grid_idx], *valid);
            }
            // vary the played actions without pulling in an RNG, agents without actions pass
            let valid: Vec<usize> = (0..grid.len()).filter(|i| grid[*i]).collect();
            let action = valid.get(turn * 7 % valid.len().max(1)).copied();
            game.step(action).unwrap();
        }
    }

    #[test]
    fn test_step_rejects_unavailable_grid_idx() {
        let mut game = Game::new();
        let unmapped = game.action_set.grid_map.iter().position(Option::is_none);
        let out_of_range = Some(game.action_set.grid_map.len());
        let grid = game.observe(0).1;
        let masked = (0..grid.len()).find(|i| game.action_set.grid_map[*i].is_some() && !grid[*i]);
        for action in [unmapped, out_of_range, masked, None] {
            assert!(game.step(action).is_err());
            assert_eq!(game.agent_selection, 0);
        }
        let valid = (0..grid.len()).find(|i| grid[*i]).unwrap();
        game.step(Some(valid)).unwrap();
        assert_eq!(game.agent_selection, 1);
        let action = game.action_set[game.action_set.grid_map[valid].unwrap()];
        assert_eq!(game.agents[0].board, action.bitboard);
    }

    #[test]
    fn test_done_agent_passes_any_action() {
        let mut game = Game::new();
        let unmapped = game.action_set.grid_map.iter().position(Option::is_none);
        game.agents[0].done = true;
        game.step(None).unwrap();
        assert_eq!(game.agent_selection, 1);
        let a400 = Some(game.action_set[400].grid_idx);
        for _ in 0..3 {
            game.step(a400).unwrap();
        }
        game.step(unmapped).unwrap();
        assert_eq!(game.agent_selection, 1);
        assert_eq!(game.agents[0].board, Bitboard::default());
    }
}
