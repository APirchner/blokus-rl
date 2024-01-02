use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;

mod game;

use game::errors::InvalidAction;
use game::{Game, BOARD_SIZE};

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

    #[getter(num_actions)]
    pub fn num_actions(&self) -> usize {
        self.0.action_set.actions.len()
    }

    #[getter(terminations)]
    pub fn terminations(&self) -> [bool; 4] {
        self.0.terminations()
    }

    #[getter(truncations)]
    pub fn truncations(&self) -> [bool; 4] {
        self.0.terminations()
    }

    #[getter(rewards)]
    pub fn rewards(&self) -> Vec<u8> {
        self.0
            .rewards()
            .map_or_else(|| vec![0u8; self.0.num_agents], |x| x)
    }

    pub fn observe(&mut self, action_idx: usize) -> PyObservation {
        let obs = self.0.observe(action_idx);
        PyObservation {
            observation: obs.0,
            action_mask: obs.1.to_vec(),
        }
    }

    pub fn step(&mut self, action_idx: usize) -> Result<(), InvalidAction> {
        self.0.step(action_idx)?;
        Ok(())
    }

    pub fn render(&self) {
        self.0.render();
    }
}

#[pyclass(unsendable)]
struct PyObservation {
    observation: [[[bool; BOARD_SIZE]; BOARD_SIZE]; 4],
    action_mask: Vec<bool>,
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

    #[getter(action_mask)]
    pub fn action_mask(&self) -> Vec<bool> {
        self.action_mask.clone()
    }
}

#[pymodule]
fn _blokus(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBlokus>()?;
    m.add_class::<PyObservation>()?;
    Ok(())
}

impl From<InvalidAction> for PyErr {
    fn from(error: InvalidAction) -> Self {
        PyAssertionError::new_err(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use crate::game::bitboard::{separating_bit_mask, Bitboard};
    use crate::game::Game;

    #[test]
    fn test_action_generation_valid() {
        let game = Game::new();
        for a in game.action_set.actions.into_iter() {
            if (a.bitboard & !separating_bit_mask()) != Bitboard::default() {
                println!("{:?}", a);
                println!("{}", a.bitboard);
                panic!("Invalid action!");
            }
        }
    }
    #[test]
    fn test_rotation_clock_valid() {
        let game = Game::new();
        for a in game.action_set.actions.into_iter() {
            let board_rot = a
                .bitboard
                .rotate_clock()
                .rotate_clock()
                .rotate_clock()
                .rotate_clock();
            if board_rot != a.bitboard {
                println!("{:?}", a);
                println!("{}", a.bitboard);
                println!("{}", board_rot);
                panic!("Invalid rotation!");
            }
        }
    }
    #[test]
    fn test_rotation_anticlock_valid() {
        let game = Game::new();
        for a in game.action_set.actions.into_iter() {
            let board_rot = a
                .bitboard
                .rotate_anticlock()
                .rotate_anticlock()
                .rotate_anticlock()
                .rotate_anticlock();
            if board_rot != a.bitboard {
                println!("{:?}", a);
                println!("{}", a.bitboard);
                println!("{}", board_rot);
                panic!("Invalid rotation!");
            }
        }
    }
    #[test]
    fn test_agent_selection() {
        let mut game = Game::new();
        assert_eq!(game.agent_selection, 0);
        let _ = game.step(400);
        assert_eq!(game.agent_selection, 1);
        let _ = game.step(400);
        assert_eq!(game.agent_selection, 2);
        let _ = game.step(400);
        assert_eq!(game.agent_selection, 3);
        let _ = game.step(0);
        assert_eq!(game.agent_selection, 0);
    }

    #[test]
    fn test_scoring() {
        let mut game = Game::new();
        game.step(400).unwrap();
        game.step(400).unwrap();
        game.step(400).unwrap();
        game.step(400).unwrap();
        game.observe(0);
        game.step(22).unwrap();
        game.observe(1);
        game.step(22).unwrap();
        game.observe(2);
        assert!(game.rewards().is_none());
        game.agents[0].done = true;
        game.agents[1].done = true;
        game.agents[2].done = true;
        game.agents[3].done = true;
        assert_eq!(game.rewards().unwrap()[0], 1);
        assert_eq!(game.rewards().unwrap()[1], 1);
        assert_eq!(game.rewards().unwrap()[2], 0);
        assert_eq!(game.rewards().unwrap()[3], 0);
    }
}
