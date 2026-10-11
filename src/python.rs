//! The Python extension `blokus_rl._blokus`.

use pyo3::exceptions::{PyAssertionError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use crate::game::errors::InvalidAction;
use crate::game::{BOARD_SIZE, GLOBALS_LEN, Game, PLANES_LEN, PieceType, RewardMode};

#[pyclass]
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

    /// An independent copy of the game.
    pub fn copy(&self) -> Self {
        Self(self.0)
    }

    /// The game state as bytes, restore it with `from_bytes`.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.0.to_bytes())
    }

    #[staticmethod]
    pub fn from_bytes(data: &[u8]) -> PyResult<Self> {
        Game::from_bytes(data)
            .map(Self)
            .ok_or_else(|| PyValueError::new_err("Not a game state written by to_bytes()"))
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

    /// The piece names, in the order of the observation's piece axis.
    #[classattr]
    pub fn piece_names(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, PieceType::ALL.map(|p| format!("{p:?}")))
    }

    #[getter(terminations)]
    pub fn terminations(&self) -> [bool; 4] {
        self.0.terminations()
    }

    /// The score of every agent that is done, None for the others.
    #[getter(rewards)]
    pub fn rewards(&self) -> Vec<Option<i16>> {
        self.0.rewards()
    }

    /// Every agent's official score as if the game ended now.
    #[getter(scores)]
    pub fn scores(&self) -> [i16; 4] {
        self.0.scores()
    }

    #[getter(is_over)]
    pub fn is_over(&self) -> bool {
        self.0.is_over()
    }

    /// The number of placements made so far, by all agents.
    #[getter(ply)]
    pub fn ply(&self) -> u16 {
        self.0.ply()
    }

    /// The final rewards in `mode` (`"score"`, `"win_share"` or `"rank"`), None until the game is
    /// over.
    pub fn outcome(&self, mode: &str) -> PyResult<Option<[f32; 4]>> {
        let mode = match mode {
            "score" => RewardMode::Score,
            "win_share" => RewardMode::WinShare,
            "rank" => RewardMode::Rank,
            _ => return Err(PyValueError::new_err(format!("Unknown reward mode {mode}"))),
        };
        Ok(self.0.outcome(mode))
    }

    pub fn observe(&self, agent_idx: usize) -> PyObservation {
        let mut planes = vec![0; PLANES_LEN];
        let mut globals = [0.0; GLOBALS_LEN];
        self.0.encode_for(agent_idx, &mut planes, &mut globals);
        let plane_len = PLANES_LEN / 3;
        let mut moves = Vec::new();
        self.0.legal_moves_for(agent_idx, &mut moves);
        // the globals are 0/1 up to the seat
        let flags: Vec<u8> = globals[..GLOBALS_LEN - 2]
            .iter()
            .map(|x| u8::from(*x > 0.5))
            .collect();
        let (pieces, done) = flags.split_at(4 * PieceType::ALL.len());
        PyObservation {
            observation: planes[..plane_len].to_vec(),
            corners: planes[plane_len..2 * plane_len].to_vec(),
            blocked: planes[2 * plane_len..].to_vec(),
            pieces: pieces.to_vec(),
            done: done.to_vec(),
            action_mask: self
                .0
                .action_set
                .grid_mask(moves.iter().map(|a| usize::from(*a))),
        }
    }

    pub fn step(&mut self, action_idx: Option<usize>) -> Result<(), InvalidAction> {
        self.0.step(action_idx)
    }

    pub fn render(&self) {
        self.0.render();
    }
}

/// An observation as bytes of 0/1, cheap to wrap with `numpy.frombuffer`. Agents are in
/// agent-relative order, see `Observation`.
#[pyclass]
struct PyObservation {
    observation: Vec<u8>,
    corners: Vec<u8>,
    blocked: Vec<u8>,
    pieces: Vec<u8>,
    done: Vec<u8>,
    action_mask: Vec<u8>,
}

#[pymethods]
impl PyObservation {
    /// The occupied cells in (agent, row, col) order.
    #[getter(observation)]
    pub fn observation<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.observation)
    }

    /// The cells each agent's next piece can attach to, in (agent, row, col) order.
    #[getter(corners)]
    pub fn corners<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.corners)
    }

    /// The cells each agent can never fill, in (agent, row, col) order.
    #[getter(blocked)]
    pub fn blocked<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.blocked)
    }

    /// The pieces each agent still has, in (agent, piece) order.
    #[getter(pieces)]
    pub fn pieces<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.pieces)
    }

    /// Whether each agent is out of moves.
    #[getter(done)]
    pub fn done<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.done)
    }

    /// The action mask over the flattened action grid.
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
