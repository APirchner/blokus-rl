//! The Python extension `blokus_rl._blokus`.

use pyo3::exceptions::{PyAssertionError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use crate::game::bitboard::Bitboard;
use crate::game::errors::InvalidAction;
use crate::game::{ACTION_GRID_SHAPE, BOARD_SIZE, Game, PieceType, RewardMode, write_planes};

/// The reward modes of `PyBlokus.outcome`, by name.
const REWARD_MODES: [(&str, RewardMode); 3] = [
    ("score", RewardMode::Score),
    ("win_share", RewardMode::WinShare),
    ("rank", RewardMode::Rank),
];

// pickle finds the class through its module
#[pyclass(module = "blokus_rl._blokus")]
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

    pub fn __copy__(&self) -> Self {
        self.copy()
    }

    pub fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.copy()
    }

    /// Pickles the game as its `to_bytes` state.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        self.to_bytes(py)
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        *self = Self::from_bytes(state)?;
        Ok(())
    }

    /// The game state as bytes, restore it with `from_bytes`.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.0.to_bytes())
    }

    #[staticmethod]
    pub fn from_bytes(data: &[u8]) -> PyResult<Self> {
        Game::from_bytes(data).map(Self).ok_or_else(|| {
            PyValueError::new_err("Not a game state written by to_bytes() of this version")
        })
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
    #[classattr]
    pub fn num_actions() -> usize {
        let (orientations, rows, cols) = ACTION_GRID_SHAPE;
        orientations * rows * cols
    }

    /// Shape (orientation, row, col) of the action grid, in C order.
    #[classattr]
    pub fn action_grid_shape() -> (usize, usize, usize) {
        ACTION_GRID_SHAPE
    }

    /// The reward modes of `outcome`.
    #[classattr]
    pub fn reward_modes(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, REWARD_MODES.map(|(name, _)| name))
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

    /// The final rewards in `mode`, one of `reward_modes`, None until the game is over.
    pub fn outcome(&self, mode: &str) -> PyResult<Option<[f32; 4]>> {
        let (_, mode) = REWARD_MODES
            .into_iter()
            .find(|(name, _)| *name == mode)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown reward mode {mode}")))?;
        Ok(self.0.outcome(mode))
    }

    pub fn observe(&self, agent_idx: usize) -> PyResult<PyObservation> {
        if agent_idx >= self.0.num_agents {
            return Err(PyValueError::new_err(format!("Unknown agent {agent_idx}")));
        }
        let obs = self.0.observe(agent_idx);
        let mut moves = Vec::new();
        self.0.observed_moves(&obs, &mut moves);
        let pieces = obs
            .pieces
            .iter()
            .flat_map(|p| (0..PieceType::ALL.len()).map(move |i| u8::from(p & (1 << i) != 0)));
        Ok(PyObservation {
            observation: planes(&obs.boards),
            corners: planes(&obs.corners),
            blocked: planes(&obs.blocked),
            pieces: pieces.collect(),
            done: obs.done.map(u8::from).to_vec(),
            action_mask: self
                .0
                .action_set
                .grid_mask(moves.iter().map(|a| usize::from(*a))),
        })
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

/// The agents' boards as (agent, row, col) bytes of 0/1.
fn planes(boards: &[Bitboard; 4]) -> Vec<u8> {
    let mut planes = vec![0; 4 * BOARD_SIZE * BOARD_SIZE];
    write_planes(boards, &mut planes);
    planes
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
