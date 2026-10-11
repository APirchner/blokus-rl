use pyo3::exceptions::{PyAssertionError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

mod game;

use game::bitboard::Bitboard;
use game::errors::InvalidAction;
use game::{BOARD_SIZE, Game, PieceType, RewardMode};

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
        let obs = self.0.observe(agent_idx);
        PyObservation {
            observation: plane_bytes(&obs.boards),
            corners: plane_bytes(&obs.corners),
            blocked: plane_bytes(&obs.blocked),
            pieces: obs
                .pieces
                .iter()
                .flat_map(|p| (0..PieceType::ALL.len()).map(move |i| u8::from(p & (1 << i) != 0)))
                .collect(),
            done: obs.done.map(u8::from).to_vec(),
            action_mask: obs.action_mask,
        }
    }

    pub fn step(&mut self, action_idx: Option<usize>) -> Result<(), InvalidAction> {
        self.0.step(action_idx)
    }

    pub fn render(&self) {
        self.0.render();
    }
}

/// The boards in (agent, row, col) order as bytes of 0/1.
fn plane_bytes(boards: &[Bitboard; 4]) -> Vec<u8> {
    boards
        .iter()
        .flat_map(|b| b.into_vecs().into_iter().flatten())
        .map(u8::from)
        .collect()
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

#[cfg(test)]
mod tests {
    use crate::game::bitboard::{Bitboard, SEPARATING_BIT_MASK};
    use crate::game::{BOARD_SIZE, Game, PieceType, START_CELLS};

    /// Number of seeded random games played by the observation test, fewer in slow debug builds.
    const RANDOM_GAMES: u64 = if cfg!(debug_assertions) { 20 } else { 1000 };

    /// A small seeded xorshift generator, so the tests need no RNG crate.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }

        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            usize::try_from(self.0 % u64::try_from(n).unwrap()).unwrap()
        }
    }

    /// Plays random legal actions until the game is over. `check` sees every position on the way
    /// and the pieces each agent placed before it, bit `i` for `PieceType::ALL[i]`.
    fn play_out(game: &mut Game, rng: &mut Rng, mut check: impl FnMut(&Game, [u32; 4])) {
        let mut placed = [0u32; 4];
        loop {
            check(game, placed);
            if game.is_over() {
                return;
            }
            let agent = game.agent_selection;
            let legal: Vec<usize> = game.legal_actions(agent).collect();
            assert_eq!(legal.is_empty(), game.agents[agent].done);
            let action =
                (!legal.is_empty()).then(|| game.action_set[legal[rng.below(legal.len())]]);
            if let Some(a) = action {
                placed[agent] |= 1 << a.piece_type.index();
            }
            game.step(action.map(|a| a.grid_idx)).unwrap();
        }
    }

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
        game.step(a22).unwrap();
        game.step(a22).unwrap();
        assert_eq!(game.rewards(), [None; 4]);
        assert_eq!(game.scores(), [-86, -86, -87, -87]);
        // scores are handed out per agent, as soon as it is done
        game.agents[0].done = true;
        assert_eq!(game.rewards(), [Some(-86), None, None, None]);
        game.agents[1].done = true;
        game.agents[2].done = true;
        game.agents[3].done = true;
        assert_eq!(game.rewards(), [Some(-86), Some(-86), Some(-87), Some(-87)]);
    }

    #[test]
    fn test_scoring_monomino_bonus() {
        // +15 for placing all pieces, +5 if the last one was the monomino
        let mut game = Game::new();
        for (agent, last_size) in [(0, 1), (1, 2)] {
            let action = game
                .legal_actions(agent)
                .map(|i| game.action_set[i])
                .find(|a| a.piece_type.size() == last_size)
                .unwrap();
            game.agents[agent].pieces = 1 << action.piece_type.index();
            game.step(Some(action.grid_idx)).unwrap();
            assert!(game.agents[agent].done);
        }
        for a in &mut game.agents {
            a.done = true;
        }
        let rewards = game.rewards();
        assert_eq!(rewards[0], Some(20));
        assert_eq!(rewards[1], Some(15));
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
    fn test_grid_mask_matches_legal_actions() {
        let mut game = Game::new();
        for turn in 0..40 {
            let agent = game.agent_selection;
            let grid = game.observe(agent).action_mask;
            let legal: Vec<usize> = game.legal_actions(agent).collect();
            assert_eq!(
                grid.iter().map(|x| usize::from(*x)).sum::<usize>(),
                legal.len()
            );
            for a in &legal {
                assert_eq!(grid[game.action_set[*a].grid_idx], 1);
            }
            // vary the played actions without pulling in an RNG, agents without actions pass
            let valid: Vec<usize> = (0..grid.len()).filter(|i| grid[*i] == 1).collect();
            let action = valid.get(turn * 7 % valid.len().max(1)).copied();
            game.step(action).unwrap();
        }
    }

    #[test]
    fn test_step_rejects_unavailable_grid_idx() {
        let mut game = Game::new();
        let unmapped = game.action_set.grid_map.iter().position(Option::is_none);
        let out_of_range = Some(game.action_set.grid_map.len());
        let grid = game.observe(0).action_mask;
        let masked =
            (0..grid.len()).find(|i| game.action_set.grid_map[*i].is_some() && grid[*i] == 0);
        for action in [unmapped, out_of_range, masked, None] {
            assert!(game.step(action).is_err());
            assert_eq!(game.agent_selection, 0);
        }
        let valid = (0..grid.len()).find(|i| grid[*i] == 1).unwrap();
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

    #[test]
    fn test_random_games_observation() {
        let mut rng = Rng::new(0);
        for _ in 0..RANDOM_GAMES {
            play_out(&mut Game::new(), &mut rng, |game, placed| {
                for agent in 0..4 {
                    check_observation(game, agent, placed);
                }
            });
        }
    }

    /// Checks the agent's observation against a brute-force recomputation.
    fn check_observation(game: &Game, agent: usize, placed: [u32; 4]) {
        let obs = game.observe(agent);
        let cells = obs.boards.map(Bitboard::into_vecs);
        let start = [
            (0, 0),
            (0, BOARD_SIZE - 1),
            (BOARD_SIZE - 1, BOARD_SIZE - 1),
            (BOARD_SIZE - 1, 0),
        ];
        for k in 0..4 {
            // agent-relative order: the agent itself, the next to move, the one after, the previous
            let other = (agent + k) % 4;
            let all_pieces = (1 << PieceType::ALL.len()) - 1;
            assert_eq!(obs.pieces[k], all_pieces & !placed[other]);
            assert_eq!(obs.done[k], game.legal_actions(other).next().is_none());
            if placed[other] != 0 {
                // the first piece covers the start cell
                assert_ne!(obs.boards[k] & START_CELLS[k], Bitboard::default());
            }
            let own = &cells[k];
            let corners = obs.corners[k].into_vecs();
            let blocked = obs.blocked[k].into_vecs();
            for r in 0..BOARD_SIZE {
                for c in 0..BOARD_SIZE {
                    let own_at = |dr: isize, dc: isize| {
                        let row = r.checked_add_signed(dr).filter(|x| *x < BOARD_SIZE);
                        let col = c.checked_add_signed(dc).filter(|x| *x < BOARD_SIZE);
                        row.zip(col).is_some_and(|(row, col)| own[row][col])
                    };
                    let is_blocked = cells.iter().any(|b| b[r][c])
                        || [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            .iter()
                            .any(|(dr, dc)| own_at(*dr, *dc));
                    let attaches = if placed[other] == 0 {
                        (r, c) == start[k]
                    } else {
                        [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                            .iter()
                            .any(|(dr, dc)| own_at(*dr, *dc))
                    };
                    assert_eq!(blocked[r][c], is_blocked);
                    assert_eq!(corners[r][c], !is_blocked && attaches);
                }
            }
        }
    }

    #[test]
    fn test_observe_is_side_effect_free() {
        play_out(&mut Game::new(), &mut Rng::new(1), |game, _| {
            let before = game.to_bytes();
            for agent in 0..4 {
                assert_eq!(game.observe(agent), game.observe(agent));
            }
            assert_eq!(game.to_bytes(), before);
        });
    }

    #[test]
    fn test_copy_plays_on_independently() {
        let mut rng = Rng::new(2);
        let mut plies = 0;
        play_out(&mut Game::new(), &mut Rng::new(3), |game, _| {
            plies += 1;
            if plies % 10 != 0 {
                return;
            }
            let legal: Vec<Vec<usize>> = (0..4).map(|a| game.legal_actions(a).collect()).collect();
            let (scores, state) = (game.scores(), game.to_bytes());
            let mut copy = *game;
            play_out(&mut copy, &mut rng, |_, _| {});
            assert!(copy.is_over());
            for (agent, actions) in legal.iter().enumerate() {
                assert_eq!(&game.legal_actions(agent).collect::<Vec<_>>(), actions);
            }
            assert_eq!(game.scores(), scores);
            assert_eq!(game.to_bytes(), state);
        });
    }

    #[test]
    fn test_state_is_small_and_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<Game>();
        assert!(std::mem::size_of::<Game>() <= 512);
    }

    #[test]
    fn test_bytes_round_trip() {
        play_out(&mut Game::new(), &mut Rng::new(4), |game, _| {
            let restored = Game::from_bytes(&game.to_bytes()).unwrap();
            assert_eq!(restored.agents, game.agents);
            assert_eq!(restored.agent_selection, game.agent_selection);
        });
        let bytes = Game::new().to_bytes();
        assert!(Game::from_bytes(&bytes[1..]).is_none());
        for (idx, value) in [(68, 4), (66, 0x3f), (15, 0x80), (bytes.len() - 1, 4)] {
            let mut invalid = bytes.clone();
            // an unknown last piece, a 22nd piece, a cell off the board, a 5th agent
            invalid[idx] = value;
            assert!(Game::from_bytes(&invalid).is_none());
        }
    }
}
