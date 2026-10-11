//! A Blokus environment for reinforcement learning: the game in Rust, exposed to Python as a
//! `PettingZoo` environment, and to Rust search code through the `game` module.

pub mod game;
#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod tests {
    use crate::game::bitboard::{Bitboard, SEPARATING_BIT_MASK};
    use crate::game::pentobi::{from_pentobi, to_pentobi};
    use crate::game::{
        ACTION_GRID_SHAPE, ActionSet, BOARD_SIZE, GLOBALS_LEN, Game, PLANES_LEN, PieceType,
        START_CELLS,
    };

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

    /// The agent's legal actions, in its frame.
    fn legal(game: &Game, agent: usize) -> Vec<usize> {
        let mut moves = Vec::new();
        game.legal_moves_for(agent, &mut moves);
        moves.into_iter().map(usize::from).collect()
    }

    /// The agent's legal actions over the flattened (orientation, row, col) action grid.
    fn grid_mask(game: &Game, agent: usize) -> Vec<u8> {
        game.action_set.grid_mask(legal(game, agent).into_iter())
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
            let legal = legal(game, agent);
            assert_eq!(legal.is_empty(), game.agents[agent].done);
            let action =
                (!legal.is_empty()).then(|| game.action_set[legal[rng.below(legal.len())]]);
            if let Some(a) = action {
                placed[agent] |= 1 << a.piece_type.index();
            }
            game.step(action.map(|a| a.grid_idx)).unwrap();
        }
    }

    /// Plays random legal actions through the search API until the game is over. `check` sees
    /// every position on the way. Returns the agents and actions played.
    fn play_out_search(
        game: &mut Game,
        rng: &mut Rng,
        mut check: impl FnMut(&Game),
    ) -> Vec<(usize, u16)> {
        let mut played = Vec::new();
        let mut moves = Vec::new();
        loop {
            check(game);
            game.legal_moves(&mut moves);
            if moves.is_empty() {
                assert!(game.is_over());
                return played;
            }
            let action = moves[rng.below(moves.len())];
            played.push((game.current_player(), action));
            game.apply(action);
        }
    }

    /// The game after the agents' actions, None if one of them is not legal.
    fn replay(played: &[(usize, u16)]) -> Option<Game> {
        let mut game = Game::new();
        for (agent, action) in played {
            let legal =
                *agent == game.current_player() && game.is_legal(*agent, usize::from(*action));
            if !legal {
                return None;
            }
            game.apply(*action);
        }
        Some(game)
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
            let action = legal(&game, agent)
                .into_iter()
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
            let grid = grid_mask(&game, agent);
            let legal = legal(&game, agent);
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
        let grid = grid_mask(&game, 0);
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
            assert_eq!(obs.done[k], game.oracle_moves(other).next().is_none());
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
            let before: Vec<Vec<usize>> = (0..4).map(|a| legal(game, a)).collect();
            let (scores, state) = (game.scores(), game.to_bytes());
            let mut copy = *game;
            play_out(&mut copy, &mut rng, |_, _| {});
            assert!(copy.is_over());
            for (agent, actions) in before.iter().enumerate() {
                assert_eq!(&legal(game, agent), actions);
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

    #[test]
    fn test_legal_moves_match_oracle() {
        let mut rng = Rng::new(5);
        let mut moves = Vec::new();
        for _ in 0..RANDOM_GAMES {
            play_out_search(&mut Game::new(), &mut rng, |game| {
                for agent in 0..4 {
                    let oracle: Vec<u16> = game.oracle_moves(agent).collect();
                    game.legal_moves_for(agent, &mut moves);
                    moves.sort_unstable();
                    assert_eq!(moves, oracle);
                    assert_eq!(moves.is_empty(), game.agents[agent].done);
                    // the moves of the Python observation
                    game.observed_moves(&game.observe(agent), &mut moves);
                    moves.sort_unstable();
                    assert_eq!(moves, oracle);
                }
                let current = game.current_player();
                assert!(game.is_over() || !game.agents[current].done);
            });
        }
    }

    #[test]
    fn test_hash_is_incremental() {
        let mut hashes = std::collections::HashSet::new();
        play_out_search(&mut Game::new(), &mut Rng::new(6), |game| {
            // from_bytes computes the hash from the boards
            let restored = Game::from_bytes(&game.to_bytes()).unwrap();
            assert_eq!(restored.hash(), game.hash());
            assert!(hashes.insert(game.hash()));
        });
    }

    #[test]
    fn test_hash_of_transposition() {
        // an agent plays two pieces in the opposite order, in an otherwise identical game
        let mut transpositions = 0;
        for seed in 0..10 {
            let played = play_out_search(&mut Game::new(), &mut Rng::new(seed), |_| {});
            for (i, (agent, _)) in played.iter().enumerate() {
                let Some(j) = (i + 1..played.len()).find(|j| played[*j].0 == *agent) else {
                    continue;
                };
                let mut swapped = played[..=j].to_vec();
                swapped.swap(i, j);
                if let Some(game) = replay(&swapped) {
                    let original = replay(&played[..=j]).unwrap();
                    // the hashes only differ for the monomino bonus, as the scores do
                    assert_eq!(
                        game.hash() == original.hash(),
                        game.scores() == original.scores()
                    );
                    assert_eq!(
                        game.agents.map(|a| a.board),
                        original.agents.map(|a| a.board)
                    );
                    transpositions += 1;
                }
            }
        }
        assert!(transpositions > 100, "only {transpositions} transpositions");
    }

    #[test]
    fn test_hash_of_finished_game() {
        let mut game = Game::new();
        play_out_search(&mut game, &mut Rng::new(9), |_| {});
        let hash = game.hash();
        // agents that are done pass, and the selected agent does not matter once all are
        for _ in 0..4 {
            game.step(None).unwrap();
            assert_eq!(game.hash(), hash);
        }
    }

    #[test]
    #[should_panic(expected = "The game is over")]
    fn test_apply_after_game_over_panics() {
        let mut game = Game::new();
        play_out_search(&mut game, &mut Rng::new(10), |_| {});
        game.apply(0);
    }

    #[test]
    fn test_hash_of_monomino_bonus() {
        // agent 0 placed all pieces, with the monomino or another piece last
        let mut bytes = Game::new().to_bytes();
        bytes[64..68].fill(0);
        let hashes = [PieceType::I1, PieceType::X].map(|last| {
            bytes[68] = last as u8;
            Game::from_bytes(&bytes).unwrap().hash()
        });
        assert_ne!(hashes[0], hashes[1]);
    }

    #[test]
    fn test_action_grid_shape() {
        let (orientations, rows, cols) = ACTION_GRID_SHAPE;
        assert_eq!(orientations * rows * cols, ActionSet::get().grid_map.len());
    }

    #[test]
    fn test_encode_matches_observe() {
        let mut planes = vec![0; PLANES_LEN];
        let mut globals = vec![0.0; GLOBALS_LEN];
        play_out_search(&mut Game::new(), &mut Rng::new(7), |game| {
            game.encode(&mut planes, &mut globals);
            let agent = game.current_player();
            let obs = game.observe(agent);
            let boards = obs.boards.iter().chain(&obs.corners).chain(&obs.blocked);
            let expected: Vec<u8> = boards
                .flat_map(|b| b.into_vecs().into_iter().flatten())
                .map(u8::from)
                .collect();
            assert_eq!(planes, expected);
            let flag = |set: bool| if set { 1.0 } else { 0.0 };
            let pieces = obs
                .pieces
                .iter()
                .flat_map(|p| (0..PieceType::ALL.len()).map(move |i| flag(p & (1 << i) != 0)));
            let seat = [0.0, 1.0, 2.0, 3.0][agent];
            let expected: Vec<f32> = pieces
                .chain(obs.done.map(flag))
                .chain([seat, f32::from(game.ply())])
                .collect();
            assert_eq!(globals, expected);
        });
    }

    #[test]
    fn test_apply_matches_step() {
        let mut applied = Game::new();
        let played = play_out_search(&mut applied, &mut Rng::new(8), |_| {});
        let mut stepped = Game::new();
        for (agent, action) in played {
            // step() passes for agents that are out of moves
            while stepped.agent_selection != agent {
                stepped.step(None).unwrap();
            }
            stepped.step(Some(ActionSet::grid_idx(action))).unwrap();
        }
        assert!(stepped.is_over());
        assert_eq!(stepped.agents, applied.agents);
        assert_eq!(stepped.hash(), applied.hash());
    }

    #[test]
    fn test_pentobi_round_trip() {
        for agent in 0..4 {
            for action in 0..u16::try_from(ActionSet::get().actions.len()).unwrap() {
                let mv = to_pentobi(agent, action);
                assert_eq!(from_pentobi(agent, &mv), Some(action), "{mv}");
            }
        }
        for mv in [
            "", "a0", "u1", "a21", "a1,a3", "a1,", "a20,a20", "a+20", "a020", "é1",
        ] {
            assert_eq!(from_pentobi(0, mv), None, "{mv}");
        }
        // Pentobi accepts upper case too
        assert_eq!(from_pentobi(0, "A20"), from_pentobi(0, "a20"));
    }

    #[test]
    fn test_pentobi_start_corners() {
        // action 0 is the monomino on the agent's start cell
        let corners: Vec<String> = (0..4).map(|agent| to_pentobi(agent, 0)).collect();
        assert_eq!(corners, ["a20", "t20", "t1", "a1"]);
    }

    /// The agents and moves of a game Pentobi played against itself, recorded by
    /// `tests/pentobi_game.py`, and its final score line.
    fn pentobi_record() -> (Vec<(usize, &'static str)>, &'static str) {
        let record = include_str!("../tests/pentobi_game.txt");
        let (moves, final_score) = record.trim_end().rsplit_once('\n').unwrap();
        let moves = moves.lines().map(|line| {
            let (color, mv) = line.split_once(' ').unwrap();
            (color.parse::<usize>().unwrap() - 1, mv)
        });
        (moves.collect(), final_score)
    }

    #[test]
    fn test_pentobi_game_replays() {
        let (moves, final_score) = pentobi_record();
        let game = Game::from_moves(moves).unwrap();
        assert!(game.is_over());
        let points: Vec<i16> = final_score
            .strip_prefix("final_score ")
            .unwrap()
            .split(' ')
            .map(|p| p.parse().unwrap())
            .collect();
        // Pentobi counts points differently, the finishing order is the same
        let scores = game.scores();
        for a in 0..4 {
            for b in 0..4 {
                assert_eq!(points[a].cmp(&points[b]), scores[a].cmp(&scores[b]));
            }
        }
    }

    #[test]
    fn test_from_moves_rejects_invalid_moves() {
        for moves in [
            // not the agent to move
            vec![(1, "t20")],
            // not on the start corner
            vec![(0, "b19")],
            // a pass with moves left
            vec![(0, "pass")],
            // no such agent
            vec![(4, "pass")],
        ] {
            assert!(Game::from_moves(moves).is_err());
        }
        assert!(Game::from_moves([(0, "a20"), (1, "t20")]).is_ok());
    }

    #[test]
    fn test_from_moves_rejects_passes_out_of_turn() {
        let (moves, _) = pentobi_record();
        let pass = moves.iter().position(|(_, mv)| *mv == "pass").unwrap();
        // the first pass, repeated
        let mut repeated = moves[..=pass].to_vec();
        repeated.push(moves[pass]);
        // the first pass, after the next agent's move
        let mut late = moves[..pass + 2].to_vec();
        late.swap(pass, pass + 1);
        assert!(Game::from_moves(moves[..pass + 2].to_vec()).is_ok());
        for moves in [repeated, late] {
            assert!(Game::from_moves(moves).is_err());
        }
    }
}
