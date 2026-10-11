use std::hint::black_box;

use blokus_rl::game::{GLOBALS_LEN, Game, PLANES_LEN};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

/// Plays a random game, calling `visit` with every position and the action played from it.
fn play_random_game(state: &mut u64, moves: &mut Vec<u16>, mut visit: impl FnMut(&Game, u16)) {
    let mut game = Game::new();
    loop {
        game.legal_moves(moves);
        if moves.is_empty() {
            return;
        }
        // xorshift, so the benchmark needs no RNG crate
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let action = moves[usize::try_from(*state % moves.len() as u64).unwrap()];
        visit(&game, action);
        game.apply(action);
    }
}

/// Every position of seeded random games, with the action played from it.
fn sample_positions(games: usize) -> Vec<(Game, u16)> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut positions = Vec::new();
    let mut moves = Vec::new();
    for _ in 0..games {
        play_random_game(&mut state, &mut moves, |game, action| {
            positions.push((*game, action));
        });
    }
    positions
}

fn bench_game(c: &mut Criterion) {
    let positions = sample_positions(100);
    let mut moves = Vec::new();
    let mut by_moves: Vec<(usize, Game)> = positions
        .iter()
        .map(|(game, _)| {
            game.legal_moves(&mut moves);
            (moves.len(), *game)
        })
        .collect();
    by_moves.sort_by_key(|(n, _)| *n);
    let (median_moves, median) = by_moves[by_moves.len() / 2];
    println!(
        "{} positions, the median one has {median_moves} legal moves",
        positions.len()
    );

    let mut i = 0;
    let mut next = || {
        i = (i + 1) % positions.len();
        &positions[i]
    };
    c.bench_function("legal_moves, median position", |b| {
        b.iter(|| black_box(&median).legal_moves(&mut moves));
    });
    c.bench_function("legal_moves, mean over positions", |b| {
        b.iter(|| black_box(&next().0).legal_moves(&mut moves));
    });
    let mut planes = vec![0; PLANES_LEN];
    let mut globals = vec![0.0; GLOBALS_LEN];
    c.bench_function("encode, mean over positions", |b| {
        b.iter(|| black_box(&next().0).encode(&mut planes, &mut globals));
    });
    c.bench_function("apply, mean over positions", |b| {
        b.iter_batched_ref(
            || *next(),
            |(game, action)| game.apply(*action),
            BatchSize::SmallInput,
        );
    });
    c.bench_function("copy", |b| b.iter(|| *black_box(&median)));
    let mut state = 1;
    c.bench_function("random game", |b| {
        b.iter(|| play_random_game(&mut state, &mut moves, |_, _| {}));
    });
}

criterion_group!(benches, bench_game);
criterion_main!(benches);
