mod game;

use game::errors::InvalidAction;
use game::{bitboard::Bitboard, Game, BOARD_SIZE};

fn main() {
    let mut env: Game = Game::new();
    // println!(
    //     "{}",
    //     (env.action_set[21311].bitboard.dilate_ortho() & env.action_set[5053].bitboard).is_empty()
    // );
    env.step(22926);
    env.step(780);
    env.step(5395);
    env.step(20665);
    env.observe(env.agent_selection);
    env.step(15561);
    env.observe(env.agent_selection);
    env.step(20700);
    env.observe(env.agent_selection);
    env.step(10812);
    env.observe(env.agent_selection);
    env.step(15834);
    env.observe(env.agent_selection);
    env.step(28283);
    env.observe(env.agent_selection);
    env.step(26885);
    env.render();
}
