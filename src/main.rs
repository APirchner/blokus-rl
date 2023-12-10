mod game;

use crate::game::agents::Color;
use crate::game::Game;

fn main() {
    let mut game = Game::new();
    game.step(1160);
    game.step(0);
    game.step(1160);
    game.step(0);
    game.step(42);
    let boards = game.observe(0).boards;
    println!(
        "{}",
        boards[Color::Blue as usize]
            | boards[Color::Yellow as usize]
            | boards[Color::Red as usize]
            | boards[Color::Green as usize]
    );
}
