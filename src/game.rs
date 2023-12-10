mod actions;
pub mod agents;
mod bitboard;
mod errors;
mod pieces;

use std::fmt;

use self::actions::{Action, ActionSet};
use self::agents::{Agent, Color};
use self::bitboard::Bitboard;
use self::errors::InvalidAction;

pub struct Observation {
    pub boards: [Bitboard; 4],
}

pub struct Game {
    agents: [Agent; 4],
    pub action_set: ActionSet,
    pub num_agents: usize,
    pub agent_selection: usize,
    agent_order: [[Color; 4]; 4],
}

impl Game {
    pub fn new() -> Self {
        let action_set = ActionSet::new(&pieces::generate_pieces());
        let agents = [
            Agent::new(Color::Blue, action_set.initial_actions()),
            Agent::new(Color::Yellow, action_set.initial_actions()),
            Agent::new(Color::Red, action_set.initial_actions()),
            Agent::new(Color::Green, action_set.initial_actions()),
        ];
        let num_agents: usize = 4;
        let agent_selection: usize = 0;
        let agent_order = [
            [Color::Blue, Color::Yellow, Color::Red, Color::Green],
            [Color::Yellow, Color::Red, Color::Green, Color::Blue],
            [Color::Red, Color::Green, Color::Blue, Color::Yellow],
            [Color::Green, Color::Blue, Color::Yellow, Color::Red],
        ];
        Self {
            agents,
            action_set,
            num_agents,
            agent_selection,
            agent_order,
        }
    }

    pub fn reset() -> Self {
        Self::new()
    }

    pub fn observe(&self, agent_idx: usize) -> Observation {
        let boards = self.align_boards(agent_idx);
        Observation { boards }
    }

    pub fn align_boards(&self, agent_idx: usize) -> [Bitboard; 4] {
        let order = self.agent_order[agent_idx];
        [
            self.agents[order[0] as usize].board,
            self.agents[order[1] as usize].board.rotate_clock(),
            self.agents[order[2] as usize]
                .board
                .rotate_anticlock()
                .rotate_anticlock(),
            self.agents[order[3] as usize].board.rotate_anticlock(),
        ]
    }

    fn execute_action(&mut self, action: &Action) {
        let aligned_boards = self.align_boards(self.agent_selection);
        let agent = &mut self.agents[self.agent_selection];
        // execute action
        agent.board = agent.board | action.bitboard;
        agent.turn += 1;
        agent.pieces.insert(action.piece_type, false);

        let agent_mask = &mut agent.action_mask;
        // mask the played piece
        let piece_range = &self.action_set.piece_map[&action.piece_type];
        let piece_mask = vec![false; piece_range.end - piece_range.start];
        let _u: Vec<bool> = agent_mask
            .splice(piece_range.start..piece_range.end, piece_mask)
            .collect();
        // check and update other actions
        for (p, _) in agent.pieces.iter().filter(|p| *p.1) {
            let piece_range = &self.action_set.piece_map[p];
            for r in piece_range.clone() {
                agent_mask[r] = (self.action_set[r].bitboard & agent.board.dilate_ortho())
                    .is_empty()
                    & !(self.action_set[r].bitboard & agent.board.dilate_diag()).is_empty()
                    & (self.action_set[r].bitboard
                        & aligned_boards[1]
                        & aligned_boards[2]
                        & aligned_boards[3])
                        .is_empty();
            }
        }
    }

    pub fn step(&mut self, action_idx: usize) -> Result<(), InvalidAction> {
        let action = self.action_set[action_idx].clone();
        let agent = &self.agents[self.agent_selection];
        if !agent.action_mask[action.index] {
            return Err(InvalidAction);
        }
        self.execute_action(&action);
        // switch control to next agent
        self.agent_selection = (self.agent_selection + 1) % 4;
        Ok(())
    }

    pub fn render() {}
}

impl fmt::Display for Game {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl fmt::Debug for Game {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Game")
            .field("agents", &self.agents)
            .field("num_agents", &self.num_agents)
            .field("agent_selection", &self.agent_selection)
            .finish()
    }
}
