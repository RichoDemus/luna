use crate::types::{DiscretizedHeight, DiscretizedVelocity};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub(crate) type QTable = [[[f32; NUMBER_OF_ACTIONS]; VELOCITY_BINS]; HEIGHT_BINS];

pub(crate) const MIN_HEIGHT: f32 = 0.0;
pub(crate) const MAX_HEIGHT: f32 = 150.0;
pub(crate) const MIN_VELOCITY: f32 = -20.0;
pub(crate) const MAX_VELOCITY: f32 = 50.0;
pub(crate) const HEIGHT_BINS: usize = 120;
pub(crate) const VELOCITY_BINS: usize = 120;
pub(crate) const NUMBER_OF_ACTIONS: usize = 2;
pub(crate) const EPISODES: usize = 250_000;
pub(crate) const MAX_STEPS_PER_EPISODE: usize = 2_000usize;

pub(crate) struct QLearningParameters {}

pub struct QLearning {
    pub table: QTable,
    learning_rate_alpha: f32,
    discount_factor_gamma: f32,
    pub epsilon: f32,
    epsilon_min: f32,
    epsilon_decay: f32,
    rng: StdRng,
}

impl QLearning {
    pub fn new(seed: u64) -> Self {
        Self {
            table: [[[0.0; NUMBER_OF_ACTIONS]; VELOCITY_BINS]; HEIGHT_BINS],
            learning_rate_alpha: 0.1,
            discount_factor_gamma: 0.99,
            epsilon: 1.0,
            epsilon_min: 0.05,
            epsilon_decay: 0.9995,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn get_greedy_action_and_q_value(
        &self,
        discretized_height: DiscretizedHeight,
        discretized_velocity: DiscretizedVelocity,
    ) -> (usize, f32) {
        let action_zero_reward = self.table[discretized_height.0][discretized_velocity.0][0];
        let action_one_reward = self.table[discretized_height.0][discretized_velocity.0][1];
        if action_one_reward > action_zero_reward {
            (1, action_one_reward)
        } else {
            (0, action_zero_reward)
        }
    }

    pub fn get_action_epsilon_greedy(
        &mut self,
        discretized_height: DiscretizedHeight,
        discretized_velocity: DiscretizedVelocity,
    ) -> usize {
        if self.rng.random::<f32>() < self.epsilon {
            self.rng.random_range(0..=1)
        } else {
            let (greedy_action, _) = self.get_greedy_action_and_q_value(discretized_height, discretized_velocity);
            greedy_action
        }
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).clamp(self.epsilon_min, 1.0);
    }

    pub fn q_update(
        &mut self,
        discretized_height: DiscretizedHeight,
        new_discretized_height: DiscretizedHeight,
        discretized_velocity: DiscretizedVelocity,
        new_discretized_velocity: DiscretizedVelocity,
        action: usize,
        immediate_reward: f32,
    ) {
        let q_value = self.table[discretized_height.0][discretized_velocity.0][action];

        let (_, new_q_max_reward) =
            self.get_greedy_action_and_q_value(new_discretized_height, new_discretized_velocity);

        let td_target = immediate_reward + self.discount_factor_gamma * new_q_max_reward;
        self.table[discretized_height.0][discretized_velocity.0][action] =
            q_value + self.learning_rate_alpha * (td_target - q_value);
    }

    pub fn print(&self) {
        println!("\nLearned policy (ASCII view, top = high height, left = slow velocity):");
        for h_idx in (0..HEIGHT_BINS).rev() {
            // print top height first
            let mut row_str = String::new();
            for v_idx in 0..VELOCITY_BINS {
                let q_off = self.table[h_idx][v_idx][0];
                let q_on = self.table[h_idx][v_idx][1];
                let symbol = if (q_on - q_off).abs() < 1e-3 {
                    '·' // roughly equal
                } else if q_on > q_off {
                    '^' // thrust
                } else {
                    '_' // no thrust
                };
                row_str.push(symbol);
            }
            println!("{row_str}");
        }
    }
}
