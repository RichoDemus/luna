use crate::core::{GRAVITATIONAL_CONSTANT, SAFE_LANDING_VELOCITY, THRUST_ACCELERATION, TIMESTEP_DT};
use crate::types::{DiscretizedHeight, DiscretizedVelocity, Height, Velocity};
use crate::util;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

pub(crate) type QTable = [[[f32; NUMBER_OF_ACTIONS]; VELOCITY_BINS]; HEIGHT_BINS];

pub(crate) const HEIGHT_BINS: usize = 120;
pub(crate) const VELOCITY_BINS: usize = 120;
pub(crate) const NUMBER_OF_ACTIONS: usize = 2;
pub(crate) const OPTIMAL_FUEL_USAGE: f32 = util::optimal_fuel_usage(
    MAX_HEIGHT,
    GRAVITATIONAL_CONSTANT,
    THRUST_ACCELERATION,
    SAFE_LANDING_VELOCITY,
    TIMESTEP_DT,
);

pub(crate) const MAX_HEIGHT: f32 = 150.0;

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct QLearningParameters {
    pub min_height: f32,
    pub max_height: f32,
    pub height_bins: usize,
    pub min_velocity: f32,
    pub max_velocity: f32,
    pub velocity_bins: usize,
    pub number_of_actions: usize,
    pub target_episodes: usize,
    pub learning_rate_alpha: f32,
    pub discount_factor_gamma: f32,
    pub starting_epsilon: f32,
    pub minimum_epsilon: f32,
    pub epsilon_decay: f32,
    pub seed: u64,
}

impl Default for QLearningParameters {
    fn default() -> Self {
        Self {
            min_height: 0.0,
            max_height: MAX_HEIGHT,
            height_bins: HEIGHT_BINS,
            min_velocity: -20.0,
            max_velocity: 50.0,
            velocity_bins: VELOCITY_BINS,
            number_of_actions: NUMBER_OF_ACTIONS,
            target_episodes: 350_000,
            learning_rate_alpha: 0.1,
            discount_factor_gamma: 0.99,
            starting_epsilon: 1.0,
            minimum_epsilon: 0.05,
            epsilon_decay: 0.9995,
            seed: 12345_u64 ^ 0xDEAD_BEEF,
        }
    }
}

pub struct QLearning {
    pub table: Box<QTable>,
    pub epsilon: f32,
    rng: StdRng,
    pub parameters: QLearningParameters,
}

impl QLearning {
    pub fn new(parameters: QLearningParameters) -> Self {
        Self {
            table: Box::new([[[0.0; NUMBER_OF_ACTIONS]; VELOCITY_BINS]; HEIGHT_BINS]),
            epsilon: 1.0,
            rng: StdRng::seed_from_u64(parameters.seed),
            parameters,
        }
    }

    pub fn get_greedy_action_and_q_value(&self, height: Height, velocity: Velocity) -> (usize, f32) {
        let discretized_height = self.discretize_height(height);
        let discretized_velocity = self.discretize_velocity(velocity);
        let action_zero_reward = self.table[discretized_height.0][discretized_velocity.0][0];
        let action_one_reward = self.table[discretized_height.0][discretized_velocity.0][1];
        if action_one_reward > action_zero_reward {
            (1, action_one_reward)
        } else {
            (0, action_zero_reward)
        }
    }

    pub fn get_action_epsilon_greedy(&mut self, height: Height, velocity: Velocity) -> usize {
        if self.rng.random::<f32>() < self.epsilon {
            self.rng.random_range(0..=1)
        } else {
            let (greedy_action, _) = self.get_greedy_action_and_q_value(height, velocity);
            greedy_action
        }
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.parameters.epsilon_decay).clamp(self.parameters.minimum_epsilon, 1.0);
    }

    pub fn q_update(
        &mut self,
        height: Height,
        new_height: Height,
        velocity: Velocity,
        new_velocity: Velocity,
        action: usize,
        immediate_reward: f32,
    ) {
        let discretized_height = self.discretize_height(height);
        let discretized_velocity = self.discretize_velocity(velocity);
        let q_value = self.table[discretized_height.0][discretized_velocity.0][action];

        let (_, new_q_max_reward) = self.get_greedy_action_and_q_value(new_height, new_velocity);

        let td_target = immediate_reward + self.parameters.discount_factor_gamma * new_q_max_reward;
        self.table[discretized_height.0][discretized_velocity.0][action] =
            q_value + self.parameters.learning_rate_alpha * (td_target - q_value);
    }

    pub fn discretize_height(&self, height: Height) -> DiscretizedHeight {
        let h = height.0.clamp(self.parameters.min_height, self.parameters.max_height);
        let h_frac =
            (h - self.parameters.min_height) / (self.parameters.max_height - self.parameters.min_height + 1e-8);
        let h_idx = (h_frac * (self.parameters.height_bins as f32)) as usize;
        DiscretizedHeight(h_idx.min(self.parameters.height_bins - 1))
    }

    pub fn discretize_velocity(&self, velocity: Velocity) -> DiscretizedVelocity {
        let h = velocity
            .0
            .clamp(self.parameters.min_velocity, self.parameters.max_velocity);
        let v_frac =
            (h - self.parameters.min_velocity) / (self.parameters.max_velocity - self.parameters.min_velocity + 1e-8);
        let h_idx = (v_frac * (self.parameters.velocity_bins as f32)) as usize;
        DiscretizedVelocity(h_idx.min(self.parameters.velocity_bins - 1))
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
