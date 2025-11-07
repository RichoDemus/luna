use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use crate::{HEIGHT_BINS, NUMBER_OF_ACTIONS, Q_ROWS, VELOCITY_BINS};
pub struct QLearning {
    table: [f32;Q_ROWS * NUMBER_OF_ACTIONS],
    learning_rate_alpha: f32,
    discount_factor_gamma:f32,
    pub epsilon: f32,
    epsilon_min: f32,
    epsilon_decay: f32,
    rng: StdRng,
}

impl QLearning {
    pub fn new(seed: u64) -> Self {
        Self {
            table: [0.0;Q_ROWS * NUMBER_OF_ACTIONS],
            learning_rate_alpha: 0.1,
            discount_factor_gamma: 0.99,
            epsilon: 1.0,
            epsilon_min: 0.05,
            epsilon_decay: 0.9995,
            rng:StdRng::seed_from_u64(seed ^ 0xDEADBEEF)
        }
    }

    pub fn get_greedy_action_and_q_value(
        &self,
        discretized_height:usize,
        discretized_velocity:usize,
    ) -> (usize, f32) {
        let row = discretized_height * VELOCITY_BINS + discretized_velocity;
        let action_zero_reward = self.table[row * NUMBER_OF_ACTIONS + 0];
        let action_one_reward = self.table[row * NUMBER_OF_ACTIONS + 1];
        if action_one_reward > action_zero_reward { (1, action_one_reward) } else { (0, action_zero_reward) }
    }
    
    pub fn get_action_epsilon_greedy(&mut self, discretized_height:usize, discretized_velocity:usize) -> usize {
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
        discretized_height:usize,
        new_discretized_height:usize,
        discretized_velocity:usize,
        new_discretized_velocity:usize,
        action:usize,
        immediate_reward: f32,
    ) {
        let q_state = discretized_height * VELOCITY_BINS + discretized_velocity;
        let q_index = q_state * NUMBER_OF_ACTIONS + action;
        let q_value = self.table[q_index];

        let (_, new_q_max_reward) = self.get_greedy_action_and_q_value(new_discretized_height, new_discretized_velocity);

        let td_target = immediate_reward + self.discount_factor_gamma * new_q_max_reward;
        self.table[q_index] = q_value + self.learning_rate_alpha * (td_target - q_value);
    }

    pub fn print(self) {
        println!("\nLearned policy (ASCII view, top = high height, left = slow velocity):");
        for h_idx in (0..HEIGHT_BINS).rev() { // print top height first
            let mut row_str = String::new();
            for v_idx in 0..VELOCITY_BINS {
                let row = h_idx * VELOCITY_BINS + v_idx;
                let q_off = self.table[row * NUMBER_OF_ACTIONS + 0];
                let q_on  = self.table[row * NUMBER_OF_ACTIONS + 1];
                let symbol = if (q_on - q_off).abs() < 1e-3 {
                    '·'  // roughly equal
                } else if q_on > q_off {
                    '^'  // thrust
                } else {
                    '_'  // no thrust
                };
                row_str.push(symbol);
            }
            println!("{}", row_str);
        }

    }
}