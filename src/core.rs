use crate::persistence;
use crate::q::{
    EPISODES, HEIGHT_BINS, MAX_STEPS_PER_EPISODE, NUMBER_OF_ACTIONS, QLearning, QLearningParameters, VELOCITY_BINS,
};
use crate::types::{Height, Velocity};
use rand::prelude::*;
use std::collections::HashSet;

/// Single-step environment state
#[derive(Copy, Clone)]
pub(crate) struct State {
    pub(crate) height: Height,
    pub(crate) velocity: Velocity,
}

/// Simple 1D lunar lander environment
pub(crate) struct LanderEnv {
    g: f32,          // gravity (m/s^2)
    thrust_acc: f32, // upward acceleration produced when thrusters are ON (m/s^2)
    dt: f32,         // timestep (s)
    pub(crate) state: State,
    rng: StdRng,
}

impl LanderEnv {
    /// Create a new environment with a given RNG seed.
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            g: 1.62,         // moon gravity ~1.62 m/s^2
            thrust_acc: 5.0, // thrust acceleration (should exceed gravity to be effective)
            dt: 1. / 60.,    // timestep
            state: State {
                height: 100.0.into(),
                velocity: 0.0.into(),
            },
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn reset(&mut self) -> State {
        self.state = State {
            height: self.rng.random_range(80.0..120.0).into(),
            velocity: 0.0.into(),
        };
        self.state
    }

    pub(crate) fn step(&mut self, action: usize) -> (State, f32, bool, f32) {
        let prev_height = self.state.height;
        let prev_velocity = self.state.velocity;
        let thrusters_on = action == 1;

        let accel = if thrusters_on { self.g - self.thrust_acc } else { self.g };

        let new_velocity = prev_velocity + accel * self.dt;

        let new_height = prev_height - new_velocity * self.dt;

        let fuel_used = if thrusters_on { self.dt } else { 0. };

        if new_height <= 0.0 {
            assert!(
                new_velocity >= 0.,
                "Managed to land while moving upwards?, height: {prev_height} new_height:  {new_height} vel: {prev_velocity}, new_vl: {new_velocity}, thrusters on: {thrusters_on}, accel: {accel}"
            );

            self.state = State {
                height: 0.0.into(),
                velocity: new_velocity,
            };
            // success if absolute touchdown speed is less than threshold
            let safe_v = 1.0_f32;
            let success = new_velocity.0.abs() < safe_v;
            let reward = if success { 1000.0 } else { -1000.0 };
            return (self.state, reward, true, fuel_used);
        }

        self.state.height = new_height;
        self.state.velocity = new_velocity;
        (self.state, 0.0, false, fuel_used)
    }
}

fn train(q_learning: &mut QLearning, seed: u64) {
    let mut env = LanderEnv::new(seed);
    let fuel_cost_per_s = -1.0_f32;

    for _ in 1..=EPISODES {
        let s0 = env.reset();
        let mut state = s0;
        let mut _total_reward = 0.0_f32;
        let mut _total_fuel = 0.0_f32;

        for _step in 0..MAX_STEPS_PER_EPISODE {
            let discretized_height = state.height.discretize();
            let discretized_velocity = state.velocity.discretize();
            let action = q_learning.get_action_epsilon_greedy(discretized_height, discretized_velocity);

            let (new_state, terminal_reward, done, fuel_used) = env.step(action);

            let immediate_reward = fuel_cost_per_s * fuel_used + terminal_reward;

            _total_reward += immediate_reward;
            _total_fuel += fuel_used;

            let new_discretized_height = new_state.height.discretize();
            let new_discretized_velocity = new_state.velocity.discretize();

            q_learning.q_update(
                discretized_height,
                new_discretized_height,
                discretized_velocity,
                new_discretized_velocity,
                action,
                immediate_reward,
            );

            if done {
                break;
            }
            state = new_state;
        }

        q_learning.decay_epsilon();
    }
}

pub struct EvaluationResults {
    successes: usize,
    eval_episodes: usize,
    eval_n: f32,
    total_eval_fuel: f32,
    avg_touch_v: f32,
}

fn eval(q_learning: &QLearning, seed: u64) -> EvaluationResults {
    let mut env = LanderEnv::new(seed);
    let eval_episodes = 200usize;
    let mut successes = 0usize;
    let mut total_eval_fuel = 0.0_f32;
    let mut avg_touch_v = 0.0_f32;

    for _ in 0..eval_episodes {
        let mut state = env.reset();
        let mut fuel_used = 0.0_f32;
        let mut touchdown_v: Option<f32> = None;

        for _step in 0..MAX_STEPS_PER_EPISODE {
            let (action, _) =
                q_learning.get_greedy_action_and_q_value(state.height.discretize(), state.velocity.discretize());

            let (s_next, terminal_reward, done, fuel) = env.step(action);
            fuel_used += fuel;

            if done {
                if terminal_reward > 0.0 {
                    successes += 1;
                }
                touchdown_v = Some(s_next.velocity.into());
                break;
            }

            state = s_next;
        }

        total_eval_fuel += fuel_used;
        avg_touch_v += touchdown_v.unwrap_or(999.0);
    }

    let eval_n = eval_episodes as f32;
    EvaluationResults {
        successes,
        eval_episodes,
        eval_n,
        total_eval_fuel,
        avg_touch_v,
    }
}

#[allow(unused)]
pub fn train_and_evaluate(seed: u64) -> (QLearning, EvaluationResults) {
    let mut q_learning = QLearning::new(QLearningParameters::default());

    println!("Starting training: {EPISODES} episodes");
    train(&mut q_learning, seed);
    println!("Training finished. Evaluating greedy policy...");
    let results = eval(&q_learning, seed ^ 0xBEEF);
    (q_learning, results)
}

pub(crate) fn run() {
    let q_learning = match persistence::load() {
        None => {
            let mut q_learning = QLearning::new((QLearningParameters::default()));
            train(&mut q_learning, 12345_u64);
            q_learning
        }
        Some(q_table) => {
            let mut q_learning = QLearning::new((QLearningParameters::default()));
            q_learning.table = q_table;
            q_learning
        }
    };
    let results = eval(&q_learning, 12345_u64 ^ 0xBEEF);

    q_learning.print();
    print_value_heatmap(&q_learning.table, HEIGHT_BINS, VELOCITY_BINS, NUMBER_OF_ACTIONS);
    println!("=== EVALUATION SUMMARY ===");
    println!(
        "Success rate: {}/{} ({:.1}%)",
        results.successes,
        results.eval_episodes,
        100.0 * (results.successes as f32) / results.eval_n
    );
    println!("Avg fuel per episode: {:.4}", results.total_eval_fuel / results.eval_n);
    println!("Avg touchdown speed (m/s): {:.4}", results.avg_touch_v / results.eval_n);
    persistence::save(&q_learning.table)
}

fn print_value_heatmap(
    q_table: &[[[f32; NUMBER_OF_ACTIONS]; VELOCITY_BINS]; HEIGHT_BINS],
    h_bins: usize,
    v_bins: usize,
    n_actions: usize,
) {
    let mut values = vec![0.0_f32; h_bins * v_bins];
    let mut v_min = f32::INFINITY;
    let mut v_max = f32::NEG_INFINITY;

    #[allow(clippy::needless_range_loop)]
    for h in 0..h_bins {
        #[allow(clippy::needless_range_loop)]
        for v in 0..v_bins {
            let row = h * v_bins + v;
            let mut best = f32::NEG_INFINITY;
            #[allow(clippy::needless_range_loop)]
            for a in 0..n_actions {
                let q = q_table[h][v][a];
                if q > best {
                    best = q;
                }
            }
            if best.is_infinite() {
                best = 0.0;
            }
            values[row] = best;
            if best < v_min {
                v_min = best;
            }
            if best > v_max {
                v_max = best;
            }
        }
    }

    let range = if (v_max - v_min).abs() < 1e-6 {
        1.0
    } else {
        v_max - v_min
    };

    // Character ramp (low -> high). You can replace with any characters you like.
    let ramp = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

    println!("\nState-value heatmap (V = max_a Q(s,a))");
    println!("Top row = high height. Left = low velocity, Right = high downward velocity");
    println!("Value range: min = {v_min:+.3}, max = {v_max:+.3}\n");

    for h in (0..h_bins).rev() {
        // print highest height first
        let mut line = String::with_capacity(v_bins);
        for v in 0..v_bins {
            let row = h * v_bins + v;
            let val = values[row];
            // normalize to 0..1
            let norm = (val - v_min) / range;
            // index into ramp
            let idx = (norm * ((ramp.len() - 1) as f32)).round() as usize;
            let ch = ramp[idx];
            line.push(ch);
        }
        if line.chars().collect::<HashSet<char>>().len() != 1 {
            println!("{line}");
        }
    }

    // Legend for ramp
    print!("Legend: ");
    for (i, c) in ramp.iter().enumerate() {
        let frac = (i as f32) / ((ramp.len() - 1) as f32);
        let val_repr = v_min + frac * range;
        print!("{c}({val_repr:+.2}) ");
    }
    println!("\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_and_evaluate() {
        if std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true") {
            return;
        }
        let seed = 12345_u64;
        let (_q_learning, results) = train_and_evaluate(seed);

        let epsilon = 1e-6;
        assert_eq!(results.successes, 200);
        let velocities = results.avg_touch_v;
        assert!((velocities - 80.98163).abs() < epsilon, "{velocities} != 80.98163");
        let fuel = results.total_eval_fuel;
        assert!((fuel - 1245.6881).abs() < epsilon, "{fuel} != 1245.6881");
    }
}
