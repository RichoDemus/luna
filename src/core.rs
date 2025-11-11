use crate::persistence;
use crate::q::{
    HEIGHT_BINS, MAX_HEIGHT, NUMBER_OF_ACTIONS, OPTIMAL_FUEL_USAGE, QLearning, QLearningParameters, VELOCITY_BINS,
};
use crate::types::{Height, Velocity};
use rand::prelude::*;
use std::collections::HashSet;

pub(crate) const GRAVITATIONAL_CONSTANT: f32 = 1.62;
pub(crate) const THRUST_ACCELERATION: f32 = 5.0;
pub(crate) const SAFE_LANDING_VELOCITY: f32 = 1.0;
pub(crate) const TIMESTEP_DT: f32 = 1. / 60.;

#[derive(Copy, Clone)]
pub(crate) struct State {
    pub(crate) height: Height,
    pub(crate) velocity: Velocity,
}

pub(crate) struct LanderEnv {
    g: f32,
    thrust_acc: f32,
    dt: f32,
    pub(crate) state: State,
    rng: StdRng,
}

impl LanderEnv {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            g: GRAVITATIONAL_CONSTANT,
            thrust_acc: THRUST_ACCELERATION,
            dt: TIMESTEP_DT,
            state: State {
                height: MAX_HEIGHT.into(),
                velocity: 0.0.into(),
            },
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn reset(&mut self) -> State {
        self.state = State {
            height: self.rng.random_range(MAX_HEIGHT * 0.9..MAX_HEIGHT).into(),
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
            let success = new_velocity.0.abs() < SAFE_LANDING_VELOCITY;
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

    for _ in 1..=q_learning.parameters.target_episodes {
        let s0 = env.reset();
        let mut state = s0;
        let mut _total_reward = 0.0_f32;
        let mut total_fuel = 0.0_f32;

        while total_fuel < OPTIMAL_FUEL_USAGE * 10. {
            let action = q_learning.get_action_epsilon_greedy(state.height, state.velocity);

            let (new_state, terminal_reward, done, fuel_used) = env.step(action);

            let immediate_reward = fuel_cost_per_s * fuel_used + terminal_reward;

            _total_reward += immediate_reward;
            total_fuel += fuel_used;

            q_learning.q_update(
                state.height,
                new_state.height,
                state.velocity,
                new_state.velocity,
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
    total_fuel_usage: f32,
    total_touchdown_velocity: f32,
}

impl EvaluationResults {
    pub(crate) fn print(&self) {
        println!("=== EVALUATION SUMMARY ===");
        println!(
            "Success rate: {}/{} ({:.1}%)",
            self.successes,
            self.eval_episodes,
            100.0 * (self.successes as f32) / self.eval_episodes as f32,
        );
        println!(
            "Avg fuel per episode: {:.4}",
            self.total_fuel_usage / self.eval_episodes as f32
        );
        println!(
            "Avg touchdown speed (m/s): {:.4}",
            self.total_touchdown_velocity / self.eval_episodes as f32
        );
    }
}

fn eval(q_learning: &QLearning) -> EvaluationResults {
    // dont use same seed for eval as for training
    let mut env = LanderEnv::new(q_learning.parameters.seed ^ 0xDEAD_BEEF);
    let eval_episodes = 1000;
    let mut successes = 0usize;
    let mut total_fuel_usage = 0.0_f32;
    let mut total_touchdown_velocity = 0.0_f32;

    for _ in 1..=eval_episodes {
        let mut state = env.reset();
        let mut fuel_used = 0.0_f32;

        while fuel_used < OPTIMAL_FUEL_USAGE * 10. {
            let (action, _) = q_learning.get_greedy_action_and_q_value(state.height, state.velocity);

            let (s_next, terminal_reward, done, fuel) = env.step(action);
            fuel_used += fuel;

            if done {
                if terminal_reward > 0.0 {
                    successes += 1;
                }
                total_fuel_usage += fuel_used;
                total_touchdown_velocity += <Velocity as Into<f32>>::into(s_next.velocity);
                break;
            }

            state = s_next;
        }
    }

    EvaluationResults {
        successes,
        eval_episodes,
        total_fuel_usage,
        total_touchdown_velocity,
    }
}

#[allow(unused)]
pub fn train_and_evaluate() -> (QLearning, EvaluationResults) {
    let mut q_learning = QLearning::new(QLearningParameters::default());

    println!("Starting training: {} episodes", q_learning.parameters.target_episodes);
    let seed = q_learning.parameters.seed;
    train(&mut q_learning, seed);
    println!("Training finished. Evaluating greedy policy...");
    let results = eval(&q_learning);
    (q_learning, results)
}

pub(crate) fn run() {
    // let q_learning = match persistence::load() {
    //     None => {
    //         let mut q_learning = QLearning::new(QLearningParameters::default());
    //         train(&mut q_learning, 12345_u64);
    //         q_learning
    //     }
    //     Some(q_table) => {
    //         let mut q_learning = QLearning::new(QLearningParameters::default());
    //         q_learning.table = q_table;
    //         q_learning
    //     }
    // };
    let mut q_learning = QLearning::new(QLearningParameters::default());
    let seed = q_learning.parameters.seed;
    train(&mut q_learning, seed);
    let results = eval(&q_learning);

    q_learning.print();
    print_value_heatmap(&q_learning.table, HEIGHT_BINS, VELOCITY_BINS, NUMBER_OF_ACTIONS);
    results.print();
    persistence::save(&q_learning.table);
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
        let (_q_learning, results) = train_and_evaluate();

        let epsilon = 1e-6;
        assert_eq!(results.successes, 1000);
        let velocities = results.total_touchdown_velocity;
        let expected_velocities = 406.67078;
        assert!(
            (velocities - expected_velocities).abs() < epsilon,
            "{velocities} != {expected_velocities}"
        );
        let fuel = results.total_fuel_usage;
        let expected_fuel = 6309.466;
        assert!((fuel - expected_fuel).abs() < epsilon, "{fuel} != {expected_fuel}");

        results.print();
    }
}
