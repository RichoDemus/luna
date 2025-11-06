//! 1D Lunar Lander (tabular Q-learning)
//!
//! - State: height (m) and vertical velocity (m/s).
//!   * We use `v` positive = downward (so gravity increases v).
//! - Actions:
//!   0 -> thrusters OFF
//!   1 -> thrusters ON (upwards acceleration)
//! - Goal: touch down with |velocity_at_touchdown| < SAFE_V (1 m/s).
//! - Reward:
//!   * Big +100 for soft landing, -100 for crash.
//!   * Small per-step penalty to encourage finishing sooner.
//!   * Fuel penalty when thrusters are used.
//!
//! This file implements:
//! - simple physics integration (constant acceleration over dt)
//! - interpolation for touchdown velocity when step overshoots surface
//! - discretization of continuous state into a tabular grid
//! - epsilon-greedy Q-learning loop
//! - final evaluation of learned greedy policy
//!
//! Lots of comments so you can follow what's happening.

use rand::prelude::*;

/// Single-step environment state
#[derive(Clone, Copy, Debug)]
struct State {
    /// height above surface in meters. When <= 0 the lander has touched down.
    height: f32,
    /// vertical velocity in m/s, positive = moving *downward*
    velocity: f32,
}

/// Simple 1D lunar lander environment
struct LanderEnv {
    g: f32,          // gravity (m/s^2)
    thrust_acc: f32, // upward acceleration produced when thrusters are ON (m/s^2)
    dt: f32,         // timestep (s)
    max_steps: usize,
    state: State,
    rng: StdRng,
}

impl LanderEnv {
    /// Create a new environment with a given RNG seed.
    fn new(seed: u64) -> Self {
        Self {
            g: 1.62,         // moon gravity ~1.62 m/s^2
            thrust_acc: 5.0, // thrust acceleration (should exceed gravity to be effective)
            dt: 1. / 60.,    // timestep
            max_steps: 2000, // safety cap per episode
            state: State {
                height: 100.0,
                velocity: 0.0,
            },
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Reset the environment. We randomize the starting height a bit so agent can't overfit.
    fn reset(&mut self) -> State {
        // NOTE: `r#gen` is used because `gen` is a reserved keyword in newer Rust.
        let h0 = 80.0 + self.rng.random::<f32>() * 40.0; // uniform in [80, 120)
        self.state = State {
            height: h0,
            velocity: 0.0,
        };
        self.state
    }

    /// Step the environment by taking `action` (0 or 1).
    ///
    /// Returns: (next_state, terminal_reward_if_any, done, fuel_used_this_step)
    ///
    /// Notes:
    /// - We compute net accel = gravity - (thruster_on * thrust_acc).
    ///   Since v is positive downward, gravity increases v; thrust reduces v.
    /// - For touchdown when new_h <= 0 we estimate the exact touchdown time (t_hit)
    ///   inside the dt to compute a more accurate touchdown velocity.
    fn step(&mut self, action: usize) -> (State, f32, bool, f32) {
        let State { height: prev_height, velocity: prev_velocity} = self.state;
        let thrusters_on = action == 1;

        let accel = if thrusters_on { self.g - self.thrust_acc } else { self.g };

        let new_velocity = prev_velocity + accel * self.dt;


        let avg_velocity_this_step = (prev_velocity + new_velocity) / 2.0;
        let new_height = prev_height - avg_velocity_this_step * self.dt;

        let fuel_used = if thrusters_on { self.dt } else { 0. };

        if new_height <= 0.0 {
            assert!(new_velocity >= 0., "Managed to land while moving upwards?, height: {prev_height} new_height:  {new_height} vel: {prev_velocity}, new_vl: {new_velocity}, thrusters on: {thrusters_on}, accel: {accel}");

            self.state = State {
                height: 0.0,
                velocity: new_velocity,
            };
            // success if absolute touchdown speed is less than threshold
            let safe_v = 1.0_f32;
            let success = new_velocity.abs() < safe_v;
            let reward = if success { 1000.0 } else { -1000.0 };
            return (self.state, reward, true, fuel_used);
        }

        // Not landed yet: update the state normally
        self.state = State {
            height: new_height,
            velocity: new_velocity,
        };

        (self.state, 0.0, false, fuel_used)
    }
}

/// Convert continuous (h, v) into discrete indices for tabular Q
///
/// The discretizer clamps the values to [min, max] and maps them into `bins` equally sized buckets.
/// You can tune the number of bins for a tradeoff: more bins = more precise but slower learning.
struct Discretizer {
    h_min: f32,
    h_max: f32,
    h_bins: usize,
    v_min: f32,
    v_max: f32,
    v_bins: usize,
}

impl Discretizer {
    fn new(h_min: f32, h_max: f32, h_bins: usize, v_min: f32, v_max: f32, v_bins: usize) -> Self {
        Self {
            h_min,
            h_max,
            h_bins,
            v_min,
            v_max,
            v_bins,
        }
    }

    /// Map a continuous state to discrete indices (h_index, v_index)
    fn discretize(&self, s: State) -> (usize, usize) {
        // clamp height and velocity to the defined ranges
        let h = s.height.clamp(self.h_min, self.h_max);
        let v = s.velocity.clamp(self.v_min, self.v_max);

        // map to [0, bins-1]
        let h_frac = (h - self.h_min) / (self.h_max - self.h_min + 1e-8);
        let h_idx = (h_frac * (self.h_bins as f32)) as usize;

        let v_frac = (v - self.v_min) / (self.v_max - self.v_min + 1e-8);
        let v_idx = (v_frac * (self.v_bins as f32)) as usize;

        // ensure indices are within bounds
        (h_idx.min(self.h_bins - 1), v_idx.min(self.v_bins - 1))
    }
}

fn main() {
    // ---------- Hyperparameters & setup ----------
    let seed = 12345_u64;
    let mut env = LanderEnv::new(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xDEADBEEF);

    // Discretization grid for (height, velocity)
    // Adjust bins for desired resolution. More bins -> larger table -> slower learning.
    let disc = Discretizer::new(
        0.0,   // h_min
        150.0, // h_max (we will start in ~[80,120], but allow margin)
        120,   // h_bins (height resolution)
        -20.0, // v_min (allow some upward velocities negative)
        50.0,  // v_max (fast downward speeds)
        120,   // v_bins (velocity resolution)
    );

    let number_of_actions = 2usize; // off or on

    // Q-table shaped as flattened 2D: (h_bins * v_bins) rows, each row has n_actions entries
    let q_rows = disc.h_bins * disc.v_bins;
    let mut q_table = vec![0.0_f32; q_rows * number_of_actions];

    // Q-learning hyperparameters
    let episodes = 10_000usize;
    let max_steps_per_episode = 2_000usize;
    let alpha = 0.1_f32; // learning rate
    let gamma = 0.99_f32; // discount factor

    // epsilon-greedy exploration
    let mut eps = 1.0_f32; // start fully random
    let eps_min = 0.05_f32; // minimal exploration
    let eps_decay = 0.9995_f32; // per-episode multiplicative decay

    // Reward shaping: step cost encourages finishing quickly; fuel cost penalizes thruster use
    let fuel_cost_per_s = -1.0_f32; // multiplied by dt when thrusters are on

    // reporting
    let report_every = 100usize;
    let mut reward_accum = 0.0_f32;
    let mut success_count_window = 0usize;

    println!("Starting training: {} episodes", episodes);

    for ep in 1..=episodes {
        // reset environment and state
        let s0 = env.reset();
        let mut s = s0;
        let (mut h_idx, mut v_idx) = disc.discretize(s);
        let mut total_reward = 0.0_f32;
        let mut total_fuel = 0.0_f32;

        for _step in 0..max_steps_per_episode {
            // choose action via epsilon-greedy
            let row = h_idx * disc.v_bins + v_idx;
            let action = if rng.random::<f32>() < eps {
                // random action
                rng.random_range(0..=1)
            } else {
                // greedy action
                let (greedy_action, _) = get_greedy_action_and_q_value(&q_table, row, number_of_actions);
                greedy_action
            };

            // step the environment
            let (new_state, terminal_reward, done, fuel_used) = env.step(action);

            // immediate reward: step cost + fuel penalty + terminal reward (if any)
            let mut immediate_reward = fuel_cost_per_s * fuel_used;
            immediate_reward += terminal_reward; // +100 or -100 if landing happened

            total_reward += immediate_reward;
            total_fuel += fuel_used;

            // discretize next state
            let (new_discretized_height, new_discretized_velocity) = disc.discretize(new_state);

            // Q-learning update:
            // Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s,a)]
            let row_idx = h_idx * disc.v_bins + v_idx;
            let q_index = row_idx * number_of_actions + action;
            let q_sa = q_table[q_index];

            let row2 = new_discretized_height * disc.v_bins + new_discretized_velocity;
            let (_, q_sprime_max) = get_greedy_action_and_q_value(&q_table, row2, number_of_actions);

            let td_target = immediate_reward + gamma * q_sprime_max;
            q_table[q_index] = q_sa + alpha * (td_target - q_sa);

            // advance to next state
            s = new_state;
            h_idx = new_discretized_height;
            v_idx = new_discretized_velocity;

            if done {
                break;
            }
        } // end episode steps

        // decay epsilon
        eps = (eps * eps_decay).clamp(eps_min, 1.0);

        // logging window
        reward_accum += total_reward;
        if total_reward > 0.0 {
            success_count_window += 1;
        }

        if ep % report_every == 0 {
            let avg_reward = reward_accum / (report_every as f32);
            println!(
                "Episode {:5} | avg reward (last {}) = {:7.2} | successes = {}/{} | eps = {:.3}",
                ep, report_every, avg_reward, success_count_window, report_every, eps
            );
            reward_accum = 0.0;
            success_count_window = 0;
        }
    } // training loop

    // ---------- Evaluation ----------
    println!("Training finished. Evaluating greedy policy...");

    let eval_episodes = 200usize;
    let mut successes = 0usize;
    let mut total_eval_fuel = 0.0_f32;
    let mut avg_touch_v = 0.0_f32;

    for _ in 0..eval_episodes {
        let mut s = env.reset();
        let (mut h_idx, mut v_idx) = disc.discretize(s);
        let mut fuel_used = 0.0_f32;
        let mut touchdown_v: Option<f32> = None;

        for _step in 0..max_steps_per_episode {
            // greedy action
            let row = h_idx * disc.v_bins + v_idx;
            let (action, _) = get_greedy_action_and_q_value(&q_table, row, number_of_actions);

            let (s_next, terminal_reward, done, fuel) = env.step(action);
            fuel_used += fuel;

            if done {
                if terminal_reward > 0.0 {
                    successes += 1;
                }
                touchdown_v = Some(s_next.velocity);
                break;
            }

            let (h2, v2) = disc.discretize(s_next);
            s = s_next;
            h_idx = h2;
            v_idx = v2;
        }

        total_eval_fuel += fuel_used;
        avg_touch_v += touchdown_v.unwrap_or(999.0);
    }

    let eval_n = eval_episodes as f32;
    println!("=== EVALUATION SUMMARY ===");
    println!(
        "Success rate: {}/{} ({:.1}%)",
        successes,
        eval_episodes,
        100.0 * (successes as f32) / eval_n
    );
    println!("Avg fuel per episode: {:.4}", total_eval_fuel / eval_n);
    println!("Avg touchdown speed (m/s): {:.4}", avg_touch_v / eval_n);

    println!("Done. If you want, we can:");
    println!("- increase discretization resolution (more bins) for finer control");
    println!("- switch to a small neural net (DQN) when the state space grows");
    println!("- change rewards to encourage more fuel efficiency or smoother landings");

    println!("\nLearned policy (ASCII view, top = high height, left = slow velocity):");
    print_q_table_ascii(&q_table, disc.h_bins, disc.v_bins, number_of_actions);


    // println!("Lets do a step by single landing to understand how this works");
    // {
    //     let mut state = env.reset();
    //     let (mut height, mut velocity) = disc.discretize(state);
    //     let mut fuel_used = 0.0f32;
    //     let mut steps = 0;
    //     let mut step_window = 0;
    //     let mut thrusts_in_current_window = 0;
    //     let mut velocity_in_window = 0;
    //     loop {
    //         steps += 1;
    //         step_window += 1;
    //         let row = height * disc.v_bins + velocity;
    //         let (action, _) = get_greedy_action_and_q_value(&q_table, row, n_actions);
    //
    //         let (next_state, reward, done, fuel) = env.step(action);
    //         fuel_used += fuel;
    //
    //         if action == 1 {
    //             thrusts_in_current_window += 1;
    //         }
    //         velocity_in_window += velocity;
    //
    //         if step_window == 10 {
    //             println!("{}\t{thrusts_in_current_window}", velocity_in_window / 10);
    //             step_window = 0;
    //             thrusts_in_current_window = 0;
    //             velocity_in_window = 0;
    //         }
    //         if done {
    //             println!("\ndone in {steps} steps, reward {reward}, {fuel_used} fuel used");
    //             break;
    //         }
    //
    //         let (next_height, next_velocity) = disc.discretize(next_state);
    //         state = next_state;
    //         height = next_height;
    //         velocity = next_velocity;
    //     }
    // }
}

/// Helper to get the greedy action and its Q-value from the Q-table for a given state row.
fn get_greedy_action_and_q_value(q_table: &[f32], row: usize, n_actions: usize) -> (usize, f32) {
    let q0 = q_table[row * n_actions + 0];
    let q1 = q_table[row * n_actions + 1];
    if q1 > q0 { (1, q1) } else { (0, q0) }
}

/// Prints a simple ASCII visualization of the Q-table.
///
/// For each discrete height row, we print a character for each velocity column:
/// - '^' : thrusters ON is better
/// - '_' : thrusters OFF is better
/// - '·' : neutral / very close
///
/// This is a terminal-friendly "heatmap" of the learned policy.
/// `q_table` is flattened as in your main code: q[state_index * n_actions + action]
fn print_q_table_ascii(
    q_table: &Vec<f32>,
    h_bins: usize,
    v_bins: usize,
    n_actions: usize,
) {
    for h_idx in (0..h_bins).rev() { // print top height first
        let mut row_str = String::new();
        for v_idx in 0..v_bins {
            let row = h_idx * v_bins + v_idx;
            let q_off = q_table[row * n_actions + 0];
            let q_on  = q_table[row * n_actions + 1];
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
