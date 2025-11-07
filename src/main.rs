mod q;

use rand::prelude::*;
use crate::q::QLearning;

pub const HEIGHT_BINS: usize = 120;
pub const VELOCITY_BINS: usize = 120;
pub const Q_ROWS: usize = HEIGHT_BINS * VELOCITY_BINS;
pub const NUMBER_OF_ACTIONS: usize  = 2;
pub const EPISODES:usize = 500_000;

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

    fn reset(&mut self) -> State {
        self.state = State {
            height: self.rng.random_range(80.0..120.0),
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

        let new_height = prev_height - new_velocity * self.dt;

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
    let mut q_learning = QLearning::new(seed);

    // Discretization grid for (height, velocity)
    // Adjust bins for desired resolution. More bins -> larger table -> slower learning.
    let disc = Discretizer::new(
        0.0,   // h_min
        150.0, // h_max (we will start in ~[80,120], but allow margin)
        HEIGHT_BINS,   // h_bins (height resolution)
        -20.0, // v_min (allow some upward velocities negative)
        50.0,  // v_max (fast downward speeds)
        VELOCITY_BINS,   // v_bins (velocity resolution)
    );

    let number_of_actions = 2usize; // off or on

    // Q-table shaped as flattened 2D: (h_bins * v_bins) rows, each row has n_actions entries
    let q_rows = disc.h_bins * disc.v_bins;
    //let mut q_table = vec![0.0_f32; q_rows * number_of_actions];


    let max_steps_per_episode = 2_000usize;

    // Reward shaping: step cost encourages finishing quickly; fuel cost penalizes thruster use
    let fuel_cost_per_s = -1.0_f32; // multiplied by dt when thrusters are on

    // reporting
    let report_every = 100usize;
    let mut reward_accum = 0.0_f32;
    let mut success_count_window = 0usize;

    println!("Starting training: {} episodes", EPISODES);

    for ep in 1..=EPISODES {
        // reset environment and state
        let s0 = env.reset();
        let mut s = s0;
        let (mut discretized_height, mut discretized_velocity) = disc.discretize(s);
        let mut total_reward = 0.0_f32;
        let mut total_fuel = 0.0_f32;

        for _step in 0..max_steps_per_episode {
            let action = q_learning.get_action_epsilon_greedy(discretized_height, discretized_velocity);

            let (new_state, terminal_reward, done, fuel_used) = env.step(action);

            let mut immediate_reward = fuel_cost_per_s * fuel_used;
            immediate_reward += terminal_reward; // +100 or -100 if landing happened

            total_reward += immediate_reward;
            total_fuel += fuel_used;

            let (new_discretized_height, new_discretized_velocity) = disc.discretize(new_state);

            q_learning.q_update(
                discretized_height,
                new_discretized_height,
                discretized_velocity,
                new_discretized_velocity,
                action,
                immediate_reward,
            );

            // advance to next state
            s = new_state;
            discretized_height = new_discretized_height;
            discretized_velocity = new_discretized_velocity;

            if done {
                break;
            }
        } // end episode steps

        // decay epsilon
        q_learning.decay_epsilon();

        // logging window
        reward_accum += total_reward;
        if total_reward > 0.0 {
            success_count_window += 1;
        }

        if ep % report_every == 0 {
            let avg_reward = reward_accum / (report_every as f32);
            println!(
                "Episode {:5} | avg reward (last {}) = {:7.2} | successes = {}/{} | eps = {:.3}",
                ep, report_every, avg_reward, success_count_window, report_every, q_learning.epsilon
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
        let (mut discretized_height, mut discretized_velocity) = disc.discretize(s);
        let mut fuel_used = 0.0_f32;
        let mut touchdown_v: Option<f32> = None;

        for _step in 0..max_steps_per_episode {
            let (action, _) = q_learning.get_greedy_action_and_q_value(discretized_height, discretized_velocity);

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
            discretized_height = h2;
            discretized_velocity = v2;
        }

        total_eval_fuel += fuel_used;
        avg_touch_v += touchdown_v.unwrap_or(999.0);
    }

    let eval_n = eval_episodes as f32;
    q_learning.print();
    println!("=== EVALUATION SUMMARY ===");
    println!(
        "Success rate: {}/{} ({:.1}%)",
        successes,
        eval_episodes,
        100.0 * (successes as f32) / eval_n
    );
    println!("Avg fuel per episode: {:.4}", total_eval_fuel / eval_n);
    println!("Avg touchdown speed (m/s): {:.4}", avg_touch_v / eval_n);

}
