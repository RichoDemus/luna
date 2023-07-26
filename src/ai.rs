use bevy::prelude::*;
use rurel::mdp;
use rurel::mdp::Agent;
use serde::{Deserialize, Serialize};

use crate::ai::MyAction::{DoNothing, Thrust};
use crate::create_ground;
use crate::lander::ShipStatus::{Crashed, Falling, Landed};
use crate::lander::{
    altitude_to_transform, calc_gravitational_force_simple, gravity, movement, touchdown, Altitude,
    FuelTank, Lander, ShipStatus, Velocity, THRUSTER_TANK_SIZE,
};
use crate::physics::{
    calc_gravitational_pull, calc_movement_by_gravity, calc_thrust, calc_touchdown,
};

#[derive(PartialEq, Eq, Hash, Clone, Debug, Serialize, Deserialize)]
pub struct MyState {
    pub altitude: i32,
    pub velocity: i32,
    pub fuel: u32,
    pub status: ShipStatus,
}
impl From<(Altitude, Velocity, FuelTank, ShipStatus)> for MyState {
    fn from(
        (altitude, velocity, fuel, status): (Altitude, Velocity, FuelTank, ShipStatus),
    ) -> Self {
        Self {
            altitude: altitude.0 / 1000,
            velocity: velocity.0 / 1000,
            fuel: fuel.0,
            status,
        }
    }
}
impl MyState {
    pub fn to_vals(&self) -> (Altitude, Velocity, FuelTank, ShipStatus) {
        (
            Altitude(self.altitude * 1000),
            Velocity(self.velocity * 1000),
            FuelTank(self.fuel),
            self.status,
        )
    }
}

impl mdp::State for MyState {
    type A = MyAction;

    fn reward(&self) -> f64 {
        if let Crashed(damage) = self.status {
            //return (100 - damage) as f64;
            return -1.;
        }
        return if self.status == Landed {
            ((1 + self.fuel) * 100) as f64
        } else {
            0.
        };
    }

    fn actions(&self) -> Vec<MyAction> {
        return if self.altitude > 1_000 {
            vec![DoNothing]
        } else if self.status == Falling {
            if self.fuel > 0 {
                vec![DoNothing, Thrust]
            } else {
                vec![DoNothing]
            }
        } else if self.status != Falling {
            vec![DoNothing]
            // vec![]
        } else {
            vec![DoNothing]
        };
    }
}

pub struct MyAgent {
    pub(crate) state: MyState,
}

#[derive(PartialEq, Eq, Hash, Clone, Debug, Serialize, Deserialize)]
pub enum MyAction {
    DoNothing,
    Thrust,
}

impl Agent<MyState> for MyAgent {
    fn current_state(&self) -> &MyState {
        &self.state
    }

    fn take_action(&mut self, action: &MyAction) {
        // println!("doing action {:?} to {:?}", action, self.state);
        let new_state = do_a_lightweight_tick(&self.state, action);
        // if new_state.status == Landed {
        //     println!("went from {:?} to {:?}", self.state, new_state);
        // }
        self.state = new_state;

        // println!("\tnow {:?}",self.state);
    }
}

pub fn do_a_lightweight_tick(state: &MyState, action: &MyAction) -> MyState {
    let (mut altitude, mut velocity, mut fuel, mut status) = state.to_vals();

    for _ in 0..THRUSTER_TANK_SIZE {
        velocity.0 -= calc_gravitational_pull();
        if let &Thrust = action {
            let (delta_velocity, delta_fuel) = calc_thrust();
            velocity.0 += delta_velocity;
            fuel.0 = fuel.0.saturating_sub(delta_fuel);
        }

        let falling = calc_movement_by_gravity(altitude.0, velocity.0);
        // println!("\talt: {}, vel: {}, fell: {}", altitude, velocity, falling);
        altitude.0 += falling;

        if let Some(new_state) = calc_touchdown(altitude.0, velocity.0) {
            status = new_state;
            if status == Landed {
                // panic!("Landed: {:?} => {:?}", state, MyState::from((altitude, velocity, fuel, status)));
                break;
            }
        }
    }

    MyState::from((altitude, velocity, fuel, status))
}
