use bevy::prelude::*;
use rurel::mdp;
use rurel::mdp::Agent;
use serde::{Deserialize, Serialize};

use crate::ai::MyAction::{DoNothing, Thrust};
use crate::create_ground;
use crate::lander::ShipStatus::{Crashed, Falling, Landed};
use crate::lander::{
    altitude_to_transform, calc_gravitational_force_simple, gravity, movement, touchdown, Altitude,
    FuelTank, Lander, ShipStatus, Velocity,
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

impl mdp::State for MyState {
    type A = MyAction;

    fn reward(&self) -> f64 {
        if let Crashed(damage) = self.status {
            return (100 - damage) as f64;
        }
        return if self.status == Landed {
            ((1 + self.fuel) * 100) as f64
        } else {
            0.
        };

        // if self.altitude > 1000000 {
        //     return (1000000 - self.altitude) as f64
        // }
        // // should this be improved?
        // return if self.status == Landed {
        //     (self.fuel * 10000000) as f64
        // } else if self.status == Crashed {
        //     -100.
        // } else {
        //     (1000000 - self.altitude) as f64
        // };
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
        self.state = do_a_lightweight_tick(&self.state, action);
        // println!("\tnow {:?}",self.state);
    }
}

pub fn do_a_lightweight_tick(state: &MyState, action: &MyAction) -> MyState {
    let MyState {
        mut altitude,
        mut velocity,
        mut fuel,
        mut status,
    } = state;
    altitude *= 1000;
    velocity *= 1000;

    //maybe do 60 ticks?
    for _ in 0..20 {
        velocity -= calc_gravitational_pull(altitude);
        if let &Thrust = action {
            let (delta_velocity, delta_fuel) = calc_thrust();
            velocity += delta_velocity;
            fuel = fuel.saturating_sub(delta_fuel);
        }

        let falling = calc_movement_by_gravity(altitude, velocity);
        // println!("\talt: {}, vel: {}, fell: {}", altitude, velocity, falling);
        altitude += falling;

        if let Some(new_state) = calc_touchdown(altitude, velocity) {
            status = new_state;
        }
    }

    MyState {
        altitude: altitude / 1000,
        velocity: velocity / 1000,
        fuel,
        status,
    }
}
