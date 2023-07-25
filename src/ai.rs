use bevy::prelude::*;
use rurel::mdp;
use rurel::mdp::Agent;
use serde::{Deserialize, Serialize};

use crate::ai::MyAction::{DoNothing, Thrust};
use crate::create_ground;
use crate::lander::ShipStatus::{Crashed, Falling, Landed};
use crate::lander::{
    altitude_to_transform, gravity, movement, touchdown, Altitude, FuelTank, Lander, ShipStatus,
    Velocity,
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
            return (100_000 - damage) as f64;
        }
        return if self.status == Landed {
            ((1 + self.fuel) * 100_000) as f64
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
        return if self.altitude > 1000000 {
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
        self.state = do_a_whole_fucking_tick(&self.state, action);
        // println!("\tnow {:?}",self.state);
    }
}

pub fn do_a_whole_fucking_tick(state: &MyState, action: &MyAction) -> MyState {
    let status1 = state.status;
    let i = state.fuel;
    let i1 = state.altitude;
    let i2 = state.velocity;
    let create_ship2 = move |mut commands: Commands| {
        commands.spawn((Lander, status1, FuelTank(i), Altitude(i1), Velocity(i2)));
    };

    let thrusting = action == &Thrust;
    let input2 =
        move |time: Res<Time>, mut ships: Query<(&mut Velocity, &ShipStatus, &mut FuelTank)>| {
            if thrusting {
                for (mut velocity, status, mut fuel) in &mut ships {
                    if status != &ShipStatus::Falling {
                        continue;
                    }
                    velocity.0 += 10 * 17;
                    fuel.0 = fuel.0.saturating_sub(1);
                }
            }
        };

    let mut app = App::new();

    app.add_plugins((MinimalPlugins));
    app.add_systems(Update, (altitude_to_transform));
    app.add_systems(
        FixedUpdate,
        (
            input2,
            gravity,
            movement.after(gravity),
            touchdown.after(movement),
        ),
    );

    app.add_systems(Startup, (create_ship2, create_ground));

    app.update();

    // for (velocity, transform, status, fuel) in app.world.query::<(&Velocity, &Transform, &ShipStatus, &FuelTank)>().iter(&app.world) {
    //     println!(
    //         "Ship: {:?}: Pos: {:?} velocity: {:?}. {:?} fuel left",
    //         status, transform.translation, velocity, fuel
    //     );
    // }

    app.world.run_schedule(FixedUpdate);

    for (velocity, altitude, status, fuel) in app
        .world
        .query::<(&Velocity, &Altitude, &ShipStatus, &FuelTank)>()
        .iter(&app.world)
    {
        // println!(
        //     "Ship: {:?}: Pos: {:?} velocity: {:?}. {:?} fuel left",
        //     status, altitude.0, velocity, fuel
        // );
        return MyState {
            altitude: altitude.0,
            velocity: velocity.0,
            fuel: fuel.0,
            status: *status,
        };
    }

    todo!()
}
