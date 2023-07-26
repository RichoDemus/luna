use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{fs, path};

use bevy::prelude::*;
use bevy_prototype_lyon::prelude::*;
use rurel::mdp::State;
use rurel::strategy::explore::RandomExploration;
use rurel::strategy::learn::QLearning;
use rurel::strategy::terminate::{FixedIterations, SinkStates};
use rurel::AgentTrainer;

use crate::ai::MyAction::Thrust;
use crate::ai::{MyAction, MyAgent, MyState};
use crate::camera::CameraPlugin;
use crate::lander::{
    Altitude, FuelTank, Lander, LanderPlugin, ShipStatus, Thruster, Velocity, STARTING_FUEL,
    THRUSTER_TANK_SIZE,
};
use crate::physics::calc_thrust;
use crate::ui::UiPlugin;

mod ai;
mod camera;
mod lander;
mod physics;
mod ui;

// const LEARNED_VALUES: usize = 1_000_000_000;
const LEARNED_VALUES: usize = 10;
// const LEARNED_VALUES: usize = 100000;

fn train_loop(train: usize) -> AgentTrainer<MyState> {
    if !path::Path::new("model.json").exists() {
        File::create("model.json").unwrap();
        let mut data_file = File::create("model.json").expect("creation failed");
        data_file.write("[]".as_bytes()).expect("write failed");
    }
    let mut last_print = Instant::now();

    let mut data_file = File::open("model.json").unwrap();
    let mut file_content = String::new();
    data_file.read_to_string(&mut file_content).unwrap();
    let learned_values_list: Vec<(MyState, Vec<(MyAction, f64)>)> =
        serde_json::from_str(file_content.as_str()).unwrap();
    let mut learned_values = learned_values_list
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().collect::<HashMap<_, _>>()))
        .collect::<HashMap<_, _>>();

    let initial_state = MyState {
        altitude: 1000,
        velocity: 0,
        fuel: STARTING_FUEL,
        status: Default::default(),
    };

    loop {
        let mut trainer = AgentTrainer::new();
        trainer.import_state(learned_values.clone());
        let mut agent = MyAgent {
            state: initial_state.clone(),
        };
        trainer.train(
            &mut agent,
            &QLearning::new(0.2, 0.01, 2.),
            &mut FixedIterations::new(1_000_000),
            &RandomExploration::new(),
        );
        learned_values = trainer.export_learned_values();

        if Instant::now() - last_print > Duration::from_secs(2) {
            let alt_max = learned_values.keys().max_by_key(|key| key.altitude);
            let alt_min = learned_values.keys().min_by_key(|key| key.altitude);
            let highest_reward = learned_values
                .keys()
                .max_by(|left, right| left.reward().partial_cmp(&right.reward()).unwrap())
                .unwrap();
            println!(
                "One set of training done, now got {} values.\n\thig/low: {:?} / {:?}.\n\tBest: {} {:?}",
                learned_values.len(), alt_max, alt_min, highest_reward.reward(), highest_reward,
            );

            // let learned_values_string_keys = learned_values.iter().map(|(k,v)|(serde_json::to_string(k).unwrap(), v))
            let learned_values_list: Vec<(MyState, Vec<(MyAction, f64)>)> = learned_values
                .clone()
                .into_iter()
                .map(|(key, value)| (key, value.into_iter().collect::<Vec<_>>()))
                .collect::<Vec<_>>();

            let json = serde_json::to_string_pretty(&learned_values_list).unwrap();
            let _ = fs::remove_file("model.json");
            let mut data_file = File::create("model.json").expect("creation failed");
            data_file.write(json.as_bytes()).expect("write failed");
            if learned_values_list.len() > train {
                return trainer;
            }
            last_print = Instant::now();
        }
    }
}

const FIXED_DELTA_TIME: f32 = 1. / 120.;

fn main() {
    let mut ai = false;
    let mut train = 0;
    let mut args = std::env::args();
    let _ = args.next();
    if let Some(arg) = args.next() {
        match arg.as_str() {
            "--ai" => ai = true,
            "--train" => {
                train = FromStr::from_str(args.next().unwrap().as_str()).unwrap();
            }
            _ => panic!(),
        }
    }

    let trainer = train_loop(train);

    let trainer = Arc::new(Mutex::new(trainer));

    // println!("data: {:?}", trainer.export_learned_values());

    let trainer_clone = Arc::clone(&trainer);
    let ai_input = move |mut query: Query<(
        &Velocity,
        &Altitude,
        &ShipStatus,
        &mut FuelTank,
        &mut Thruster,
    )>| {
        let (mut velocity, altitude, status, mut fuel, mut thruster) = query.single_mut();
        if status == &ShipStatus::Falling && thruster.0 == 0 {
            let state = MyState::from((*altitude, *velocity, *fuel, *status));
            let action = trainer_clone.clone().lock().unwrap().best_action(&state);

            if let Some(thrusting2) = action {
                if thrusting2 == Thrust {
                    if thruster.0 == 0 {
                        let amount_to_move = fuel.0.min(THRUSTER_TANK_SIZE);
                        fuel.0 -= amount_to_move;
                        thruster.0 += amount_to_move;
                    }
                } else {
                    info!("Decided not to thrust");
                }
            } else {
                // panic!("No action for state {:?}", state);
                //info!("No idea what to do");
            }
        }
    };

    // if 1 == 1 {
    //     return;
    // }
    let brain = trainer.lock().unwrap().export_learned_values();
    // println!("brain {}: {:#?}", brain.len(), brain);

    if ai {
        App::new()
            .add_plugins((
                DefaultPlugins,
                ShapePlugin,
                CameraPlugin,
                LanderPlugin,
                UiPlugin,
            ))
            .add_systems(Startup, create_ground)
            .add_systems(FixedUpdate, (ai_input))
            .insert_resource(FixedTime::new_from_secs(FIXED_DELTA_TIME))
            .run();
    } else {
        App::new()
            .add_plugins((
                DefaultPlugins,
                ShapePlugin,
                CameraPlugin,
                LanderPlugin,
                UiPlugin,
            ))
            .add_systems(Startup, create_ground)
            .add_systems(FixedUpdate, (input))
            .insert_resource(FixedTime::new_from_secs(FIXED_DELTA_TIME))
            .run();
    }
}

pub fn input(
    keys: Res<Input<KeyCode>>,
    mut ships: Query<(&mut Thruster, &ShipStatus, &mut FuelTank), With<Lander>>,
) {
    if keys.pressed(KeyCode::Space) {
        for (mut thruster, status, mut fuel) in &mut ships {
            if status != &ShipStatus::Falling {
                continue;
            }
            if thruster.0 == 0 {
                let amount_to_move = fuel.0.min(THRUSTER_TANK_SIZE);
                fuel.0 -= amount_to_move;
                thruster.0 += amount_to_move;
            }
        }
    }
}

//////////////////////////////////

fn create_ground(mut commands: Commands) {
    commands.spawn((
        ShapeBundle {
            path: GeometryBuilder::build_as(&shapes::Line(
                Vec2::new(-100., 0.),
                Vec2::new(100.0, 0.0),
            )),
            ..default()
        },
        Fill::color(Color::CYAN),
        Stroke::new(Color::BLACK, 10.),
        Velocity::default(),
    ));
}

#[cfg(test)]
mod tests {
    use crate::lander::calc_gravitational_force_simple;
    use crate::lander::FuelTank;
    use crate::lander::ShipStatus;

    use super::*;

    #[test]
    fn test_grav_force() {
        let force = calc_gravitational_force_simple(1.);
        println!("{force:?}");
    }

    #[test]
    fn try_ml() {}

    #[test]
    fn do_a_tick() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, LanderPlugin));
        app.update();

        for (velocity, transform, status, fuel) in app
            .world
            .query::<(&Velocity, &Transform, &ShipStatus, &FuelTank)>()
            .iter(&app.world)
        {
            println!(
                "Ship: {:?}: Pos: {:?} velocity: {:?}. {:?} fuel left",
                status, transform.translation, velocity, fuel
            );
        }

        app.world.run_schedule(FixedUpdate);

        for (velocity, transform, status, fuel) in app
            .world
            .query::<(&Velocity, &Transform, &ShipStatus, &FuelTank)>()
            .iter(&app.world)
        {
            println!(
                "Ship: {:?}: Pos: {:?} velocity: {:?}. {:?} fuel left",
                status, transform.translation, velocity, fuel
            );
        }
    }
}
