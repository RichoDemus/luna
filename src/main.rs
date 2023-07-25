use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy_prototype_lyon::prelude::*;
use rurel::strategy::explore::RandomExploration;
use rurel::strategy::learn::QLearning;
use rurel::strategy::terminate::FixedIterations;
use rurel::AgentTrainer;

use crate::ai::MyAction::Thrust;
use crate::ai::{MyAgent, MyState};
use crate::camera::CameraPlugin;
use crate::lander::{Altitude, FuelTank, Lander, LanderPlugin, ShipStatus, Velocity};
use crate::ui::UiPlugin;

mod ai;
mod camera;
mod lander;
mod ui;

fn main() {
    let initial_state = MyState {
        altitude: 1000000,
        velocity: 0,
        fuel: 1000,
        status: Default::default(),
    };
    let mut trainer = AgentTrainer::new();
    let mut agent = MyAgent {
        state: initial_state.clone(),
    };
    trainer.train(
        &mut agent,
        &QLearning::new(0.2, 0.01, 2.),
        &mut FixedIterations::new(100000),
        &RandomExploration::new(),
    );
    println!(
        "Done training, got {} values",
        trainer.export_learned_values().len()
    );

    let trainer = Arc::new(Mutex::new(trainer));

    // println!("data: {:?}", trainer.export_learned_values());

    let trainer_clone = Arc::clone(&trainer);
    let ai_input =
        move |mut query: Query<(&mut Velocity, &Altitude, &ShipStatus, &mut FuelTank)>,
              time: Res<Time>| {
            let (mut velocity, altitude, status, mut fuel) = query.single_mut();
            if status == &ShipStatus::Falling {
                let state = MyState {
                    altitude: altitude.0,
                    velocity: velocity.0,
                    fuel: fuel.0,
                    status: *status,
                };
                let action = trainer_clone.clone().lock().unwrap().best_action(&state);
                if let Some(thrusting2) = action {
                    if thrusting2 == Thrust {
                        info!("Thrusting");
                        velocity.0 += 10 * 17;
                        fuel.0 = fuel.0.saturating_sub(1);
                    } else {
                        info!("Decided not to thrust");
                    }
                } else {
                    // info!("No idea what to do");
                }
            }
        };

    let brain = trainer.lock().unwrap().export_learned_values();
    // println!("brain {}: {:#?}", brain.len(), brain);

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
        .run();

    // App::new()
    //     .add_plugins((
    //         DefaultPlugins,
    //         ShapePlugin,
    //         CameraPlugin,
    //         LanderPlugin,
    //         UiPlugin,
    //     ))
    //     .add_systems(Startup, create_ground)
    //     .add_systems(FixedUpdate, (input))
    //     .run();
}

pub fn input(
    keys: Res<Input<KeyCode>>,
    mut ships: Query<(&mut Velocity, &ShipStatus, &mut FuelTank), With<Lander>>,
) {
    if keys.pressed(KeyCode::Space) {
        for (mut velocity, status, mut fuel) in &mut ships {
            if status != &ShipStatus::Falling {
                continue;
            }
            velocity.0 += 10 * 17;
            fuel.0 = fuel.0.saturating_sub(1);
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
