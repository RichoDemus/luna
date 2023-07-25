use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use bevy_prototype_lyon::prelude::*;

use crate::camera::CameraPlugin;
use crate::lander::{
    create_lander, gravity, movement, touchdown, FuelTank, Lander, LanderPlugin, ShipStatus,
    Velocity,
};
use crate::ui::UiPlugin;

mod camera;
mod lander;
mod ui;

fn main() {
    // let initial_state = MyState {
    //     height: BigDecimal::from(600),
    //     velocity: BigDecimal::from(0),
    //     fuel: BigDecimal::from(1000),
    //     status: Default::default(),
    //     // thrusting: false,
    // };
    // let mut trainer = AgentTrainer::new();
    // let mut agent = MyAgent {
    //     state: initial_state.clone(),
    // };
    // trainer.train(
    //     &mut agent,
    //     &QLearning::new(0.2, 0.01, 2.),
    //     &mut FixedIterations::new(100000),
    //     &RandomExploration::new(),
    // );
    //
    // let trainer = Arc::new(Mutex::new(trainer));
    //
    // // println!("data: {:?}", trainer.export_learned_values());
    //
    // println!("woho");
    //
    // let trainer_clone = Arc::clone(&trainer);
    // let ai_input = move |mut query: Query<(&mut Velocity, &Transform, &ShipStatus, &mut FuelTank)>, time: Res<Time>| {
    //     let (mut velocity, transform, status, mut fuel) = query.single_mut();
    //         if status == &ShipStatus::Falling {
    //             let state = MyState {
    //                 height: BigDecimal::from_f32(transform.translation.y).unwrap(),
    //                 velocity: BigDecimal::from_f32(velocity.0.y).unwrap(),
    //                 fuel: BigDecimal::from_f32(fuel.0).unwrap(),
    //                 status: *status,
    //                 // thrusting: false, //todo remove thrusting
    //             };
    //             let action = trainer_clone.clone().lock().unwrap().best_action(&state);
    //             if let Some(thrusting2) = action {
    //                 if thrusting2 == Thrust {
    //                     info!("Thrusting");
    //                     velocity.0.y += 50. * time.delta_seconds();
    //                     fuel.0 -= 10. * time.delta_seconds();
    //                 } else {
    //                     info!("Decided not to thrust");
    //                 }
    //             } else {
    //                 info!("No idea what to do");
    //             }
    //         }
    //
    // };

    // App::new()
    //     .add_plugins((DefaultPlugins, ShapePlugin))
    //     .add_systems(Startup, (create_ship, create_ground))
    //     .add_systems(Update, (ai_input, gravity, movement.after(gravity), touchdown))
    //     .run();

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
        .run();
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
    use std::time::Duration;

    use bevy::app::ScheduleRunnerPlugin;

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

        app.add_plugins((
            TaskPoolPlugin::default(),
            TypeRegistrationPlugin::default(),
            FrameCountPlugin::default(),
            ScheduleRunnerPlugin::default(),
        ));
        app.add_systems(Startup, (create_lander, create_ground));
        app.add_systems(Update, (gravity, movement.after(gravity), touchdown));
        // let mut time = app.world.resource_mut::<Time>();
        // time.update();
        // println!("{}", time.elapsed_seconds());
        app.world.init_resource::<Time>();
        let mut time = app.world.resource_mut::<Time>();
        time.update();
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

        let mut time = app.world.resource_mut::<Time>();
        let last_update = time.last_update().unwrap();
        time.update_with_instant(last_update + Duration::from_millis(6));
        // println!("{}", time.elapsed_seconds());

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
        // let mut time = app.world.resource_mut::<Time>();
        // println!("{}", time.elapsed_seconds());

        // assert_eq!(app.world.query::<&Enemy>().iter(&app.world).len(), 1);
    }
}
