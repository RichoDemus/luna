use std::sync::{Arc, Mutex};
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use bevy_prototype_lyon::prelude::*;
use bigdecimal::{BigDecimal, FromPrimitive, ToPrimitive};
use rurel::{AgentTrainer, mdp};
use rurel::mdp::Agent;
use rurel::strategy::explore::RandomExploration;
use rurel::strategy::learn::QLearning;
use rurel::strategy::terminate::FixedIterations;

use crate::MyAction::{DoNothing, Thrust};
use crate::ShipStatus::{Crashed, Landed};

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct MyState {
    pub height: BigDecimal,
    pub velocity: BigDecimal,
    pub fuel: BigDecimal,
    pub status: ShipStatus,
    // pub thrusting: bool,
}


impl mdp::State for MyState {
    type A = MyAction;

    fn reward(&self) -> f64 {
        // should this be improved?
        return if self.status == Landed {
            self.fuel.to_f64().expect("couldn't convert fuel to f64")
        } else if self.status == Crashed {
            -100.
        } else {
            0.
        }
    }

    fn actions(&self) -> Vec<MyAction> {
        vec![
            DoNothing, Thrust,
        ]
    }
}


struct MyAgent {
    state: MyState,
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
enum MyAction {
    DoNothing,
    Thrust,
}

impl Agent<MyState> for MyAgent {
    fn current_state(&self) -> &MyState {
        &self.state
    }

    fn take_action(&mut self, action: &MyAction) {
        self.state = do_a_whole_fucking_tick(&self.state, action);
    }
}

fn do_a_whole_fucking_tick(state: &MyState, action: &MyAction) -> MyState {
    let ship_height = state.height.to_f32().unwrap();
    let velocity = state.velocity.to_f32().unwrap();
    let status = state.status;
    let fuel = state.fuel.to_f32().unwrap();
    let create_ship2 = move |mut commands: Commands| {
        commands.spawn((
            Ship,
            status,
            FuelTank(fuel),
            Transform::from_xyz(0., ship_height, 10.),
            Velocity(Vec2::new(0.,velocity )),
        ));
    };

    let thrusting = action == &Thrust;
    let input2 = move |time: Res<Time>,mut ships: Query<(&mut Velocity, &ShipStatus, &mut FuelTank)>| {
        if thrusting {
        for (mut velocity, status, mut fuel) in &mut ships {
            if status != &ShipStatus::Falling {
                continue;
            }
            velocity.0.y += 50. * time.delta_seconds();
            fuel.0 -= 10. * time.delta_seconds();
        }}
    };

    let mut app = App::new();

    app.add_plugins((TaskPoolPlugin::default(), TypeRegistrationPlugin::default(), FrameCountPlugin::default(), ScheduleRunnerPlugin::default()));
    app.add_systems(Startup, (create_ship2, create_ground));
    app.add_systems(Update, (input2, gravity.after(input2), movement.after(gravity), touchdown));
    // let mut time = app.world.resource_mut::<Time>();
    // time.update();
    // println!("{}", time.elapsed_seconds());
    app.world.init_resource::<Time>();
    let mut time = app.world.resource_mut::<Time>();
    time.update();
    app.update();

    // for (velocity, transform, status, fuel) in app.world.query::<(&Velocity, &Transform, &ShipStatus, &FuelTank)>().iter(&app.world) {
    //     println!(
    //         "Ship: {:?}: Pos: {:?} velocity: {:?}. {:?} fuel left",
    //         status, transform.translation, velocity, fuel
    //     );
    // }

    let mut time = app.world.resource_mut::<Time>();
    let last_update = time.last_update().unwrap();
    time.update_with_instant(last_update + Duration::from_millis(6));
    // println!("{}", time.elapsed_seconds());

    app.update();

    for (velocity, transform, status, fuel) in app.world.query::<(&Velocity, &Transform, &ShipStatus, &FuelTank)>().iter(&app.world) {
        // println!(
        //     "Ship: {:?}: Pos: {:?} velocity: {:?}. {:?} fuel left",
        //     status, transform.translation, velocity, fuel
        // );
        return MyState {
            height: BigDecimal::from_f32(transform.translation.y).unwrap(),
            velocity: BigDecimal::from_f32(velocity.0.y).unwrap(),
            fuel: BigDecimal::from_f32(fuel.0).unwrap(),
            status:*status,
            // thrusting,
        };
    }

    todo!()
}

fn main() {
    let initial_state = MyState {
        height: BigDecimal::from(600),
        velocity: BigDecimal::from(0),
        fuel: BigDecimal::from(1000),
        status: Default::default(),
        // thrusting: false,
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

    let trainer = Arc::new(Mutex::new(trainer));

    // println!("data: {:?}", trainer.export_learned_values());

    println!("woho");

    let trainer_clone = Arc::clone(&trainer);
    let ai_input = move |mut query: Query<(&mut Velocity, &Transform, &ShipStatus, &mut FuelTank)>, time: Res<Time>| {
        let (mut velocity, transform, status, mut fuel) = query.single_mut();
            if status == &ShipStatus::Falling {
                let state = MyState {
                    height: BigDecimal::from_f32(transform.translation.y).unwrap(),
                    velocity: BigDecimal::from_f32(velocity.0.y).unwrap(),
                    fuel: BigDecimal::from_f32(fuel.0).unwrap(),
                    status: *status,
                    // thrusting: false, //todo remove thrusting
                };
                let action = trainer_clone.clone().lock().unwrap().best_action(&state);
                if let Some(thrusting2) = action {
                    if thrusting2 == Thrust {
                        info!("Thrusting");
                        velocity.0.y += 50. * time.delta_seconds();
                        fuel.0 -= 10. * time.delta_seconds();
                    } else {
                        info!("Decided not to thrust");
                    }
                } else {
                    info!("No idea what to do");
                }
            }

    };


    App::new()
        .add_plugins((DefaultPlugins, ShapePlugin))
        .add_systems(Startup, (create_ship, create_ground))
        .add_systems(Update, (ai_input, gravity, movement.after(gravity), touchdown))
        .run();
}


//////////////////////////////////

#[derive(Component)]
struct Ship;

#[derive(Component, Debug)]
struct FuelTank(f32);

impl Default for FuelTank {
    fn default() -> Self {
        Self(1000.)
    }
}

#[derive(Component, Debug, Default, PartialEq, Eq, Hash, Copy, Clone)]
enum ShipStatus {
    #[default]
    Falling,
    Landed,
    Crashed,
}

#[derive(Component, Default, Debug)]
struct Velocity(Vec2);

fn create_ship(mut commands: Commands) {
    let mut camera = Camera2dBundle::default();
    camera.transform.translation.y = 300.;

    commands.spawn(camera);
    commands.spawn((
        Ship,
        ShipStatus::default(),
        FuelTank::default(),
        ShapeBundle {
            path: GeometryBuilder::build_as(&RegularPolygon {
                sides: 3,
                feature: RegularPolygonFeature::Radius(20.0),
                ..RegularPolygon::default()
            }),
            transform: Transform::from_xyz(0., 600., 10.),
            ..default()
        },
        Fill::color(Color::CYAN),
        Stroke::new(Color::BLACK, 1.),
        Velocity::default(),
    ));
}

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

fn debug(ships: Query<(&Velocity, &Transform, &ShipStatus, &FuelTank), With<Ship>>) {
    for (velocity, transform, status, fuel) in &ships {
        info!(
            "Ship: {:?}: Pos: {:?} velocity: {:?}. {:?} fuel left",
            status, transform, velocity, fuel
        );
    }
}

fn gravity(
    time: Res<Time>,
    mut bodies: Query<(&mut Velocity, &Transform, &ShipStatus), With<Ship>>,
) {
    let delta_time = time.delta_seconds();
    for (mut velocity, transform, status) in &mut bodies {
        if status != &ShipStatus::Falling {
            continue;
        }
        let force = calc_gravitational_force_simple(transform.translation.y);
        velocity.0 += force * delta_time;
    }
}

fn movement(time: Res<Time>, mut bodies: Query<(&Velocity, &mut Transform, &ShipStatus)>) {
    for (velocity, mut transform, status) in &mut bodies {
        if status != &ShipStatus::Falling {
            continue;
        }
        if transform.translation.y > 0. {
            transform.translation += velocity.0.extend(0.) * time.delta_seconds();
        }
    }
}

fn touchdown(mut flying_ships: Query<(&Velocity, &Transform, &mut ShipStatus), With<Ship>>) {
    for (velocity, transform, mut status) in &mut flying_ships {
        if transform.translation.y > 15. {
            continue;
        }
        if velocity.0.y < -20. {
            // too fast
            *status = ShipStatus::Crashed;
        } else {
            *status = ShipStatus::Landed;
        }
    }
}

fn calc_gravitational_force_simple(height: f32) -> Vec2 {
    let ground = Vec2::new(0., 0.);
    let body = Vec2::new(0., ground.y + 5000. + height * 30.);
    let moon_weight = 10000000.;
    let body_weight = 100.;
    calculate_gravitational_force(body, body_weight, ground, moon_weight)
}

fn calculate_gravitational_force(
    position: Vec2,
    mass: f32,
    other_position: Vec2,
    other_mass: f32,
) -> Vec2 {
    let difference: Vec2 = other_position - position;
    let distance = difference.length();
    let gravity_direction: Vec2 = difference.normalize();
    let gravity: f32 = 1. * (mass * other_mass) / (distance * distance);

    gravity_direction * gravity
}

fn input(
    keys: Res<Input<KeyCode>>,
    mut ships: Query<(&mut Velocity, &ShipStatus, &mut FuelTank), With<Ship>>,
    time: Res<Time>,
) {
    if keys.pressed(KeyCode::Space) {
        for (mut velocity, status, mut fuel) in &mut ships {
            if status != &ShipStatus::Falling {
                continue;
            }
            velocity.0.y += 50. * time.delta_seconds();
            fuel.0 -= 10. * time.delta_seconds();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use bevy::app::ScheduleRunnerPlugin;

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

        app.add_plugins((TaskPoolPlugin::default(), TypeRegistrationPlugin::default(), FrameCountPlugin::default(), ScheduleRunnerPlugin::default()));
        app.add_systems(Startup, (create_ship, create_ground));
        app.add_systems(Update, (gravity, movement.after(gravity), touchdown));
        // let mut time = app.world.resource_mut::<Time>();
        // time.update();
        // println!("{}", time.elapsed_seconds());
        app.world.init_resource::<Time>();
        let mut time = app.world.resource_mut::<Time>();
        time.update();
        app.update();

        for (velocity, transform, status, fuel) in app.world.query::<(&Velocity, &Transform, &ShipStatus, &FuelTank)>().iter(&app.world) {
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

        for (velocity, transform, status, fuel) in app.world.query::<(&Velocity, &Transform, &ShipStatus, &FuelTank)>().iter(&app.world) {
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
