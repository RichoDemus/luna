use bevy::prelude::*;
use bevy_prototype_lyon::prelude::*;

#[derive(Component)]
pub struct Ship;

#[derive(Component, Debug)]
pub struct FuelTank(f32);

impl Default for FuelTank {
    fn default() -> Self {
        Self(1000.)
    }
}

#[derive(Component, Debug, Default, PartialEq, Eq, Hash, Copy, Clone)]
pub enum ShipStatus {
    #[default]
    Falling,
    Landed,
    Crashed,
}

#[derive(Component, Default, Debug)]
pub struct Velocity(Vec2);

pub fn create_lander(mut commands: Commands) {
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

fn debug(ships: Query<(&Velocity, &Transform, &ShipStatus, &FuelTank), With<Ship>>) {
    for (velocity, transform, status, fuel) in &ships {
        info!(
            "Ship: {:?}: Pos: {:?} velocity: {:?}. {:?} fuel left",
            status, transform, velocity, fuel
        );
    }
}

pub fn gravity(
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

pub fn movement(time: Res<Time>, mut bodies: Query<(&Velocity, &mut Transform, &ShipStatus)>) {
    for (velocity, mut transform, status) in &mut bodies {
        if status != &ShipStatus::Falling {
            continue;
        }
        if transform.translation.y > 0. {
            transform.translation += velocity.0.extend(0.) * time.delta_seconds();
        }
    }
}

pub fn touchdown(mut flying_ships: Query<(&Velocity, &Transform, &mut ShipStatus), With<Ship>>) {
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

pub fn calc_gravitational_force_simple(height: f32) -> Vec2 {
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

pub fn input(
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
