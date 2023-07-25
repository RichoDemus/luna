use bevy::prelude::*;
use bevy_prototype_lyon::prelude::*;

pub(crate) struct LanderPlugin;

impl Plugin for LanderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, create_lander);
        app.add_systems(Update, (touchdown, altitude_to_transform));
        app.add_systems(FixedUpdate, (gravity, movement.after(gravity)));
    }
}

#[derive(Component)]
pub struct Lander;

#[derive(Component)]
pub struct Altitude(pub i32); //in millimeters

#[derive(Component, Default, Debug)]
pub struct Velocity(pub i32); //in milimeters per second

#[derive(Component, Debug)]
pub struct FuelTank(pub u32);

impl Default for FuelTank {
    fn default() -> Self {
        Self(1000)
    }
}

#[derive(Component, Debug, Default, PartialEq, Eq, Hash, Copy, Clone)]
pub enum ShipStatus {
    #[default]
    Falling,
    Landed,
    Crashed,
}

pub fn create_lander(mut commands: Commands) {
    commands.spawn((
        Lander,
        ShipStatus::default(),
        FuelTank::default(),
        Altitude(1000000),
        Velocity::default(),
        ShapeBundle {
            path: GeometryBuilder::build_as(&RegularPolygon {
                sides: 3,
                feature: RegularPolygonFeature::Radius(20.0),
                ..RegularPolygon::default()
            }),
            ..default()
        },
        Fill::color(Color::CYAN),
        Stroke::new(Color::BLACK, 1.),
    ));
}

fn debug(ships: Query<(&Velocity, &Transform, &ShipStatus, &FuelTank), With<Lander>>) {
    for (velocity, transform, status, fuel) in &ships {
        info!(
            "Ship: {:?}: Pos: {:?} velocity: {:?}. {:?} fuel left",
            status, transform, velocity, fuel
        );
    }
}

pub fn gravity(mut bodies: Query<(&mut Velocity, &Altitude, &ShipStatus), With<Lander>>) {
    for (mut velocity, altitude, status) in &mut bodies {
        if status != &ShipStatus::Falling {
            continue;
        }
        let gravity_at_sea_level = 1.625;
        let mean_moon_radius_meters = 1737400.;
        let altitude_in_meters = (altitude.0 / 1000) as f64;
        let acceleration = gravity_at_sea_level
            * (mean_moon_radius_meters / (mean_moon_radius_meters + altitude_in_meters));
        let delta = (acceleration * 17.) as i32;
        // println!("Changing velocity from {} by {}. accel: {}", velocity.0, delta, acceleration);
        velocity.0 -= delta;
    }
}

pub fn movement(mut bodies: Query<(&Velocity, &mut Altitude, &ShipStatus)>) {
    // println!("movement");
    for (velocity, mut altitude, status) in &mut bodies {
        if status != &ShipStatus::Falling {
            continue;
        }
        if altitude.0 > 0 {
            let delta = (velocity.0 / 60);
            altitude.0 += delta;
            // println!("Updated altitude to {} by {}", altitude.0, delta);
        }
    }
}

pub fn touchdown(mut flying_ships: Query<(&Velocity, &Altitude, &mut ShipStatus), With<Lander>>) {
    for (velocity, altitude, mut status) in &mut flying_ships {
        if altitude.0 > 5 {
            continue;
        }
        if velocity.0.abs() > 10_000 {
            // too fast
            *status = ShipStatus::Crashed;
        } else {
            *status = ShipStatus::Landed;
        }
    }
}

pub fn altitude_to_transform(mut query: Query<(&Altitude, &mut Transform)>) {
    for (altitude, mut transform) in &mut query {
        transform.translation.y = (altitude.0 / 1000) as f32;
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
