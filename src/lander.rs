use crate::physics::{
    calc_gravitational_pull, calc_movement_by_gravity, calc_thrust, calc_touchdown,
};
use bevy::prelude::*;
use bevy_prototype_lyon::prelude::*;
use serde::{Deserialize, Serialize};

pub(crate) struct LanderPlugin;

impl Plugin for LanderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, create_lander);
        app.add_systems(
            FixedUpdate,
            (
                gravity,
                thrust.after(gravity),
                movement.after(thrust),
                touchdown,
                altitude_to_transform,
            ),
        );
    }
}

#[derive(Component)]
pub struct Lander;

#[derive(Component, Debug, Copy, Clone)]
pub struct Altitude(pub i32); //in millimeters

#[derive(Component, Default, Debug, Copy, Clone)]
pub struct Velocity(pub i32); //in milimeters per second

#[derive(Component, Debug, Default, Copy, Clone)]
pub struct Thruster(pub u32);

#[derive(Component, Debug, Copy, Clone)]
pub struct FuelTank(pub u32);

pub const STARTING_FUEL: u32 = 10000;
pub const THRUSTER_TANK_SIZE: u32 = 60;

impl Default for FuelTank {
    fn default() -> Self {
        Self(STARTING_FUEL)
    }
}

#[derive(Component, Debug, Default, PartialEq, Eq, Hash, Copy, Clone, Serialize, Deserialize)]
pub enum ShipStatus {
    #[default]
    Falling,
    Landed,
    Crashed(i32), //how much damage it took
}

pub fn create_lander(mut commands: Commands) {
    commands.spawn((
        Lander,
        ShipStatus::default(),
        FuelTank::default(),
        Altitude(1000000),
        Velocity::default(),
        Thruster::default(),
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
        velocity.0 -= calc_gravitational_pull();
    }
}

pub fn thrust(mut lander: Query<(&mut Velocity, &mut Thruster, &ShipStatus), With<Lander>>) {
    for (mut velocity, mut thruster, ship_status) in &mut lander {
        if ship_status != &ShipStatus::Falling {
            continue;
        }
        if thruster.0 > 0 {
            let (force, fuel_consumption) = calc_thrust();
            velocity.0 += force;
            thruster.0 = thruster.0.saturating_sub(fuel_consumption);
        }
    }
}

pub fn movement(mut bodies: Query<(&Velocity, &mut Altitude, &ShipStatus)>) {
    // println!("movement");
    for (velocity, mut altitude, status) in &mut bodies {
        if status != &ShipStatus::Falling {
            continue;
        }
        altitude.0 += calc_movement_by_gravity(altitude.0, velocity.0)
    }
}

pub fn touchdown(mut flying_ships: Query<(&Velocity, &Altitude, &mut ShipStatus), With<Lander>>) {
    for (velocity, altitude, mut status) in &mut flying_ships {
        if let Some(new_state) = calc_touchdown(altitude.0, velocity.0) {
            *status = new_state;
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
