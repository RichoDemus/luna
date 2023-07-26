use crate::lander::ShipStatus;
use crate::lander::ShipStatus::{Crashed, Landed};

pub fn calc_gravitational_pull(altitude: i32) -> i32 {
    let gravity_at_sea_level = 1.625;
    let mean_moon_radius_meters = 1737400.;
    let altitude_in_meters = (altitude / 1000) as f64;
    let acceleration = gravity_at_sea_level
        * (mean_moon_radius_meters / (mean_moon_radius_meters + altitude_in_meters));
    let delta = (acceleration * 17.) as i32;
    delta
}

pub fn calc_thrust() -> (i32, u32) {
    (10 * 17, 1)
}

pub fn calc_movement_by_gravity(altitude: i32, velocity: i32) -> i32 {
    if altitude > 0 {
        velocity / 60
    } else {
        0
    }
}

pub fn calc_touchdown(altitude: i32, velocity: i32) -> Option<ShipStatus> {
    if altitude > 5 {
        return None;
    }
    if velocity.abs() > 10_000 {
        // too fast
        return Some(Crashed(velocity.abs()));
    } else {
        return Some(Landed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grav_force() {
        println!("{}", calc_movement_by_gravity(1, -1000));
    }
}
