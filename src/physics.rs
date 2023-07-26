use crate::lander::ShipStatus;
use crate::lander::ShipStatus::{Crashed, Landed};
use crate::FIXED_DELTA_TIME;

pub fn calc_gravitational_pull() -> i32 {
    (1.625 * FIXED_DELTA_TIME * 1000.).round() as i32
}

pub fn calc_thrust() -> (i32, u32) {
    ((10000. * FIXED_DELTA_TIME).round() as i32, 1)
}

pub fn calc_movement_by_gravity(altitude: i32, velocity: i32) -> i32 {
    if altitude > 0 {
        (velocity as f32 * FIXED_DELTA_TIME).round() as i32
    } else {
        0
    }
}

pub fn calc_touchdown(altitude: i32, velocity: i32) -> Option<ShipStatus> {
    if altitude > 5_000 {
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
        println!(
            "movement by grav per tick: {}",
            calc_movement_by_gravity(1, -100)
        );
    }

    #[test]
    fn test_grav_pull() {
        println!("grav pull per tick: {}", calc_gravitational_pull());
    }

    #[test]
    fn test_thrust() {
        println!("thrust per tick: {:?}", calc_thrust());
    }
}
