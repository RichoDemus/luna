use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::ops::{Add, Mul, Sub};
use crate::{HEIGHT_BINS, MAX_HEIGHT, MAX_VELOCITY, MIN_HEIGHT, MIN_VELOCITY, VELOCITY_BINS};

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Height (pub f32);

impl Height {
    pub fn discretize(self) -> DiscretizedHeight {
        let h = self.0.clamp(MIN_HEIGHT, MAX_HEIGHT);
        let h_frac = (h - MIN_HEIGHT) / (MAX_HEIGHT - MIN_HEIGHT + 1e-8);
        let h_idx = (h_frac * (HEIGHT_BINS as f32)) as usize;
        DiscretizedHeight(h_idx.min(HEIGHT_BINS - 1))
    }
}

impl Display for Height {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1}", self.0)
    }
}

impl Into<Height> for f32 {
    fn into(self) -> Height {
        Height(self)
    }
}

impl Sub<Velocity> for Height {
    type Output = Height;

    fn sub(self, rhs: Velocity) -> Self::Output {
        (self.0 - rhs.0).into()
    }
}

impl PartialEq<f32> for Height {
    fn eq(&self, other: &f32) -> bool {
        self.0.eq(other)
    }
}

impl PartialOrd<f32> for Height {
    fn partial_cmp(&self, other: &f32) -> Option<Ordering> {
        self.0.partial_cmp(other)
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Velocity (pub f32);

impl Velocity {
    pub fn discretize(self) -> DiscretizedVelocity {
        let h = self.0.clamp(MIN_VELOCITY, MAX_VELOCITY);
        let v_frac = (h - MIN_VELOCITY) / (MAX_VELOCITY - MIN_VELOCITY + 1e-8);
        let h_idx = (v_frac * (VELOCITY_BINS as f32)) as usize;
        DiscretizedVelocity(h_idx.min(VELOCITY_BINS - 1))
    }
}

impl Display for Velocity {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1}", self.0)
    }
}

impl From<f32> for Velocity {
    fn from(value: f32) -> Self {
        Velocity(value)
    }
}

impl From<Velocity> for f32 {
    fn from(value: Velocity) -> Self {
        value.0
    }
}

impl Add<f32> for Velocity {
    type Output = Velocity;

    fn add(self, rhs: f32) -> Self::Output {
        (self.0 + rhs).into()
    }
}

impl Mul<f32> for Velocity {
    type Output = Velocity;

    fn mul(self, rhs: f32) -> Self::Output {
        (self.0 * rhs).into()
    }
}

impl Sub<f32> for Velocity {
    type Output = Velocity;

    fn sub(self, rhs: f32) -> Self::Output {
        (self.0 - rhs).into()
    }
}

impl PartialEq<f32> for Velocity {
    fn eq(&self, other: &f32) -> bool {
        self.0.eq(other)
    }
}

impl PartialOrd<f32> for Velocity {
    fn partial_cmp(&self, other: &f32) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct DiscretizedHeight (pub usize);
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct DiscretizedVelocity (pub usize);
