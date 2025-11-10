use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::ops::{Add, Mul, Sub};

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Height(pub f32);

impl Display for Height {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1}", self.0)
    }
}

impl From<f32> for Height {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl Sub<Velocity> for Height {
    type Output = Self;

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
pub struct Velocity(pub f32);

impl Display for Velocity {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1}", self.0)
    }
}

impl From<f32> for Velocity {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<Velocity> for f32 {
    fn from(value: Velocity) -> Self {
        value.0
    }
}

impl Add<f32> for Velocity {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        (self.0 + rhs).into()
    }
}

impl Mul<f32> for Velocity {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        (self.0 * rhs).into()
    }
}

impl Sub<f32> for Velocity {
    type Output = Self;

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
pub struct DiscretizedHeight(pub usize);
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct DiscretizedVelocity(pub usize);
