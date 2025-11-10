pub(crate) const fn optimal_fuel_usage(
    height: f32,
    gravity: f32,
    thrust_accel: f32,
    safe_velocity: f32,
    dt: f32,
) -> f32 {
    assert!(thrust_accel > gravity);
    let term_a = thrust_accel * thrust_accel - thrust_accel * gravity;
    let b = 2.0 * thrust_accel * safe_velocity;
    let c = safe_velocity * safe_velocity - 2.0 * gravity * height;
    let discriminant = b * b - 4.0 * term_a * c;
    assert!(discriminant >= 0.0);
    let sqrt_d = const_sqrt(discriminant);
    let tau = (-b + sqrt_d) / (2.0 * term_a);
    (tau / dt).ceil() * dt
}

const fn const_sqrt(x: f32) -> f32 {
    assert!(x >= 0.0);
    let mut guess = x / 2.0 + 1.0;
    let mut i = 0;
    while i < 10 {
        guess = 0.5 * (guess + x / guess);
        i += 1;
    }
    guess
}
