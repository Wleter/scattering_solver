use std::f64::consts::FRAC_PI_2;

pub fn asymptotic_bessel_j(x: f64, l: usize) -> f64 {
    (x - FRAC_PI_2 * (l as f64)).sin()
}

pub fn asymptotic_bessel_n(x: f64, l: usize) -> f64 {
    (x - FRAC_PI_2 * (l as f64)).cos()
}

pub fn bessel_j_ratio(x1: f64, x2: f64) -> f64 {
    (x1 - x2).exp() * (1.0 - (-2.0 * x1).exp()) / (1.0 - (-2.0 * x2).exp())
}

pub fn bessel_n_ratio(x1: f64, x2: f64) -> f64 {
    (x2 - x1).exp()
}
