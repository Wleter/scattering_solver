extern crate nalgebra as na;
use na::*;
type Matrix9x9 = SMatrix<f64, 9, 9>;

fn bench_matrix_mul<R: Dim>(a: &Matrix9x9, b: &Matrix9x9) {
    let _c = a * b;
}

fn bench_matrix_scalar_mul<T, R, S>(a: &Matrix9x9, s: f64) {
    let _c = s * a;
}