// #[cfg(bench)]


// #[test]
// fn it_works() {
//     let result = 2 + 2;
//     assert_eq!(result, 4);
// }

// #[bench]
// fn bench_matrix_mul(b: &mut test::Bencher) {
//     let a = DMatrix::from_element(9, 9, 2.0);
//     let b = DMatrix::from_element(9, 9, 3.0);
//     b.iter(|| {
//         let _c = a * b;
//     });
// }

// #[bench]
// fn bench_matrix_scalar_mul(b: &mut test::Bencher) {
//     let a = DMatrix::from_element(9, 9, 2.0);
//     let s = 3.0;
//     b.iter(|| {
//         let _c = a * s;
//     });
// }
extern crate nalgebra as na;
use na::*;

type Matrix9x9 = SMatrix<f64, 9, 9>;
fn bench_matrix_mul(a: &Matrix9x9, b: &Matrix9x9) {
    let _c = a * b;
    let _d = a * &_c;
}

fn bench_matrix_scalar_mul(a: &Matrix9x9, s: f64) {
    let _c = s * a;
}

// use lib::{bench_matrix_mul, bench_matrix_scalar_mul};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn matrix_mul_benchmark(c: &mut Criterion) {
    let a = black_box(Matrix9x9::repeat(2.0));
    let b = black_box(Matrix9x9::repeat(3.0));
    c.bench_function("matrix multiplication", |g| g.iter(|| bench_matrix_mul(&a, &b)));
}

fn matrix_scal_mul_benchmark(c: &mut Criterion) {
    let a = black_box(Matrix9x9::repeat(2.0));
    let s = black_box(3.0);
    c.bench_function("matrix scalar multiplication", |b| b.iter(|| bench_matrix_scalar_mul(&a, s)));
}

criterion_group!(benches, matrix_mul_benchmark, matrix_scal_mul_benchmark);
criterion_main!(benches);