pub mod types;
pub mod potentials;
pub mod numerovs;

// pub mod propagation;

extern crate nalgebra;
extern crate quantum;

extern crate num_traits;
use std::{time::Instant, ops::{Mul, Add}};

use nalgebra::*;
use num_traits::identities::One;

fn main() 
{
    type Matrix9x9 = SMatrix<f64, 9, 9>;

    generic_impl(&Matrix9x9::repeat(-2.0));
    generic_impl(&5.0);
}

fn generic_impl<T>(value : &T) -> T
    where 
    T: SimdValue + std::fmt::Display + Copy + One + Mul<f64, Output = T> + Add<T, Output = T>
{
    let identity = T::one();
    let result = *value + (identity * 5.0);

    println!("{}", result.to_string());
    result
}