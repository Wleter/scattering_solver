pub mod composite_potential;
pub mod dispersion_potential;
pub mod function_potential;
pub mod gaussian_coupling;
pub mod potential;
pub mod potential_factory;
pub mod morse_long_range;
pub mod pair_potential;

#[cfg(any(feature = "faer", feature = "ndarray", feature = "nalgebra"))]
pub mod multi_coupling;

#[cfg(any(feature = "faer", feature = "ndarray", feature = "nalgebra"))]
pub mod multi_diag_potential;

#[cfg(test)]
mod test {
    use quantum::units::{distance_units::Distance, energy_units::{CmInv, Energy}, Au};

    use crate::potentials::{dispersion_potential::Dispersion, function_potential::FunctionPotential, morse_long_range::MorseLongRangeBuilder, pair_potential::PairPotential, potential::SimplePotential};

    use super::potential_factory::create_lj;

    #[test]
    fn test_potentials() {
        let const_potential = create_lj(Energy(0., Au), Distance(1., Au));
        assert_eq!(const_potential.value(1.), 2.);
        assert_eq!(const_potential.value(5.), 2.);

        let func_potential = FunctionPotential::new(|r, val| *val = r);
        assert_eq!(func_potential.value(1.), 1.);
        assert_eq!(func_potential.value(5.), 5.);

        let pair_potential = PairPotential::new(const_potential.clone(), func_potential.clone());
        assert_eq!(pair_potential.value(1.), 3.);
        assert_eq!(pair_potential.value(5.), 7.);
    }

    #[test]
    fn test_morse() {
        let d0 = Energy(0.002, CmInv);

        let tail = vec![
            Dispersion::new(1394.180, -6),
            Dispersion::new(83461.675549, -8),
            Dispersion::new(7374640.77, -10),
        ];
        
        let morse = MorseLongRangeBuilder::new(d0.to(Au), 7.880185, tail)
            .set_params(5, 3, 15.11784, 0.54)
            .set_betas(vec![-0.516129, -0.0980, 0.1133, -0.0251])
            .build();

        let r = 7.880185;
        let val = Energy(morse.value(r), Au).to(CmInv);

        assert!(d0.value() + val.value() < 1e-3, "Expected: {}, Got: {}", d0.value(), val.value());
    }

    #[test]
    #[cfg(feature = "faer")]
    fn test_faer() {
        use faer::{assert_matrix_eq, mat, Mat};

        use crate::potentials::{multi_coupling::MultiCoupling, potential::Potential};

        use super::multi_diag_potential::Diagonal;

        const N: usize = 4;
        let potentials = (0..N)
            .map(|x| Dispersion::new(x as f64, 0))
            .collect();

        let diagonal = Diagonal::<Mat<f64>, _>::from_vec(potentials);

        let expected = mat![[0., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 2., 0.],
                            [0., 0., 0., 3.]];

        let mut value = Mat::zeros(N, N);
        diagonal.value_inplace(1., &mut value);
        assert_matrix_eq!(value, expected, comp = abs, tol = 1e-12);

        let potentials = (0..(N-1))
            .map(|x| (Dispersion::new(x as f64, 0), x, x + 1))
            .collect();

        let coupling = MultiCoupling::<Mat<f64>, _>::new(N, potentials, false);

        let expected = mat![[0., 0., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 2.],
                            [0., 0., 0., 0.]];

        let mut value = Mat::zeros(4, 4);
        coupling.value_inplace(1., &mut value);
        assert_matrix_eq!(value, expected, comp = abs, tol = 1e-12);

        let potentials = (0..(N-1))
            .map(|x| (Dispersion::new(x as f64, 0), x, x + 1))
            .collect();

        let coupling = MultiCoupling::<Mat<f64>, _>::new(N, potentials, true);

        let expected = mat![[0., 0., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 1., 0., 2.],
                            [0., 0., 2., 0.]];

        let mut value = Mat::zeros(4, 4);
        coupling.value_inplace(1., &mut value);
        assert_matrix_eq!(value, expected, comp = abs, tol = 1e-12);

        let combined = PairPotential::new(diagonal, coupling);

        let expected = mat![[0., 0., 0., 0.],
                            [0., 1., 1., 0.],
                            [0., 1., 2., 2.],
                            [0., 0., 2., 3.]];

        let mut value = Mat::zeros(4, 4);
        combined.value_inplace(1., &mut value);
        assert_matrix_eq!(value, expected, comp = abs, tol = 1e-12);
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn test_ndarray() {
        use ndarray::{arr2, Array2};

        use crate::potentials::{multi_coupling::MultiCoupling, potential::Potential};

        use super::multi_diag_potential::Diagonal;

        const N: usize = 4;
        let potentials = (0..N)
            .map(|x| Dispersion::new(x as f64, 0))
            .collect();

        let diagonal = Diagonal::<Array2<f64>, _>::from_vec(potentials);

        let potentials = (0..(N-1))
            .map(|x| (Dispersion::new(x as f64, 0), x, x + 1))
            .collect();

        let coupling = MultiCoupling::<Array2<f64>, _>::new(N, potentials, false);
        let combined = PairPotential::new(diagonal, coupling);

        let expected = arr2(&[[0., 0., 0., 0.],
                            [0., 1., 1., 0.],
                            [0., 0., 2., 2.],
                            [0., 0., 0., 3.]]);

        let mut value = Array2::zeros((4, 4));
        combined.value_inplace(1., &mut value);
        assert!((value - expected).abs().sum() < 1e-8);
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_nalgebra() {
        use nalgebra::{DMatrix, Matrix4};

        use crate::potentials::{multi_coupling::MultiCoupling, potential::Potential};

        use super::multi_diag_potential::Diagonal;

        //////// SMatrix

        const N: usize = 4;
        let potentials = (0..N)
            .map(|x| Dispersion::new(x as f64, 0))
            .collect();

        let diagonal = Diagonal::<Matrix4<f64>, _>::from_vec(potentials);

        let potentials = (0..(N-1))
            .map(|x| (Dispersion::new(x as f64, 0), x, x + 1))
            .collect();

        let coupling = MultiCoupling::<Matrix4<f64>, _>::new(N, potentials, false);
        let combined = PairPotential::new(diagonal, coupling);

        let expected = Matrix4::new(0., 0., 0., 0.,
                                    0., 1., 1., 0.,
                                    0., 0., 2., 2.,
                                    0., 0., 0., 3.);

        let mut value = Matrix4::zeros();
        combined.value_inplace(1., &mut value);
        assert!((value - expected).abs().sum() < 1e-8);

        //////// DMatrix

        let potentials = (0..N)
            .map(|x| Dispersion::new(x as f64, 0))
            .collect();

        let diagonal = Diagonal::<DMatrix<f64>, _>::from_vec(potentials);

        let potentials = (0..(N-1))
            .map(|x| (Dispersion::new(x as f64, 0), x, x + 1))
            .collect();

        let coupling = MultiCoupling::<DMatrix<f64>, _>::new(N, potentials, false);
        let combined = PairPotential::new(diagonal, coupling);

        let expected = DMatrix::from_row_iterator(4, 4, vec![0., 0., 0., 0.,
                                                    0., 1., 1., 0.,
                                                    0., 0., 2., 2.,
                                                    0., 0., 0., 3.]);

        let mut value = DMatrix::zeros(4, 4);
        combined.value_inplace(1., &mut value);

        assert!((value - expected).abs().sum() < 1e-8);
    }
}