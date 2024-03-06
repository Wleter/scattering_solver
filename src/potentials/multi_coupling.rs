use nalgebra::{Const, Dim, Dyn};

use crate::types::{DFMatrix, FMatrix};

use super::potential::Potential;

/// Multi coupling potential used to couple multi channel potentials.
#[derive(Clone)]
pub struct MultiCoupling<N: Dim, P>
where
    P: Potential<Space = f64>,
{
    potentials: Vec<(P, usize, usize)>,
    symmetric: bool,
    size: N,
}

impl<N: Dim, P> MultiCoupling<N, P>
where
    P: Potential<Space = f64>,
{
    /// Creates new multi coupling potential with given vector of potentials with their coupling indices in potential matrix.
    /// If `symmetric` is true, the coupling matrix will be symmetric.
    pub fn new(size: N, potentials: Vec<(P, usize, usize)>, symmetric: bool) -> Self {
        for (_, i, j) in potentials.iter() {
            assert!(*i < size.value());
            assert!(*j < size.value());
        }

        Self {
            potentials,
            symmetric,
            size: size,
        }
    }

    pub fn new_neighboring(size: N, couplings: Vec<P>) -> Self {
        assert!(couplings.len() + 1 == size.value());

        let numbered_potentials = couplings
            .into_iter()
            .enumerate()
            .map(|(i, potential)| (potential, i, i + 1))
            .collect();

        Self::new(size, numbered_potentials, true)
    }
}

impl<const N: usize, P> Potential for MultiCoupling<Const<N>, P>
where
    P: Potential<Space = f64>,
{
    type Space = FMatrix<N>;

    fn value(&self, r: &f64) -> FMatrix<N> {
        let mut result = FMatrix::<N>::zeros();

        for (potential, i, j) in self.potentials.iter() {
            let value = potential.value(r);

            result[(*i, *j)] = value;

            if self.symmetric {
                result[(*j, *i)] = value;
            }
        }

        result
    }

    fn size(&self) -> usize {
        N
    }
}

impl<P> Potential for MultiCoupling<Dyn, P>
where
    P: Potential<Space = f64>,
{
    type Space = DFMatrix;

    fn value(&self, r: &f64) -> DFMatrix {
        let mut result = DFMatrix::zeros(self.size.0, self.size.0);

        for (potential, i, j) in self.potentials.iter() {
            let value = potential.value(r);

            result[(*i, *j)] = value;

            if self.symmetric {
                result[(*j, *i)] = value;
            }
        }

        result
    }

    fn size(&self) -> usize {
        self.size.value()
    }
}
