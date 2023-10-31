use crate::types::FMatrix;

use super::potential::{OnePotential, MultiPotential};

/// Multi coupling potential used to couple multi channel potentials.
#[derive(Clone)]
pub struct MultiCoupling
{
    potentials: Vec<(Box<dyn OnePotential>, usize, usize)>,
    dim: usize,
    symmetric: bool,
}

impl MultiCoupling
{
    /// Creates new multi coupling potential with given vector of potentials with their coupling indices in potential matrix.
    /// If `symmetric` is true, the coupling matrix will be symmetric.
    pub fn new(potentials: Vec<(Box<dyn OnePotential>, usize, usize)>, dim: usize, symmetric: bool) -> Self {
        Self {
            potentials,
            dim,
            symmetric,
        }
    }
}

impl MultiPotential for MultiCoupling
{
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    fn value(&self, r: &f64) -> FMatrix {
        let mut values_matrix = FMatrix::zeros(self.dim, self.dim);
        for (potential, i, j) in self.potentials.iter() {
            let value = potential.value(r);

            values_matrix[(*i, *j)] = value;

            if self.symmetric {
                values_matrix[(*j, *i)] = value;
            }
        }

        values_matrix
    }
}
