use crate::types::FMatrix;

use super::potential::{OnePotential, MultiPotential};

#[derive(Clone)]
pub struct MultiDiagPotential {
    potentials: Vec<Box<dyn OnePotential>>,
    dim: usize,
}

impl MultiDiagPotential {
    pub fn new(potentials: Vec<Box<dyn OnePotential>>) -> Self {
        let dim = potentials.len();
        Self { 
            potentials, 
            dim,
        }
    }
}

impl MultiPotential for MultiDiagPotential
{
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    fn value(&self, r: &f64) -> FMatrix {
        let mut value_array = FMatrix::zeros(self.dim, self.dim);
        for (i, potential) in self.potentials.iter().enumerate() {
            value_array[(i, i)] = potential.value(r);
        }

        value_array
    }
}
