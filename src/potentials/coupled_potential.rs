use crate::types::FMatrix;

use super::potential::MultiPotential;

#[derive(Clone)]
pub struct CoupledPotential
{
    potential: Box<dyn MultiPotential>,
    coupling: Box<dyn MultiPotential>,
}

impl CoupledPotential
{
    pub fn new(potential: impl MultiPotential, coupling: impl MultiPotential) -> Self {
        Self {
            potential: Box::new(potential),
            coupling: Box::new(coupling),
        }
    }
}

impl MultiPotential for CoupledPotential
{
    fn dim(&self) -> usize {
        self.potential.dim()
    }
    
    #[inline(always)]
    fn value(&self, r: &f64) -> FMatrix {
        self.potential.value(r) + self.coupling.value(r)
    }
}
