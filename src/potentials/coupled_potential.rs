use crate::types::MultiField;

use super::potential::PotentialSurface;

#[derive(Clone)]
pub struct CoupledPotential<T, P, C>
where
    P: PotentialSurface<T>,
    C: PotentialSurface<T>, 
    T: MultiField
{
    potential: P,
    coupling: C,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, P, C> CoupledPotential<T, P, C>
where
    P: PotentialSurface<T>,
    C: PotentialSurface<T>,
    T: MultiField
{
    pub fn new(potential: P, coupling: C) -> Self {
        Self {
            potential,
            coupling,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, P, C> PotentialSurface<T> for CoupledPotential<T, P, C>
where
    P: PotentialSurface<T>,
    C: PotentialSurface<T>,
    T: MultiField
{
    #[inline(always)]
    fn value_inplace(&self, r: &f64, destination: &mut T) {
        *destination = self.potential.value(r) + self.coupling.value(r)
    }
}
