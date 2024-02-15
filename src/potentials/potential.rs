use crate::types::MultiField;

/// One-dimensional potential curve.
pub trait PotentialCurve: Clone + Send + Sync {
    fn value_inplace(&self, r: &f64, destination: &mut f64);

    #[inline(always)]
    fn value(&self, r: &f64) -> f64 {
        let mut destination = 0.0;
        
        self.value_inplace(r, &mut destination);
        destination
    }

    fn asymptotic_value(&self) -> f64 {
        self.value(&f64::INFINITY)
    }
}

/// Multi-dimensional matrix potential surface.
pub trait PotentialSurface<T>: Clone + Send + Sync 
where
    T: MultiField
{
    fn value_inplace(&self, r: &f64, destination: &mut T);

    #[inline(always)]
    fn value(&self, r: &f64) -> T {
        let mut destination = T::zero();
        
        self.value_inplace(r, &mut destination);
        destination
    }

    fn asymptotic_value(&self) -> T {
        self.value(&f64::INFINITY)
    }
}