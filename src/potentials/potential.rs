
/// Trait defining potential functionality
pub trait Potential {
    type Space;

    fn value(&self, r: &f64) -> Self::Space;

    fn asymptotic_value(&self) -> Self::Space {
        self.value(&f64::INFINITY)
    }

    fn size(&self) -> usize;
}