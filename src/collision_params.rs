use quantum::particles::Particles;

use crate::potentials::potential::Potential;

/// Parameters for collision calculation.
/// Contains particles and potential.
#[derive(Clone)]
pub struct CollisionParams<P: Potential> {
    pub particles: Particles,
    pub potential: P,
}

impl<P: Potential> CollisionParams<P> {
    /// Creates new collision parameters
    pub fn new(particles: Particles, potential: P) -> Self {
        Self {
            particles,
            potential,
        }
    }
}
