use quantum::particles::Particles;

use crate::potentials::potential::Potential;

pub struct CollisionParams<P: Potential> {
    pub particles: Particles,
    pub potential: P,
}

impl<P: Potential> CollisionParams<P> {
    pub fn new(particles: Particles, potential: P) -> Self {
        Self {
            particles,
            potential,
        }
    }
}
