pub mod collision_params;
pub mod numerovs;
pub mod potentials;
pub mod types;

extern crate nalgebra;
extern crate quantum;

#[cfg(test)]
mod tests {
    use nalgebra::Matrix3;
    use quantum::{
        particle_factory::create_atom, particles::Particles, units::energy_units::EnergyUnit,
    };

    use crate::{
        collision_params::CollisionParams,
        numerovs::{
            propagator::{Numerov, NumerovResult},
            ratio_numerov::RatioNumerov,
        },
        potentials::{
            multi_diag_potential::MultiDiagPotential, potential::Potential,
            potential_factory::create_lj,
        },
    };

    #[test]
    fn test_potential() {
        let mut potential = create_lj(0.0002, 8.0, 0.0);
        assert_eq!(potential.value(&8.0), -0.0002);

        let mut multi_potential =
            MultiDiagPotential::new([potential.clone(), potential.clone(), potential.clone()]);
        let matrix = Matrix3::from_diagonal_element(potential.value(&12.0));
        assert_eq!(multi_potential.value(&12.0), matrix);
    }

    #[test]
    fn test_collision_params() {
        let potential = create_lj(0.0002, 8.0, 0.0);

        let atom1 = create_atom("Li6").unwrap();
        let atom2 = create_atom("Li7").unwrap();
        let particles = Particles::new_pair(atom1, atom2, EnergyUnit::Kelvin.to_au(1e-7));

        let collision_params = CollisionParams::new(particles, potential);
        assert_eq!(collision_params.particles.particle_count(), 2);
    }

    #[test]
    fn test_numerov() {
        let potential = create_lj(0.0002, 8.0, 0.0);

        let atom1 = create_atom("Li6").unwrap();
        let atom2 = create_atom("Li7").unwrap();
        let particles = Particles::new_pair(atom1, atom2, EnergyUnit::Kelvin.to_au(1e-7));

        let collision_params = CollisionParams::new(particles, potential);
        let mut numerov = RatioNumerov::new(collision_params);
        numerov.prepare(7.0, (1.1, 1.2));
        numerov.propagate_to(100.0);
        let wave_ratio = numerov.wave_ratio();
        let r = numerov.r();
        let dr = numerov.dr();

        print!("r: {:?}\n", r);
        print!("dr: {:?}\n", dr);
        print!("wave_ratio: {:?}\n", wave_ratio);
    }
}
