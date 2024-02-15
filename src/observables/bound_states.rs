use quantum::units::{Unit, energy_units::Energy, Au};

use crate::{collision_params::CollisionParams, numerovs::{ratio_numerov::RatioNumerov, propagator::Numerov}, potentials::potential::Potential, boundary::{Boundary, Direction}, defaults::SingleDefaults};

pub struct SingleBounds;

impl SingleBounds {
    /// Returns highest bound state in energy range if it exists
    pub fn bound_energy<P, U: Unit>(collision_params: &mut CollisionParams<P>, energy_range: (Energy<U>, Energy<U>), r_range: (f64, f64), err: Energy<U>) -> Option<Energy<Au>>
    where
        P: Potential<Space = f64>    
    {
        let boundary = Boundary::new(r_range.0, Direction::Outwards, SingleDefaults::boundary());

        let mut upper_energy = energy_range.1.to_au();
        let mut lower_energy = energy_range.0.to_au();

        collision_params.particles.internals.insert_value("energy", upper_energy);
        let mut numerov = RatioNumerov::new(&collision_params, 1.0);
        numerov.prepare(&boundary);
        let (diff, mut nodes_max) = Self::bound_diffs(collision_params, r_range); 
        numerov.propagate_node_counting(r_range.1);
        if diff < 0.0 {
            nodes_max -= 1;
        }

        collision_params.particles.internals.insert_value("energy", lower_energy);
        let mut numerov = RatioNumerov::new(&collision_params, 1.0);
        numerov.prepare(&boundary);
        let (diff, mut nodes_min) = Self::bound_diffs(collision_params, r_range);
        numerov.propagate_node_counting(r_range.1);
        if diff < 0.0 {
            nodes_min -= 1;
        }

        if nodes_max == nodes_min {
            return None;
        }

        let err = err.to_au(); 
        while upper_energy - lower_energy > err {
            let mid_energy = (upper_energy + lower_energy) / 2.0;

            collision_params.particles.internals.insert_value("energy", mid_energy);
            let (diff, nodes) =  Self::bound_diffs(&collision_params, r_range);

            if nodes > nodes_max {
                upper_energy = mid_energy
            } else if nodes < nodes_max{
                lower_energy = mid_energy
            } else if diff > 0.0 {
                upper_energy = mid_energy;
            } else {
                lower_energy = mid_energy
            }
        }
        collision_params.particles.internals.insert_value("energy", (lower_energy + upper_energy) / 2.0);

        Some(Energy((lower_energy + upper_energy) / 2.0, Au))
    }
    
    pub fn n_bound_energy<P, U: Unit>(collision_params: &mut CollisionParams<P>, n_bound: isize, r_range: (f64, f64), err: Energy<U>) -> Energy<Au>
    where
        P: Potential<Space = f64>    
    {
        let err = err.to_au(); 
        let boundary = Boundary::new(r_range.0, Direction::Outwards, SingleDefaults::boundary());

        let mut upper_energy = collision_params.potential.asymptotic_value() - 1e-50;
        let low_energy = collision_params.potential.value(&((r_range.0 + r_range.1) / 2.0));
        
        if n_bound < 0 {
            collision_params.particles.internals.insert_value("energy", upper_energy);
            let mut numerov = RatioNumerov::new(&collision_params, 1.0);
            numerov.prepare(&boundary);
            let (diff, mut nodes_max) = Self::bound_diffs(collision_params, r_range); 
            numerov.propagate_node_counting(r_range.1);
            if diff < 0.0 {
                nodes_max -= 1;
            }

            if nodes_max < (-n_bound) as usize - 1{
                panic!("Not enough bound states for n = {}", n_bound);
            }
            let target_nodes = nodes_max - n_bound.abs() as usize + 1;

            let mut lower_energy = low_energy / 128.0;

            collision_params.particles.internals.insert_value("energy", lower_energy);
            let mut numerov = RatioNumerov::new(&collision_params, 1.0);
            numerov.prepare(&boundary);
            let mut lower_nodes = numerov.propagate_node_counting(r_range.1);

            while lower_nodes >= target_nodes {
                lower_energy *= 2.0;

                collision_params.particles.internals.insert_value("energy", lower_energy);
                let mut numerov = RatioNumerov::new(&collision_params, 1.0);
                numerov.prepare(&boundary);
                lower_nodes = numerov.propagate_node_counting(r_range.1);
            }

            while upper_energy - lower_energy > err {
                let mid_energy = (upper_energy + lower_energy) / 2.0;

                collision_params.particles.internals.insert_value("energy", mid_energy);
                let (diff, nodes) =  Self::bound_diffs(&collision_params, r_range);

                if nodes > target_nodes {
                    upper_energy = mid_energy
                } else if nodes < target_nodes{
                    lower_energy = mid_energy
                } else if diff > 0.0 {
                    upper_energy = mid_energy;
                } else {
                    lower_energy = mid_energy
                }
            }
            
            collision_params.particles.internals.insert_value("energy", (lower_energy + upper_energy) / 2.0);
        } else {
            // Only asymptotic bound states are supported for now
            todo!()
        }
        
        Energy(collision_params.particles.internals.get_value("energy"), Au)
    }

    pub fn bound_wave<P>(collision_params: &CollisionParams<P>, r_range: (f64, f64)) -> (Vec<f64>, Vec<f64>)     
    where
    P: Potential<Space = f64>
    {
        let inwards_boundary = Boundary::new(r_range.1, Direction::Inwards, SingleDefaults::boundary());
        let outwards_boundary = Boundary::new(r_range.0, Direction::Outwards, SingleDefaults::boundary());

        let mut numerov = RatioNumerov::new(&collision_params, 1.0);
        numerov.prepare(&inwards_boundary);

        let mut wave_inwards = Vec::new();
        let mut rs_inwards = Vec::new();
        wave_inwards.push(1.0);
        rs_inwards.push(numerov.r());

        while *numerov.wave_last() > 1.0 && numerov.r() > r_range.0 {
            numerov.single_step();
            wave_inwards.insert(0, *numerov.wave_last() * wave_inwards.first().unwrap());
            rs_inwards.insert(0, numerov.r());
        }

        numerov.prepare(&outwards_boundary);
        let r_match = *rs_inwards.first().unwrap();
        let (mut rs, mut wave) = numerov.propagate_values(r_match, 1e-50);
        rs.pop();
        wave.pop();

        let normalization = wave_inwards.first().unwrap() / wave.last().unwrap();
        wave.iter_mut().for_each(|w| *w *= normalization);

        wave.extend(wave_inwards);
        rs.extend(rs_inwards);

        (rs, wave)
    }

    pub fn bound_diff_dependence<P, U: Unit>(mut collision_params: CollisionParams<P>, energies: &[Energy<U>], r_range: (f64, f64)) -> (Vec<f64>, Vec<usize>)
    where
        P: Potential<Space = f64>
    {
        let mut bound_differences = Vec::new();
        let mut node_counts = Vec::new();
        for energy in energies {
            collision_params.particles.internals.insert_value("energy", energy.to_au());
            
            let (bound_difference, node_count) = Self::bound_diffs(&collision_params, r_range);
            bound_differences.push(bound_difference);
            node_counts.push(node_count);
        }
        
        (bound_differences, node_counts)
    }

    pub fn bound_diffs<P>(collision_params: &CollisionParams<P>, r_range: (f64, f64)) -> (f64, usize)
    where
        P: Potential<Space = f64>, 
    {   
        let inwards_boundary = Boundary::new(r_range.1, Direction::Inwards, SingleDefaults::boundary());
        let outwards_boundary = Boundary::new(r_range.0, Direction::Outwards, SingleDefaults::boundary());

        let mut numerov = RatioNumerov::new(&collision_params, 1.0);
        numerov.prepare(&inwards_boundary);

        while *numerov.wave_last() > 1.0 && numerov.r() > r_range.0 {
            numerov.single_step();
        }
        let outwards_result = numerov.result();
        
        numerov.prepare(&outwards_boundary);
        let r_match = outwards_result.r_last;
        
        let node_count = numerov.propagate_node_counting(r_match);
        numerov.propagate_to(r_match);
        let inwards_result = numerov.result();

        ((1.0 / inwards_result.wave_ratio - 1.0) * (1.0 / inwards_result.dr.abs()) - (outwards_result.wave_ratio - 1.0) * (1.0 / outwards_result.dr.abs()), node_count)
    }
}

impl<P> RatioNumerov<'_, f64, P> 
where
    P: Potential<Space = f64>
{
    pub(self) fn propagate_node_counting(&mut self, r_stop: f64) -> usize {
        let mut node_count = 0;
        while self.r() < r_stop {
            self.single_step();

            if *self.wave_last() < 0.0 {
                node_count += 1;
            }
        }

        node_count
    }
}