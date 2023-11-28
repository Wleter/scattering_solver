use crate::{collision_params::CollisionParams, numerovs::{ratio_numerov::RatioNumerov, propagator::Numerov}, potentials::potential::Potential, boundary::{Boundary, Direction}, defaults::SingleDefaults};


pub struct SingleBounds;

impl SingleBounds {
    pub fn bound_energy<P>(collision_params: &mut CollisionParams<P>, n_bound: isize, r_min: f64, r_max: f64, err: f64) -> f64
    where
        P: Potential<Space = f64>    
    {
        let boundary = Boundary::new(r_min, Direction::Outwards, SingleDefaults::boundary());

        let mut upper_energy = collision_params.potential.asymptotic_value() - 1e-50;
        let low_energy = collision_params.potential.value(&((r_min + r_max) / 2.0));
        
        if n_bound < 0 {
            collision_params.particles.internals.insert_value("energy", upper_energy);
            let mut numerov = RatioNumerov::new(&collision_params, 1.0);
            numerov.prepare(&boundary);
            let nodes_max = numerov.propagate_node_counting(r_max);
            if nodes_max < (-n_bound) as usize {
                panic!("Not enough bound states for n = {}", n_bound);
            }
            let target_nodes = nodes_max - n_bound.abs() as usize;

            let mut lower_energy = low_energy / 128.0;

            collision_params.particles.internals.insert_value("energy", lower_energy);
            let mut numerov = RatioNumerov::new(&collision_params, 1.0);
            numerov.prepare(&boundary);
            let mut lower_nodes = numerov.propagate_node_counting(r_max);

            while lower_nodes >= target_nodes {
                lower_energy *= 2.0;

                collision_params.particles.internals.insert_value("energy", lower_energy);
                let mut numerov = RatioNumerov::new(&collision_params, 1.0);
                numerov.prepare(&boundary);
                lower_nodes = numerov.propagate_node_counting(r_max);
            }

            while upper_energy - lower_energy > err {
                let mid_energy = (upper_energy + lower_energy) / 2.0;

                collision_params.particles.internals.insert_value("energy", mid_energy);
                let (diff, nodes) =  Self::bound_diffs(&collision_params, r_min, r_max);

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
        
        collision_params.particles.internals.get_value("energy")
    }

    pub fn bound_diff_dependence<P>(mut collision_params: CollisionParams<P>, energies: &[f64], r_min: f64, r_max: f64) -> (Vec<f64>, Vec<usize>)
    where
        P: Potential<Space = f64>
    {
        let mut bound_differences = Vec::new();
        let mut node_counts = Vec::new();
        for energy in energies {
            collision_params.particles.internals.insert_value("energy", *energy);
            
            let (bound_difference, node_count) = Self::bound_diffs(&collision_params, r_min, r_max);
            bound_differences.push(bound_difference);
            node_counts.push(node_count);
        }
        
        (bound_differences, node_counts)
    }

    pub fn bound_diffs<P>(collision_params: &CollisionParams<P>, r_min: f64, r_max: f64) -> (f64, usize)
    where
        P: Potential<Space = f64>, 
    {   
        let inwards_boundary = Boundary::new(r_max, Direction::Inwards, SingleDefaults::boundary());
        let outwards_boundary = Boundary::new(r_min, Direction::Outwards, SingleDefaults::boundary());

        let mut numerov = RatioNumerov::new(&collision_params, 1.0);
        numerov.prepare(&inwards_boundary);

        while *numerov.wave_last() > 1.0 {
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