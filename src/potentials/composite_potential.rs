use super::potential::OnePotential;

/// Composite potential that has value equal to the sum of its components
#[derive(Clone)]
pub struct CompositePotential {
    potentials: Vec<Box<dyn OnePotential>>,
}

impl CompositePotential {
    pub fn new() -> Self {
        Self {
            potentials: Vec::new(),
        }
    }

    pub fn add_potential(&mut self, potential: Box<dyn OnePotential>) -> &mut Self { 
        let potential: Box<dyn OnePotential> = potential;        
        self.potentials.push(potential);

        self
    }
}

impl OnePotential for CompositePotential {
    #[inline(always)]
    fn value(&self, r: &f64) -> f64 {
        self.potentials.iter().map(|p| p.value(r)).sum()
    }
}
