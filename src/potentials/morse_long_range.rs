use quantum::units::{energy_units::Energy, Unit};

use super::{dispersion_potential::DispersionPotential, potential::Potential};


pub struct MorseLongRangeBuilder {
    d0: f64,
    r_e: f64,
    tail: Vec<DispersionPotential>,

    p: Option<i32>,
    q: Option<i32>,

    r_ref: Option<f64>,
    rho: Option<f64>,
    betas: Vec<f64>,
}

impl MorseLongRangeBuilder {
    pub fn new<U: Unit>(d0: Energy<U>, r_e: f64, tail: Vec<DispersionPotential>) -> Self {
        let d0 = d0.to_au();

        Self {
            d0,
            tail,
            p: None,
            q: None,
            r_ref: None,
            r_e,
            rho: None,
            betas: vec![],
        }
    }

    pub fn set_betas(self, betas: Vec<f64>) -> Self {
        Self {
            betas,
            ..self
        }
    }

    pub fn set_params(self, p: i32, q: i32, r_ref: f64, rho: f64) -> Self {
        Self {
            p: Some(p),
            q: Some(q),
            r_ref: Some(r_ref),
            rho: Some(rho),
            ..self
        }
    }

    pub fn build(self) -> MorseLongRange {
        let p = self.p.unwrap();
        let q = self.q.unwrap();
        let r_ref = self.r_ref.unwrap();
        let r_e = self.r_e;
        let rho = self.rho;
        let betas = self.betas;

        let tail_re: f64 = if let Some(rho) = rho {
            self.tail.iter().map(|tail| douketis_damping(r_e, rho, -tail.n) * tail.value(&r_e)).sum()
        } else {
            self.tail.iter().map(|tail| tail.value(&r_e)).sum()
        };
        
        let b_inf = (2.0f64 * self.d0 / tail_re).ln();

        MorseLongRange {
            d0: self.d0,
            tail: self.tail,
            p,
            q,
            r_ref,
            r_e,
            rho,
            betas,
            b_inf,
            tail_re
        }
    }
}

pub struct MorseLongRange {
    d0: f64,
    tail: Vec<DispersionPotential>,
    p: i32,
    q: i32,
    r_ref: f64,
    r_e: f64,
    rho: Option<f64>,
    betas: Vec<f64>,
    b_inf: f64,

    tail_re: f64
}

impl MorseLongRange {
    fn u_lr(&self, r: f64) -> f64 {
        if let Some(rho) = self.rho {
            self.tail.iter()
            .map(|tail| douketis_damping(r, rho, -tail.n) * tail.value(&r))
            .sum()
        } else {
            self.tail.iter()
            .map(|tail| tail.value(&r))
            .sum()
        }
    }
    
    fn beta(&self, r: f64) -> f64 {
        let y_p = y_func(r, self.p, self.r_ref);
        let y_q = y_func(r, self.q, self.r_ref);

        let beta_factor = self.betas.iter()
            .enumerate()
            .map(|(i, beta)| beta * y_q.powi(i as i32))
            .sum::<f64>();

        self.b_inf * y_p + (1. - y_p) * beta_factor
    }
}

impl Potential for MorseLongRange {
    type Space = f64;
    
    fn value(&self, r: &f64) -> f64 {
        let r = *r;
        self.d0 * (1. - self.u_lr(r) / self.tail_re * (-self.beta(r) * y_func(r, self.p, self.r_e)).exp()).powi(2) - self.d0
    }
    
    fn size(&self) -> usize {
        1
    }

}

#[inline(always)]
fn douketis_damping(r: f64, rho: f64, m: i32) -> f64 {
    (1. - (-(3.3 + 0.423 * rho * r) * rho * r / (m as f64)).exp()).powi(m - 1)
}

#[inline(always)]
fn y_func(r: f64, n: i32, x: f64) -> f64 {
    (r.powi(n) - x.powi(n)) / (r.powi(n) + x.powi(n))
}


#[cfg(test)]
mod test {
    use super::*;
    use quantum::units::{energy_units::{CmInv, Energy}, Au};

    #[test]
    fn test_morse() {
        let d0 = Energy(333.7795, CmInv);
        let tail = vec![
            DispersionPotential::new(Energy(1394.180, Au), -6, 0.0),
            DispersionPotential::new(Energy(83461.675549, Au), -8, 0.0),
            DispersionPotential::new(Energy(7374640.77, Au), -10, 0.0),
        ];
        
        let morse = MorseLongRangeBuilder::new(d0, 7.880185, tail)
            .set_params(5, 3, 15.11784, 0.54)
            .set_betas(vec![-0.516129, -0.0980, 0.1133, -0.0251])
            .build();

        let r = 7.880185;
        let val = Energy(morse.value(&r), Au).to(CmInv);

        assert!(d0.value() + val.value() < 1e-3, "Expected: {}, Got: {}", d0.value(), val.value());
    }
}