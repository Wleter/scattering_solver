use std::{f64::consts::PI, rc::Rc, mem::swap};

use num::complex::Complex64;
use quantum::particles::Particles;

use crate::{
    boundary::Boundary,
    potentials::potential::{OnePotential, MultiPotential, MultiCPotential},
    types::{CMatrix, FMatrix},
};

use super::propagator::{MultiStep, Numerov, NumerovResult};

/// Numerov method propagating ratios of the wave function,
/// implementing Numerov and NumerovResult trait for single channel and multi channel cases
pub struct RatioNumerov<T, P>
{
    pub potential: P,
    pub particles: Rc<Particles>,
    energy: f64,
    mass: f64,

    r: f64,
    dr: f64,
    psi1: T,
    psi2: T,

    f1: T,
    f2: T,
    f3: T,

    identity: T,
    current_g_func: T,

    doubled_step_before: bool,
    is_set_up: bool,
    step_factor: f64,
}

impl RatioNumerov<f64, Rc<dyn OnePotential>>
{
    /// Creates a new instance of the RatioNumerov struct
    pub fn new_single(particles: Rc<Particles>, potential: Rc<dyn OnePotential>, step_factor: f64) -> Self {
        let mass = particles.red_mass();
        let energy = particles.internals.get_value("energy");

        Self {
            potential,
            particles,
            energy,
            mass,

            r: 0.0,
            dr: 0.0,
            psi1: 0.0,
            psi2: 0.0,

            f1: 0.0,
            f2: 0.0,
            f3: 0.0,

            identity: 1.0,
            current_g_func: 0.0,

            doubled_step_before: false,
            is_set_up: false,
            step_factor,
        }
    }
}

impl RatioNumerov<FMatrix, Rc<dyn MultiPotential>>
{
    /// Creates a new instance of the RatioNumerov struct
    pub fn new_multi(particles: Rc<Particles>, potential: Rc<dyn MultiPotential>, step_factor: f64) -> Self {
        let mass = particles.red_mass();
        let energy = particles.internals.get_value("energy");
        let dim = potential.dim();
        let zero = FMatrix::zeros(dim, dim);

        Self {
            potential,
            particles,
            energy,
            mass,

            r: 0.0,
            dr: 0.0,
            psi1: zero.clone(),
            psi2: zero.clone(),

            f1: zero.clone(),
            f2: zero.clone(),
            f3: zero.clone(),

            identity: FMatrix::identity(dim, dim),
            current_g_func: zero,

            doubled_step_before: false,
            is_set_up: false,
            step_factor,
        }
    }
}

impl RatioNumerov<CMatrix, Rc<dyn MultiCPotential>>
{
    /// Creates a new instance of the RatioNumerov struct
    pub fn new_multi_c(particles: Rc<Particles>, potential: Rc<dyn MultiCPotential>, step_factor: f64) -> Self {
        let mass = particles.red_mass();
        let energy = particles.internals.get_value("energy");
        let dim = potential.dim();
        let zero = CMatrix::zeros(dim, dim);

        Self {
            potential,
            particles,
            energy,
            mass,

            r: 0.0,
            dr: 0.0,
            psi1: zero.clone(),
            psi2: zero.clone(),

            f1: zero.clone(),
            f2: zero.clone(),
            f3: zero.clone(),

            identity: CMatrix::identity(dim, dim),
            current_g_func: zero,

            doubled_step_before: false,
            is_set_up: false,
            step_factor,
        }
    }
}


impl RatioNumerov<f64, Rc<dyn OnePotential>>
{
    /// Returns the g function described in the Numerov method at position r
    #[inline(always)]
    fn g_func(&mut self, &r: &f64) -> f64 {
        2.0 * self.mass * (self.energy - self.potential.value(&r))
    }
}

impl MultiStep for RatioNumerov<f64, Rc<dyn OnePotential>>
{
    fn variable_step(&mut self) {
        self.current_g_func = self.g_func(&(self.r + self.dr));
        
        let step_size = self.recommended_step_size();
        if step_size > 2.0 * self.dr.abs() && !self.doubled_step_before {
            self.doubled_step_before = true;
            self.double_step();
            self.current_g_func = self.g_func(&(self.r + self.dr));
        } else {
            self.doubled_step_before = false;
            let mut halved = false;
            while 1.2 * step_size < self.dr.abs() {
                self.half_step();
                halved = true;
            }
            
            if halved {
                self.current_g_func = self.g_func(&(self.r + self.dr));
            }
        }

        self.step();

    }

    fn step(&mut self) {
        self.r += self.dr;

        let f = 1.0 + self.dr * self.dr * self.current_g_func / 12.0;
        let psi = (12.0 - 10.0 * self.f1 - self.f2 / self.psi1) / f;

        self.f3 = self.f2;
        self.f2 = self.f1;
        self.f1 = f;

        self.psi2 = self.psi1;
        self.psi1 = psi;
    }

    fn half_step(&mut self) {
        self.dr /= 2.0;

        let f = 1.0 + self.dr * self.dr * self.g_func(&(self.r - self.dr)) / 12.0;
        self.f1 = self.f1 / 4.0 + 0.75;
        self.f2 = self.f2 / 4.0 + 0.75;

        let psi = (self.f1 * self.psi1 + self.f2) / (12.0 - 10.0 * f);
        self.f2 = f;

        self.psi2 = psi;
        self.psi1 /= psi;
    }

    fn double_step(&mut self) {
        self.dr *= 2.0;

        self.f2 = 4.0 * self.f3 - 3.0;
        self.f1 = 4.0 * self.f1 - 3.0;

        self.psi1 *= self.psi2;
    }

    fn recommended_step_size(&mut self) -> f64 {
        let lambda = 2.0 * PI / self.current_g_func.abs().sqrt();
        let lambda_step_ratio = 500.0;

        self.step_factor * lambda / lambda_step_ratio
    }
}

impl Numerov<f64> for RatioNumerov<f64, Rc<dyn OnePotential>>
{
    fn prepare(&mut self, boundary: &Boundary<f64>) {
        self.r = boundary.r_start;

        self.current_g_func = self.g_func(&boundary.r_start);
        self.dr = self.recommended_step_size();

        self.psi1 = boundary.start_value;
        self.psi2 = boundary.before_value;

        self.f3 = 1.0 + self.dr * self.dr * self.g_func(&(self.r - 2.0 * self.dr)) / 12.0;
        self.f2 = 1.0 + self.dr * self.dr * self.g_func(&(self.r - self.dr)) / 12.0;
        self.f1 = 1.0 + self.dr * self.dr * self.current_g_func / 12.0;

        self.is_set_up = true;
    }

    fn propagate_to(&mut self, r: f64) {
        assert!(self.is_set_up, "Numerov method not set up");
        while self.r < r {
            self.variable_step();
        }
    }

    fn propagate_values(&mut self, r: f64, wave_init: f64) -> (Vec<f64>, Vec<f64>) {
        assert!(self.is_set_up, "Numerov method not set up");
        let max_capacity: usize = 1000;
        let r_push_step = (r - self.r) / max_capacity as f64;

        let mut wave_functions = Vec::with_capacity(max_capacity);
        let mut positions = Vec::with_capacity(max_capacity);

        let mut psi_actual = wave_init;
        positions.push(self.r);
        wave_functions.push(psi_actual);

        while self.r < r {
            self.variable_step();

            psi_actual = self.psi1 * psi_actual;

            if (self.r - positions.last().unwrap()) > r_push_step {
                positions.push(self.r);
                wave_functions.push(psi_actual);
            }
        }

        (positions, wave_functions)
    }

    fn result(&self) -> NumerovResult<f64> {
        NumerovResult {
            r_last: self.r,
            dr: self.dr,
            wave_ratio: self.psi1,
        }
    }
}

impl RatioNumerov<FMatrix, Rc<dyn MultiPotential>>
{
    /// Returns the g function described in the Numerov method at position r
    #[inline(always)]
    fn g_func(&mut self, &r: &f64) -> FMatrix {
        2.0 * self.mass * (self.energy * &self.identity - self.potential.value(&r))
    }
}

impl MultiStep for RatioNumerov<FMatrix, Rc<dyn MultiPotential>>
{
    fn variable_step(&mut self) {
        self.current_g_func = self.g_func(&(self.r + self.dr));
        
        let step_size = self.recommended_step_size();
        if step_size > 2.0 * self.dr && !self.doubled_step_before {
            self.doubled_step_before = true;
            self.double_step();
            self.current_g_func = self.g_func(&(self.r + self.dr));
        } else {
            self.doubled_step_before = false;
            let mut halved = false;
            while 1.2 * step_size < self.dr.abs() {
                halved = true;
                self.half_step();
            }
            if halved {
                self.current_g_func = self.g_func(&(self.r + self.dr));
            }
            
        }
        
        self.step();

    }

    fn step(&mut self) {
        self.r += self.dr;

        let f = &self.identity + self.dr * self.dr * &self.current_g_func / 12.0;
        let psi = f.clone().try_inverse().unwrap()
            * (12.0 * &self.identity - 10.0 * &self.f1 - &self.f2 * self.psi1.clone().try_inverse().unwrap());

        swap(&mut self.f3, &mut self.f2);
        swap(&mut self.f2, &mut self.f1);
        self.f1 = f;

        self.psi2 = self.psi1.clone();
        self.psi1 = psi;
    }

    fn half_step(&mut self) {
        self.dr /= 2.0;
        let g_func = self.g_func(&(self.r - self.dr));

        let f = &self.identity + self.dr * self.dr * &g_func / 12.0;
        self.f1 = &self.f1 / 4.0 + 0.75 * &self.identity;
        self.f2 = &self.f2 / 4.0 + 0.75 * &self.identity;

        let psi = (12.0 * &self.identity - 10.0 * &f).try_inverse().unwrap()
            * (&self.f1 * &self.psi1 + &self.f2);
        self.f2 = f;

        self.psi2 = psi.clone();
        self.psi1 *= psi.try_inverse().unwrap();
    }

    fn double_step(&mut self) {
        self.dr *= 2.0;

        self.f2 = 4.0 * &self.f3 - 3.0 * &self.identity;
        self.f1 = 4.0 * &self.f1 - 3.0 * &self.identity;

        self.psi1 *= &self.psi2;
    }

    fn recommended_step_size(&mut self) -> f64 {
        let max_g_func_val = self
            .current_g_func
            .iter()
            .fold(0.0, |acc, &x| x.abs().max(acc));

        let lambda = 2.0 * PI / max_g_func_val.sqrt();
        let lambda_step_ratio = 500.0;

        self.step_factor * lambda / lambda_step_ratio
    }
}

impl Numerov<FMatrix> for RatioNumerov<FMatrix, Rc<dyn MultiPotential>>
{
    fn prepare(&mut self, boundary: &Boundary<FMatrix>) {
        self.r = boundary.r_start;

        self.current_g_func = self.g_func(&boundary.r_start);
        self.dr = self.recommended_step_size();

        self.psi1 = boundary.start_value.clone();
        self.psi2 = boundary.before_value.clone();

        let g_func_3 = self.g_func(&(self.r - 2.0 * self.dr));
        let g_func_2 = self.g_func(&(self.r - self.dr));

        self.f3 = &self.identity + self.dr * self.dr * g_func_3 / 12.0;
        self.f2 = &self.identity + self.dr * self.dr * g_func_2 / 12.0;
        self.f1 = &self.identity + self.dr * self.dr * &self.current_g_func / 12.0;

        self.is_set_up = true;
    }

    fn propagate_to(&mut self, r: f64) {
        assert!(self.is_set_up, "Numerov method not set up");
        while self.r < r {
            self.variable_step();
        }
    }

    fn propagate_values(&mut self, r: f64, wave_init: FMatrix) -> (Vec<f64>, Vec<FMatrix>) {
        assert!(self.is_set_up, "Numerov method not set up");
        let max_capacity: usize = 1000;
        let r_push_step = (r - self.r) / max_capacity as f64;

        let mut wave_functions = Vec::with_capacity(max_capacity);
        let mut positions = Vec::with_capacity(max_capacity);

        let mut psi_actual = wave_init;
        positions.push(self.r);
        wave_functions.push(psi_actual.clone());

        while self.r < r {
            self.variable_step();

            psi_actual = &self.psi1 * psi_actual;

            if (self.r - positions.last().unwrap()) > r_push_step {
                positions.push(self.r);
                wave_functions.push(psi_actual.clone());
            }
        }

        (positions, wave_functions)
    }

    fn result(&self) -> NumerovResult<FMatrix> {
        NumerovResult {
            r_last: self.r,
            dr: self.dr,
            wave_ratio: self.psi1.clone(),
        }
    }
}

impl RatioNumerov<CMatrix, Rc<dyn MultiCPotential>>
{
    /// Returns the g function described in the Numerov method at position r
    fn g_func(&mut self, &r: &f64) -> CMatrix {
        (&self.identity * Complex64::from(self.energy) - self.potential.value(&r)) * Complex64::from(2.0 * self.mass)
    }
}

impl MultiStep for RatioNumerov<CMatrix, Rc<dyn MultiCPotential>>
{
    fn variable_step(&mut self) {
        self.current_g_func = self.g_func(&(self.r + self.dr));

        let step_size = self.recommended_step_size();

        if step_size > 2.0 * self.dr && !self.doubled_step_before {
            self.doubled_step_before = true;
            self.double_step();
            self.current_g_func = self.g_func(&(self.r + self.dr));
        } else {
            self.doubled_step_before = false;
            let mut halved = false;
            while 1.2 * step_size < self.dr.abs() {
                halved = true;
                self.half_step();
            }
            if halved {
                self.current_g_func = self.g_func(&(self.r + self.dr));
            }
        }

        self.step();
    }

    fn step(&mut self) {
        self.r += self.dr;

        let f = &self.identity + &self.current_g_func * Complex64::from(self.dr * self.dr / 12.0);
        let psi = f.clone().try_inverse().unwrap()
            * (&self.identity * Complex64::from(12.0)
                - &self.f1 * Complex64::from(10.0)
                - &self.f2 * self.psi1.clone().try_inverse().unwrap());

        swap(&mut self.f3, &mut self.f2);
        swap(&mut self.f2, &mut self.f1);
        self.f1 = f;

        self.psi2 = self.psi1.clone();
        self.psi1 = psi;
    }

    fn half_step(&mut self) {
        self.dr /= 2.0;
        let g_func = self.g_func(&(self.r - self.dr));

        let f = &self.identity
            + g_func * Complex64::from(self.dr * self.dr / 12.0);
        self.f1 = &self.f1 / Complex64::from(4.0) + &self.identity * Complex64::from(0.75);
        self.f2 = &self.f2 / Complex64::from(4.0) + &self.identity * Complex64::from(0.75);

        let psi = (&self.identity * Complex64::from(12.0) - &f * Complex64::from(10.0))
            .try_inverse()
            .unwrap()
            * (&self.f1 * &self.psi1 + &self.f2);
        self.f2 = f;

        self.psi2 = psi.clone();
        self.psi1 *= psi.try_inverse().unwrap();
    }

    fn double_step(&mut self) {
        self.dr *= 2.0;

        self.f2 = &self.f3 * Complex64::from(4.0) - &self.identity * Complex64::from(3.0);
        self.f1 = &self.f1 * Complex64::from(4.0) - &self.identity * Complex64::from(3.0);

        self.psi1 *= &self.psi2;
    }

    fn recommended_step_size(&mut self) -> f64 {
        let max_g_func_val = self
            .current_g_func
            .iter()
            .fold(0.0, |acc, &x| x.norm().max(acc));

        let lambda = 2.0 * PI / max_g_func_val.sqrt();
        let lambda_step_ratio = 500.0;

        self.step_factor * lambda / lambda_step_ratio
    }
}

impl Numerov<CMatrix> for RatioNumerov<CMatrix, Rc<dyn MultiCPotential>>
{
    fn prepare(&mut self, boundary: &Boundary<CMatrix>) {
        self.r = boundary.r_start;

        self.current_g_func = self.g_func(&boundary.r_start);
        self.dr = self.recommended_step_size();

        self.psi1 = boundary.start_value.clone();
        self.psi2 = boundary.before_value.clone();

        let g_func_3 = self.g_func(&(self.r - 2.0 * self.dr));
        let g_func_2 = self.g_func(&(self.r - self.dr));

        self.f3 = &self.identity + g_func_3 * Complex64::from(self.dr * self.dr / 12.0);
        self.f2 = &self.identity + g_func_2 * Complex64::from(self.dr * self.dr / 12.0);
        self.f1 = &self.identity + &self.current_g_func * Complex64::from(self.dr * self.dr / 12.0);

        self.is_set_up = true;
    }

    fn propagate_to(&mut self, r: f64) {
        assert!(self.is_set_up, "Numerov method not set up");
        while self.r < r {
            self.variable_step();
        }
    }

    fn propagate_values(&mut self, r: f64, wave_init: CMatrix) -> (Vec<f64>, Vec<CMatrix>) {
        assert!(self.is_set_up, "Numerov method not set up");

        let max_capacity: usize = 1000;
        let r_push_step = (r - self.r) / max_capacity as f64;

        let mut wave_functions = Vec::with_capacity(max_capacity);
        let mut positions = Vec::with_capacity(max_capacity);

        let mut psi_actual = wave_init;
        positions.push(self.r);
        wave_functions.push(psi_actual.clone());

        while self.r < r {
            self.variable_step();

            psi_actual = &self.psi1 * psi_actual;

            if (self.r - positions.last().unwrap()) > r_push_step {
                positions.push(self.r);
                wave_functions.push(psi_actual.clone());
            }
        }

        (positions, wave_functions)
    }

    fn result(&self) -> NumerovResult<CMatrix> {
        NumerovResult {
            r_last: self.r,
            dr: self.dr,
            wave_ratio: self.psi1.clone(),
        }
    }
}
