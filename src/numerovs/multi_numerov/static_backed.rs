use std::{f64::consts::PI, mem::swap};

use nalgebra::SMatrix;
use crate::{numerovs::propagator::{MultiStep, MultiStepRule, PropagatorData, StepAction, StepRule}, potentials::potential::Potential};
use super::{MultiNumerovData, MultiRatioNumerovStep};

type MultiNumerovDataS<'a, const N: usize, P> = MultiNumerovData<'a, SMatrix<f64, N, N>, P>;

impl<P, const N: usize> MultiNumerovDataS<'_, N, P> 
where 
    P: Potential<Space = SMatrix<f64, N, N>>
{
    pub fn get_g_func(&mut self, r: f64, out: &mut SMatrix<f64, N, N>) {
        self.potential.value_inplace(r, &mut self.potential_buffer);

        out.iter_mut()
            .zip(self.unit.iter())
            .zip(self.potential_buffer.iter())
            .for_each(|((o, u), p)| {
                *o = 2.0 * self.mass * (u - p)
            });
    }
}

impl<const N: usize, P> PropagatorData for MultiNumerovData<'_, SMatrix<f64, N, N>, P> 
where 
    P: Potential<Space = SMatrix<f64, N, N>>
{
    fn step_size(&self) -> f64 {
        self.dr
    }
    
    fn current_g_func(&mut self) {
        self.potential.value_inplace(self.r + self.dr, &mut self.potential_buffer);

        self.current_g_func.iter_mut()
            .zip(self.unit.iter())
            .zip(self.potential_buffer.iter())
            .for_each(|((c, u), p)| {
                *c = 2.0 * self.mass * (u - p)
            });
    }

    fn advance(&mut self) {
        self.r += self.dr;
    }
}

impl<const N: usize, P> StepRule<MultiNumerovDataS<'_, N, P>> for MultiStepRule<MultiNumerovDataS<'_, N, P>>
where 
    P: Potential<Space = SMatrix<f64, N, N>>
{
    fn get_step(&self, data: &MultiNumerovDataS<N, P>) -> f64 {
        let mut max_g_val = 0.;
        zipped!(data.current_g_func.as_ref())
            .for_each(|unzipped!(c)| max_g_val = f64::max(max_g_val, c.read().abs()));

        let lambda = 2. * PI / max_g_val.sqrt();

        f64::clamp(lambda / self.wave_step_ratio, self.min_step, self.max_step)
    }
    
    fn assign(&mut self, data: &MultiNumerovDataS<N, P>) -> StepAction {
        let prop_step = data.step_size();
        let step = self.get_step(data);

        if prop_step > 1.2 * step {
            self.doubled_step = false;
            StepAction::Halve
        } else if prop_step < 2. * step && !self.doubled_step {
            self.doubled_step = true;
            StepAction::Double
        } else {
            self.doubled_step = false;
            StepAction::Keep
        }
    }
}

impl<const N: usize, P> MultiStep<MultiNumerovDataS<'_, N, P>> for MultiRatioNumerovStep<SMatrix<f64, N, N>>
where 
    P: Potential<Space = SMatrix<f64, N, N>>
{
    fn step(&mut self, data: &mut MultiNumerovDataFaer<P>) {
        data.r += data.dr;

        zipped!(self.buffer1.as_mut(), data.unit.as_ref(), data.current_g_func.as_ref())
            .for_each(|unzipped!(mut b1, u, c)| 
                b1.write(u.read() + data.dr * data.dr / 12. * c.read())
            );

        let mut piv = self.buffer1.partial_piv_lu();
        self.f3 = piv.inverse();

        piv = data.psi1.partial_piv_lu();
        data.psi2 = piv.inverse();
        matmul(self.buffer2.as_mut(), self.f2.as_ref(), data.psi2.as_ref(), None, 1., faer::Parallelism::None);
        zipped!(self.buffer2.as_mut(), data.unit.as_ref(), self.f1.as_ref())
            .for_each(|unzipped!(mut b2, u, f1)| 
                b2.write(12. * u.read() - 10. * f1.read() - b2.read())
            );
        matmul(data.psi2.as_mut(), self.f3.as_ref(), self.buffer2.as_ref(), None, 1., faer::Parallelism::None);

        swap(&mut self.f3, &mut self.f2);
        swap(&mut self.f2, &mut self.f1);
        swap(&mut self.f1, &mut self.buffer1);

        swap(&mut data.psi2, &mut data.psi1);
    }

    fn halve_step(&mut self, data: &mut MultiNumerovDataS<N, P>) {
        data.dr /= 2.0;

        zipped!(self.f2.as_mut(), data.unit.as_ref())
            .for_each(|unzipped!(mut f2, u)| 
                f2.write(f2.read() / 4. + 0.75 * u.read())
            );

        zipped!(self.f1.as_mut(), data.unit.as_ref())
            .for_each(|unzipped!(mut f1, u)| 
                f1.write(f1.read() / 4. + 0.75 * u.read())
            );

        data.get_g_func(data.r - data.dr, self.buffer1.as_mut());
        zipped!(self.buffer1.as_mut(), data.unit.as_ref())
            .for_each(|unzipped!(mut b1, u)| 
                b1.write(2. * u.read() - data.dr * data.dr * 10. / 12. * b1.read())
            );

        let mut piv = self.buffer1.partial_piv_lu();
        self.buffer2 = piv.inverse(); // todo! consider inverse inplace?

        zipped!(self.f2.as_mut(), data.unit.as_ref(), self.buffer1.as_ref())
            .for_each(|unzipped!(mut f2, u, b1)| 
                f2.write(1.2 * u.read() - b1.read() / 10.)
            );

        matmul(self.buffer1.as_mut(), self.f1.as_ref(), data.psi1.as_ref(), None, 1., faer::Parallelism::None);
        self.buffer1 += self.f2.as_ref();
        matmul(data.psi2.as_mut(), self.buffer2.as_ref(), self.buffer1.as_ref(), None, 1., faer::Parallelism::None);

        piv = data.psi2.partial_piv_lu();
        self.buffer1 = piv.inverse();

        matmul(self.buffer2.as_mut(), data.psi1.as_ref(), self.buffer1.as_ref(), None, 1., faer::Parallelism::None);
        swap(&mut data.psi1, &mut self.buffer2);
    }

    fn double_step(&mut self, data: &mut MultiNumerovDataS<N, P>) {
        data.dr *= 2.;

        zipped!(self.f2.as_mut(), data.unit.as_ref(), self.f3.as_ref())
            .for_each(|unzipped!(mut f2, u, f3)| 
                f2.write(4.0 * f3.read() - 3. * u.read())
            );

        zipped!(self.f1.as_mut(), data.unit.as_ref())
            .for_each(|unzipped!(mut f1, u)| 
                f1.write(4.0 * f1.read() - 3. * u.read())
            );

        matmul(self.buffer1.as_mut(), data.psi1.as_ref(), data.psi2.as_ref(), None, 1., faer::Parallelism::None);
        swap(&mut self.buffer1, &mut data.psi1);
    }
}