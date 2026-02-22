use pulp::Simd;
use pulp::x86::{V3, V4};
use pulp::{f32x8, f64x4, f32x16, f64x8};
use num_traits::{Float, FromPrimitive};
use core::iter;
use std::marker::PhantomData;
use paste::paste;
use crate::const_parameters::*;
use crate::scalar::eval_weideman;
use crate::calc_constants;

// This supertrait extends pulp's Simd trait over various SIMD data types (e.g. f32x8, f64x4, ...)
// enabling a generic implementation of the algorithm. Only required methods are generailzed.  
pub(crate) trait SimdData<T, V> : Simd {
    fn splat(&self, x: T) -> V;
    fn as_simd<'a>(&'a self, x: &'a [T]) -> (&'a [V], &'a [T] );
    fn as_mut_simd<'a>(&'a self, x: &'a mut [T]) -> (&'a mut [V], &'a mut [T] );
    fn add(&self, l: V, r:V) -> V;
    fn mul(&self, l: V, r:V) -> V;
    fn mul_by_const(&self, l: V, r:T) -> V;
    fn mul_add(&self, l: V, m:V, r:V) -> V;
    fn mul_sub(&self, l: V, m:V, r:V) -> V;
    fn div(&self, l: V, r:V) -> V;
    fn sub(&self, l: V, r:V) -> V;
    fn z1_mul_z2_add_r(&self, z1_re: &mut V, z1_im: &mut V, z2_re: V, z2_im: V, r: T);
}

macro_rules! impl_simd_data {
    ($float_prec:ty, $simd_data:ty, $instr_set:ty) => {
        impl SimdData<$float_prec, $simd_data> for $instr_set {
            paste! {
                #[inline(always)]
                fn as_simd<'a>(&'a self, x: &'a [$float_prec]) -> (&'a [$simd_data], &'a [$float_prec] ) {
                    $instr_set::[<as_simd_ $float_prec s>](x)
                }

                #[inline(always)]    
                fn as_mut_simd<'a>(&'a self, x: &'a mut [$float_prec]) -> (&'a mut [$simd_data], &'a mut [$float_prec] ) {
                    $instr_set::[<as_mut_simd_ $float_prec s>](x)
                }

                #[inline(always)]    
                fn splat(&self, x: $float_prec) -> $simd_data {
                    self.[<splat_ $simd_data>](x)
                }

                #[inline(always)]    
                fn add(&self, l: $simd_data, r: $simd_data) -> $simd_data {
                    self.[<add_ $simd_data>](l, r)
                }

                #[inline(always)]    
                fn mul(&self, l: $simd_data, r: $simd_data) -> $simd_data {
                    self.[<mul_ $simd_data>](l, r)
                }

                #[inline(always)]    
                fn mul_by_const(&self, l: $simd_data, r: $float_prec) -> $simd_data {
                    self.mul(l, self.splat(r))
                }

                #[inline(always)]
                fn div(&self, l: $simd_data, r: $simd_data) -> $simd_data {
                    self.[<div_ $simd_data>](l, r)
                }

                #[inline(always)]    
                fn mul_add(&self, l: $simd_data, m: $simd_data, r: $simd_data) -> $simd_data {
                    self.[<mul_add_ $simd_data>](l, m, r)
                }

                #[inline(always)]
                fn mul_sub(&self, l: $simd_data, m: $simd_data, r: $simd_data) -> $simd_data {
                    self.[<mul_sub_ $simd_data>](l, m, r)
                }

                #[inline(always)]
                fn sub(&self, l: $simd_data, r: $simd_data) -> $simd_data {
                    self.[<sub_ $simd_data>](l, r)
                }

                #[inline(always)]
                fn z1_mul_z2_add_r(&self, z1_re: &mut $simd_data, z1_im: &mut $simd_data, z2_re: $simd_data, z2_im: $simd_data, r: $float_prec) {
                    let tmp = self.mul(*z1_im, z2_im);
                    let tmp_re = self.mul_sub(*z1_re, z2_re, tmp);

                    let tmp = self.mul(*z1_im, z2_re);
                    *z1_im = self.mul_add(*z1_re , z2_im , tmp);
                    *z1_re = self.add(tmp_re, self.splat(r));
                }
            }            
        }
    }
}

impl_simd_data!(f32, f32x8, V3);
impl_simd_data!(f64, f64x4, V3);
impl_simd_data!(f32, f32x16, V4);
impl_simd_data!(f64, f64x8, V4);

pub(crate) fn lorentz_simd<P, S, V>(simd:S, xvec: &[P], x0:P, gamma:P, intensity:P) -> Vec<P> where 
P : Float + FromPrimitive  + VoigtConstants + 'static,
S : SimdData<P, V>,
V: Copy {
    struct Impl<'a, P, S, V> where 
    P : Float + FromPrimitive  + VoigtConstants + 'static,
    S : SimdData<P, V>,
    V : Copy {
        simd: S,
        xvec: &'a [P],
        x0: P,
		gamma: P,
		intensity: P,
        phantom: std::marker::PhantomData<V>
    }
    
    impl<P, S, V> pulp::NullaryFnOnce for Impl<'_, P, S, V> where 
    P : Float + FromPrimitive  + VoigtConstants + 'static,
    S : SimdData<P, V>,
    V: Copy {
        type Output = Vec<P>;

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self { simd, xvec, x0, gamma,  intensity, ..} = self;

            let mut y = vec![P::from_f64(0.0).unwrap(); xvec.len()];
            let c0 = intensity * gamma / P::PI;
            let gamma2 = gamma*gamma;
            let c0splat = simd.splat(intensity * gamma / P::PI);
            let gamma2slpat = simd.splat(gamma * gamma);

            let (x8, x1) = simd.as_simd(xvec);
            let (y8, y1) = simd.as_mut_simd(&mut y);

            let x0vec = simd.splat(x0);

            for (x, y) in iter::zip(x8, y8) {
                let dx = simd.sub(*x, x0vec);
                let dx2= simd.mul(dx, dx);
                *y = simd.div(c0splat, simd.add(gamma2slpat, dx2));
            }
            for (x, y) in iter::zip(x1, y1) {
                let dx2 = (*x-x0) * (*x-x0);
                *y = c0 / (dx2 + gamma2);
            }
            y
        }
    }
    simd.vectorize(Impl { simd, xvec, x0, gamma, intensity, phantom: PhantomData{}  })    
}

pub(crate) fn weideman_simd<P, S, V>(simd:S, xvec: &[P], x0:P, gamma:P, sigma:P, intensity:P, approx: &'static WeidemanParams<P>) -> Vec<P> where 
P : Float + FromPrimitive  + VoigtConstants,
S : SimdData<P, V>,
V: Copy {
    struct Impl<'a, P, S, V> where 
    P : Float + FromPrimitive  + VoigtConstants + 'static,
    S : SimdData<P, V>,
    V : Copy {
        simd: S,
        xvec: &'a [P],
        x0: P,
		gamma: P,
		sigma: P,
		intensity: P,
        approx: &'static  WeidemanParams<P>,
        phantom: std::marker::PhantomData<V>
    }
    
    impl<P, S, V> pulp::NullaryFnOnce for Impl<'_, P, S, V> where 
    P : Float + FromPrimitive  + VoigtConstants,
    S : SimdData<P, V>,
    V: Copy {
        type Output = Vec<P>;

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self { simd, xvec, x0, gamma,  sigma, intensity, approx, ..} = self;
            if sigma == P::zero() {
                return lorentz_simd(simd, xvec, x0, gamma, intensity);
            }

            let mut y = vec![P::from_f64(0.0).unwrap(); xvec.len()];
            
            let (c0, c1, c2, c3, c4, c5) = calc_constants(gamma, sigma, intensity, approx.l);
            let (x8, x1) = simd.as_simd(xvec);
            let (y8, y1) = simd.as_mut_simd(&mut y);

            let x0vec = simd.splat(x0);

            for (x, y) in iter::zip(x8, y8) {
                let dx = simd.sub(*x, x0vec);
                let dx2= simd.mul(dx, dx);
            
                let tmp = simd.mul_add(dx2, simd.splat(P::TWO), simd.splat(c1));
                // let denominator = simd.approx_reciprocal_f32x8(tmp);     //TODO
                let denominator = simd.div(simd.splat(P::ONE),tmp); //TODO
                
                // let z_re = _2*(c4 - dx*dx) * denominator;
                let tmp = simd.sub(simd.splat(c4), dx2);
                let tmp = simd.mul(simd.splat(P::TWO), tmp);                    
                let z_re = simd.mul(tmp, denominator);
                // let z_im = c5*dx * denominator; 
                let tmp = simd.mul(simd.splat(c5), dx);
                let z_im = simd.mul(tmp, denominator);
                
                // eval polynom
                let mut p_re = simd.splat(approx.coef[0]);
                let mut p_im = simd.splat(P::ZERO);
                for w in approx.coef[1..].iter() {
                    simd.z1_mul_z2_add_r(&mut p_re, &mut p_im, z_re, z_im, *w); 
                }
                
                // (L16 + c₀⋅γ - i⋅c₀⋅x)⁻¹
                let t_re = simd.mul(simd.splat(c2), denominator);
                let tmp = simd.mul(simd.splat(c3), dx);
                let t_im = simd.mul(tmp, denominator);

                simd.z1_mul_z2_add_r( 
                    &mut p_re, 
                    &mut p_im, 
                    simd.mul_by_const(t_re, P::TWO), 
                    simd.mul_by_const(t_im, P::TWO), 
                    P::RSQRTPI); 
                
                let tmp = simd.mul(p_im, t_im);
                let tmp = simd.mul_sub(p_re, t_re, tmp);
                *y = simd.mul(simd.splat(c0), tmp);
                
            }
            for (x, y) in iter::zip(x1, y1) {
                *y = eval_weideman(*x-x0, (c0, c1, c2, c3, c4, c5), &approx);
            }
            y
        }
    }
    simd.vectorize(Impl { simd, xvec, x0, gamma,  sigma, intensity, approx , phantom: PhantomData{}  })    
}




