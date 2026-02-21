use pulp::Simd;
use pulp::x86::{V3, V4};
use pulp::{cast, f32x8, f64x4, f32x16, f64x8};
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




// TODO DELETE: Non-generic implementation, left here for testing and reference
pub fn weideman16_avx2_f32(simd: V3, xvec: &[f32], x0:f32, gamma:f32, sigma:f32, intensity:f32) -> Vec<f32> {
    struct Impl<'a> {
        simd: V3,
        xvec: &'a [f32],
        x0: f32,
		gamma: f32,
		sigma: f32,
		intensity: f32,
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = Vec<f32>;

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self { simd, xvec, x0, gamma,  sigma, intensity} = self;
            // let mut y = Vec::with_capacity(xvec.len());
            let mut y = vec![0.0f32; xvec.len()];
            
            let (c0, c1, c2, c3, c4, c5) = calc_constants(gamma, sigma, intensity, L16S);
            let (x8, x1) = pulp::as_arrays::<8, _>(xvec);
            let (y8, y1) = V3::as_mut_simd_f32s(&mut y);

            let x0vec = simd.splat_f32x8(x0);
            
            
            for (x, y) in iter::zip(x8, y8) {
                let tmp: f32x8 = cast(*x);
                let dx = simd.sub_f32x8(tmp, x0vec);
                let dx2= simd.mul_f32x8(dx, dx);
            
                let tmp = simd.mul_add_f32x8(dx2, simd.splat_f32x8(2.0), simd.splat_f32x8(c1));
                // let denominator = simd.approx_reciprocal_f32x8(tmp);     //TODO
                let denominator = simd.div_f32x8(simd.splat_f32x8(1.0),tmp); //TODO
                
                // let z_re = _2*(c4 - dx*dx) * denominator;
                let tmp = simd.sub_f32x8(simd.splat_f32x8(c4), dx2);
                let tmp = simd.mul_f32x8(simd.splat_f32x8(2.0), tmp);                    
                let z_re = simd.mul_f32x8(tmp, denominator);
                // let z_im = c5*dx * denominator; 
                let tmp = simd.mul_f32x8(simd.splat_f32x8(c5), dx);
                let z_im = simd.mul_f32x8(tmp, denominator);
                
                // eval polynom
                let mut p_re = simd.splat_f32x8(W16S[0]);
                let mut p_im = simd.splat_f32x8(0.0);
                for w in W16[1..].iter() {
                    z1_mul_z2_add_real_f32x8(simd, &mut p_re, &mut p_im, z_re, z_im, *w as f32); 
                }
                
                // (L16 + c₀⋅γ - i⋅c₀⋅x)⁻¹
                let t_re = simd.mul_f32x8(simd.splat_f32x8(c2), denominator);
                let tmp = simd.mul_f32x8(simd.splat_f32x8(c3), dx);
                let t_im = simd.mul_f32x8(tmp, denominator);

                z1_mul_z2_add_real_f32x8(simd, 
                    &mut p_re, 
                    &mut p_im, 
                    mul_c_f32x8(simd, t_re, 2.0), 
                    mul_c_f32x8(simd, t_im, 2.0), 
                    RSQRTPI as f32); 
                
                let tmp = simd.mul_f32x8(p_im, t_im);
                let tmp = simd.mul_sub_f32x8(p_re, t_re, tmp);
                *y = simd.mul_f32x8(simd.splat_f32x8(c0), tmp);
                
            }
            for (x, y) in iter::zip(x1, y1) {
                *y = eval_weideman(*x-x0, (c0, c1, c2, c3, c4, c5), &WP16S);
            }
            y
        }
    }

    simd.vectorize(Impl { simd, xvec, x0, gamma,  sigma, intensity })    

}


#[inline(always)]
fn z1_mul_z2_add_real_f32x8(simd: V3, z1_re: &mut f32x8, z1_im: &mut f32x8, z2_re: f32x8, z2_im: f32x8, r: f32) {
        let tmp = simd.mul_f32x8(*z1_im, z2_im);
        let tmp_re = simd.mul_sub_f32x8(*z1_re, z2_re, tmp);

        let tmp = simd.mul_f32x8(*z1_im, z2_re);
        *z1_im = simd.mul_add_f32x8(*z1_re , z2_im , tmp);
        *z1_re = simd.add_f32x8(tmp_re, simd.splat_f32x8(r));
}

#[inline(always)]
fn mul_c_f32x8(simd: V3, x: f32x8, c: f32) -> f32x8 {
    simd.mul_f32x8(x, simd.splat_f32x8(c))
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_accuarcy() {
        use crate::test_utils::assert_accuracy;
        
        fn test_w16_avx2_f32(x: &[f32], x0: f32, gamma: f32, sigma: f32, intensity: f32) -> Vec<f32> {
            let simd= V3::try_new().unwrap();
             weideman16_avx2_f32(simd, &x, x0, gamma, sigma, intensity)
        }
        // assert!(evaluate_accuracy(test_w16_avx2_f32, 1E-6));
        assert_accuracy(test_w16_avx2_f32, 3E-7);
    }

    // #[test]
    // fn compare_to_scalar_w16f32() {
    //     use crate::test_utils::linspace;
    //     use crate::w16_f32_f32scalar;

    //     let x = linspace(0.0, 10.0, 101);
    //     let x0 = 0.0;
    //     let gamma = 0.5;
    //     let sigma = 0.5;
    //     let intensity = 1.0;
        
    //     let simd= V3::try_new().unwrap();
    //     let ys = w16_f32_f32scalar(&x, x0, gamma, sigma, intensity);
    //     let yv = weideman16_avx2_f32(simd, &x, x0, gamma, sigma, intensity);
    //     for (s, v) in ys.iter().zip(yv.iter()) {
    //         println!("{}  {}  {}", s, v, s-v);
    //     }
    // }


}
