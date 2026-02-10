use pulp::Simd;
use pulp::x86::V3;
use pulp::{cast, f32x8};
use num_traits::{Float, FromPrimitive};
use core::iter;
use std::marker::PhantomData;
use crate::const_parameters::*;
use crate::scalar::eval_weideman;
use crate::calc_constants;

pub(crate) trait SimdVec<T, V> : Simd {
    fn splat(&self, x: T) -> V;
    fn as_simd<'a>(&'a self, x: &'a [T]) -> (&'a [V], &'a [T] );
    fn as_mut_simd<'a>(&'a self, x: &'a mut [T]) -> (&'a mut [V], &'a mut [T] );
    fn mul(&self, l: V, r:V) -> V;
    fn mul_by_const(&self, l: V, r:T) -> V;
    fn mul_add(&self, l: V, m:V, r:V) -> V;
    fn mul_sub(&self, l: V, m:V, r:V) -> V;
    fn div(&self, l: V, r:V) -> V;
    fn sub(&self, l: V, r:V) -> V;
    fn z1_mul_z2_add_r(&self, z1_re: &mut V, z1_im: &mut V, z2_re: V, z2_im: V, r: T);
}

impl SimdVec<f32, f32x8> for V3 {
    #[inline(always)]    
    fn as_simd<'a>(&'a self, x: &'a [f32]) -> (&'a [f32x8], &'a [f32] ) {
        V3::as_simd_f32s(x)
    }

    #[inline(always)]    
    fn as_mut_simd<'a>(&'a self, x: &'a mut [f32]) -> (&'a mut [f32x8], &'a mut [f32] ) {
        V3::as_mut_simd_f32s(x)
    }

    #[inline(always)]    
    fn splat(&self, x: f32) -> f32x8 {
        self.splat_f32x8(x)
    }

    #[inline(always)]    
    fn mul(&self, l: f32x8, r: f32x8) -> f32x8 {
        self.mul_f32x8(l, r)
    }

    #[inline(always)]    
    fn mul_by_const(&self, l: f32x8, r: f32) -> f32x8 {
        self.mul_f32x8(l, self.splat(r))
    }

    #[inline(always)]
    fn div(&self, l: f32x8, r: f32x8) -> f32x8 {
        self.div_f32x8(l, r)
    }

    #[inline(always)]    
    fn mul_add(&self, l: f32x8, m: f32x8, r: f32x8) -> f32x8 {
        self.mul_add_f32x8(l, m, r)
    }

    #[inline(always)]
    fn mul_sub(&self, l: f32x8, m: f32x8, r: f32x8) -> f32x8 {
        self.mul_sub_f32x8(l, m, r)
    }

    #[inline(always)]
    fn sub(&self, l: f32x8, r: f32x8) -> f32x8 {
        self.sub_f32x8(l, r)
    }

    #[inline(always)]
    fn z1_mul_z2_add_r(&self, z1_re: &mut f32x8, z1_im: &mut f32x8, z2_re: f32x8, z2_im: f32x8, r: f32) {
        let tmp = self.mul(*z1_im, z2_im);
        let tmp_re = self.mul_sub(*z1_re, z2_re, tmp);

        let tmp = self.mul_f32x8(*z1_im, z2_re);
        *z1_im = self.mul_add_f32x8(*z1_re , z2_im , tmp);
        *z1_re = self.add_f32x8(tmp_re, self.splat(r));
    }
}

// simd:
// V3 / V4
// f32x8, f32x16, f64x4, f64x8
// 16, 24, 32
//   as_arrays/as_mut_simd_f32s, 
//   splat
//   cast
//   sub, mul, mul_add, div,  
pub(crate) fn weideman_simd<P, S, V>(simd:S, xvec: &[P], x0:P, gamma:P, sigma:P, intensity:P, approx: &'static WeidemanParams<P>) -> Vec<P> where 
P : Float + FromPrimitive  + VoigtConstants,
S : SimdVec<P, V>,
V: Copy {


    struct Impl<'a, P, S, V> where 
    P : Float + FromPrimitive  + VoigtConstants + 'static,
    S : SimdVec<P, V>,
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
    S : SimdVec<P, V>,
    V: Copy {
        type Output = Vec<P>;
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

// #[inline(always)]
// fn z1_mul_z2_add_real(simd: V3, z1_re: &mut f32x8, z1_im: &mut f32x8, z2_re: f32x8, z2_im: f32x8, r: f32) {
//         let tmp = simd.mul_f32x8(*z1_im, z2_im);
//         let tmp_re = simd.mul_sub_f32x8(*z1_re, z2_re, tmp);

//         let tmp = simd.mul_f32x8(*z1_im, z2_re);
//         *z1_im = simd.mul_add_f32x8(*z1_re , z2_im , tmp);
//         *z1_re = simd.add_f32x8(tmp_re, simd.splat_f32x8(r));
// }

fn mul_by_num<P, S, V>(simd:S, xvec: &[P], multiplier:P) -> Vec<P> where 
P : Float + FromPrimitive  + VoigtConstants,
S : SimdVec<P, V>, 
V: Copy {

    struct Impl<'a, P, S, V> where 
    P : Float + FromPrimitive  + VoigtConstants,
    S : SimdVec<P, V>,
    V: Copy
    {
        simd: S,
        xvec: &'a [P],
        multiplier: P,
        phantom: std::marker::PhantomData<V>
    }

    impl<P, S, V> pulp::NullaryFnOnce for Impl<'_, P, S, V> where 
    P : Float + FromPrimitive  + VoigtConstants,
    S : SimdVec<P, V>,
    V: Copy {
        type Output = Vec<P>;

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self { simd, xvec, multiplier, phantom: _} = self;
            
            let c = simd.splat(multiplier);

            let mut y = vec![P::from_f64(0.0).unwrap(); xvec.len()];
            let (x8, x1) = simd.as_simd(xvec);
            let (y8, y1) = simd.as_mut_simd(&mut y);
            for (x, y) in iter::zip(x8, y8) {
                *y = simd.mul(*x, c);
            }     
            y
        }
    }
    simd.vectorize(Impl { simd, xvec, multiplier, phantom: PhantomData{} })    
}

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
            
            let (c0, c1, c2, c3, c4, c5) = calc_constants(gamma, sigma, intensity, L16s);
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

    #[test]
    fn compare_to_scalar_w16f32() {
        use crate::test_utils::linspace;
        use crate::w16_f32_f32scalar;

        let x = linspace(0.0, 10.0, 101);
        let x0 = 0.0;
        let gamma = 0.5;
        let sigma = 0.5;
        let intensity = 1.0;
        
        let simd= V3::try_new().unwrap();
        let ys = w16_f32_f32scalar(&x, x0, gamma, sigma, intensity);
        let yv = weideman16_avx2_f32(simd, &x, x0, gamma, sigma, intensity);
        for (s, v) in ys.iter().zip(yv.iter()) {
            println!("{}  {}  {}", s, v, s-v);
        }
    }


    #[test]
    fn test_work() {
        use crate::test_utils::linspace;

        let x = linspace(0.0, 10.0, 2001);
        let x0 = 0.0;
        let gamma = 0.5;
        let sigma = 0.5;
        let intensity = 1.0;

        if let Some(simd) = V3::try_new() {
            let y = weideman16_avx2_f32(simd, &x, x0, gamma, sigma, intensity); 
            println!("{}", y[0]);
            println!("{}", y[1]);
            println!("{}", y[7]);
            println!("{}", y[63]);
            println!("{}", y[127]);
            println!("{}", y[255]);
            println!("{}", y[1023]);
            println!("{}", y[2000]);

        // (0, 4.17418561040735436e-01),
        // (1, 4.17408650320942820e-01),
        // (7, 4.16933282958152407e-01),
        // (63, 3.80311100325604223e-01),
        // (127, 2.90046558879590188e-01),
        // (255, 1.20770262633453737e-01),
        // (511, 2.64831190930530785e-02),
        // (1023, 6.20284807080986877e-03),
        // (2000, 1.59956736012200696e-03)
        }
    }

}
