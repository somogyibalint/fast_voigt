use pulp::Simd;
use pulp::x86::V3;
use pulp::{cast, f32x8};
use crate::const_parameters::*;
use crate::calc_constants;
use core::iter;



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
            
            let (c0, c1, c2, c3, c4, c5) = calc_constants(gamma, sigma, intensity, L16);
            let (x8, x1) = pulp::as_arrays::<8, _>(xvec);
            let (y8, y1) = V3::as_mut_simd_f32s(&mut y);

            let x0 = simd.splat_f32x8(x0);
            
            
            for (x, y) in iter::zip(x8, y8) {
                let tmp: f32x8 = cast(*x);
                let dx = simd.sub_f32x8(tmp, x0);
                let dx2= simd.mul_f32x8(dx, dx);
            
                let tmp = simd.mul_add_f32x8(dx2, simd.splat_f32x8(2.0), simd.splat_f32x8(c1));
                let denominator = simd.approx_reciprocal_f32x8(tmp);
                
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
                    z1_mul_z2_add_real(simd, &mut p_re, &mut p_im, z_re, z_im, *w as f32); 
                }
                
                // (L16 + c₀⋅γ - i⋅c₀⋅x)⁻¹
                let t_re = simd.mul_f32x8(simd.splat_f32x8(c2), denominator);
                let tmp = simd.mul_f32x8(simd.splat_f32x8(c3), dx);
                let t_im = simd.mul_f32x8(tmp, denominator);

                z1_mul_z2_add_real(simd, 
                    &mut p_re, 
                    &mut p_im, 
                    mul_c_f32x8(simd, t_re, 2.0), 
                    mul_c_f32x8(simd, t_im, 2.0), 
                    RSQRTPI as f32); 
                
                let tmp = simd.mul_f32x8(p_im, t_im);
                let tmp = simd.mul_sub_f32x8(p_re, t_re, tmp);
                *y = simd.mul_f32x8(simd.splat_f32x8(c0), tmp);
                
            }
            y
        }
    }

    simd.vectorize(Impl { simd, xvec, x0, gamma,  sigma, intensity })    

}


#[inline(always)]
fn z1_mul_z2_add_real(simd: V3, z1_re: &mut f32x8, z1_im: &mut f32x8, z2_re: f32x8, z2_im: f32x8, r: f32) {
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
    use std::ops::AddAssign;
    use num_traits::{Float, FromPrimitive};

    fn linspace<T>(fr: T, to: T, n: u32) -> Vec<T> where 
    T: Float + FromPrimitive + AddAssign {
        let mut x = T::zero();
        let dx = (to - fr) / ( T::from(n - 1).unwrap()  );
        let mut array = Vec::with_capacity(n as usize);
        for _ in 0..n {
            array.push(x);
            x += dx;
        }
        array
    }
 
    #[test]
    fn test_simd() {
        let x = linspace(0.0, 5.0, 1024);
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

            // (127, 2.94541176272260508e-01),
            // (255, 1.26062625829457348e-01),
            // (1023, 6.49746971953819082e-03),
        }

    }

}