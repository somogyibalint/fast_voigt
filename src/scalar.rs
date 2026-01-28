use crate::const_parameters::*;
use num_traits::{Float, FromPrimitive};
use std::{f64::consts::SQRT_2};
// Voigt profile: f(x, x₀, γ, σ, A) = A⋅w(z) / (σ √π) where
//     x: independent variable
//     x₀: center parameter
//     γ: Lorentz variance parameter
//     σ: Gauss variance parameter
//     A: amplitude
//     w(z) Faddaeva function evaluated at (x + i⋅γ ) / (√2 σ)
// Weidemann's approximation for the Faddaeva function w(z):
//   let 
//     l± ≡ Ln ± i⋅z
//   introduce t and Z 
//     t = 1 / l-
//     Z = l+ / l- = l+ ⋅ t 
//   evaluate the polynom:
//     p = c0 + (c1 + (c2 + ( ...cn*Z)*Z)*Z)*Z)...)*Z)*Z)
//   w(z) = (c+ 2⋅c⋅p)⋅p  where c = 1 / √π
//  
// There are 3 version with n=16, 24 and 32
//
// Expanding complex variables into reals
//     (a+bi)(c+di) = ac - bd + (ad + bc)i
//     1 / (a + bi) = (a - bi) / (a² + b²)
//
//     z = (dx + i γ ) / (√2 σ)
//     iz = (-γ + i⋅x) / (√2 σ) = -c₀⋅γ  +  i⋅c₀⋅x
//     t = 1 / (L16 - iz) = 1 / (L16 + c₀⋅γ - i⋅c₀⋅x) = (L16 + c₀⋅γ + i⋅c₀⋅x) / ((L16 + c₀⋅γ)²  + (c₀⋅x))²)
//     z = (L16 + iz) * t = (L16 - c₀⋅γ + i⋅c₀⋅x) (L16 + c₀⋅γ + i⋅c₀⋅x) / ((L16 + c₀⋅γ)²  + (c₀⋅x))²)
//
//     Re(1/t) = 2⋅σ⋅(2⋅L16⋅σ + sqrt(2)⋅γ)  / (2⋅x² + (2⋅L16*σ + sqrt(2)⋅γ)²)
//     Im(1/t) = 2⋅sqrt(2)⋅x⋅σ             / (2⋅x² + (2⋅L16*σ + sqrt(2)⋅γ)²)
//     Re(z) = 2(2⋅L16²σ² - x² - γ²) /  (2⋅x² + (2⋅L16*σ + sqrt(2)⋅γ)²)
//     Im(z) = (4⋅sqrt(2)⋅L16⋅x⋅σ)    / (2⋅x² + (2⋅L16*σ + sqrt(2)⋅γ)²)



pub fn weideman_scalar<P>(xvec: &[P], x0:P, gamma:P, sigma:P, intensity:P, approx: WeidemanParams) -> Vec<P> where 
P : Float + FromPrimitive, // output precision
{   
    let mut y = Vec::with_capacity(xvec.len());

    let rsqrtpi = P::from_f64(RSQRTPI).unwrap();
    
    // let c0 = (intensity / (sqrt2pi * sigma)) as P;
    // let mut c1 = _2*l*sigma + sqrt2*gamma;
    // c1 = c1*c1;
    // let c1 = c1 as P;
    // let c2 = (_2*sigma*(_2*l*sigma + sqrt2*gamma)) as P;
    // let c3 = (_2*sqrt2*sigma) as P;
    // let c4 = (_2*l*l*sigma*sigma - gamma*gamma) as P;
    // let c5 = (_4*sqrt2*l*sigma) as P;
    
    let (c0, c1, c2, c3, c4, c5) = calc_constants(gamma, sigma, intensity, approx.l);
    let _2 = P::from_f64(2.0).unwrap();

    for x in xvec.iter() {
        let dx = (*x - x0) as P;
        
        let denominator: P = P::one() / (_2*dx*dx + c1);

        // z = (L16 - c₀⋅γ + i⋅c₀⋅x)(L16 + c₀⋅γ - i⋅c₀⋅x)⁻¹
        let z_re = _2*(c4 - dx*dx) * denominator;
        let z_im = c5*dx * denominator;

        // eval the polynom
        let mut p_re = P::from_f64(approx.coef[0]).unwrap();
        let mut p_im: P = P::zero();
        for w in approx.coef[1..].iter() {
            z1_mul_z2_add_real(&mut p_re, &mut p_im, z_re, z_im, P::from_f64(*w).unwrap()); 
        }

        // t = (L16 + c₀⋅γ - i⋅c₀⋅x)⁻¹
        let t_re = c2 * denominator;
        let t_im = c3*dx * denominator;

        z1_mul_z2_add_real(&mut p_re, &mut p_im, _2*t_re, _2*t_im, rsqrtpi); 

        let res = c0 * (p_re*t_re - p_im*t_im) ;
        y.push(res);
    }
    y
    
}


// non-generic f32 implementation for benchmarks
pub fn w16_f32_f32scalar(xvec: &[f32], x0:f32, gamma:f32, sigma:f32, intensity:f32) -> Vec<f64> {
    let sqrt2pi = SQRT2PI as f32;
    let l16 = L16 as f32;
    let sqrt2 = SQRT_2 as f32;
    let mut y = Vec::with_capacity(xvec.len());

    let c0 = (intensity / (sqrt2pi * sigma)) as f32;
    let mut c1 = 2.0*l16*sigma + sqrt2*gamma;
    c1 = c1*c1;
    let c1 = c1 as f32;
    let c2 = (2.0*sigma*(2.0*l16*sigma + sqrt2*gamma)) as f32;
    let c3 = (2.0*sqrt2*sigma) as f32;
    let c4 = (2.0*l16*l16*sigma*sigma - gamma*gamma) as f32 ;
    let c5 = (4.0*sqrt2*l16*sigma) as f32;
    
    for x in xvec.iter() {
        let dx = (x - x0) as f32;
        
        let denominator = 1.0 / (2.0*dx*dx + c1);

        // (L16 - c₀⋅γ + i⋅c₀⋅x)
        let z_re = 2.0*(c4 - dx*dx) * denominator;
        let z_im = c5*dx * denominator;

        // eval the polynom
        let mut p_re = W16[0] as f32;
        let mut p_im = 0.0;
        for w in W16[1..].iter() {
            z1_mul_z2_add_real(&mut p_re, &mut p_im, z_re, z_im, *w as f32); 
        }

        // (L16 + c₀⋅γ - i⋅c₀⋅x)⁻¹
        let t_re = c2 * denominator;
        let t_im = c3*dx * denominator;

        z1_mul_z2_add_real(&mut p_re, &mut p_im, 2.0*t_re, 2.0*t_im, RSQRTPI as f32); 

        let res = c0 * (p_re*t_re - p_im*t_im) ;
        y.push(res as f64);
    }
    y
    
}


// Evaluate z₁ = z₁⋅z₂ + r = 
//    1⋅(Re(z₁)Re(z₂) - Im(z₁)Im(z₂) + r
//  + i⋅(Re(z₁)Im(z₂) + Im(z₁)Re(z₂))
#[inline(always)]
fn z1_mul_z2_add_real<T>(z1_re: &mut T, z1_im: &mut T, z2_re: T, z2_im: T, x: T) 
where T: Float {
        let tmp = *z1_re * z2_re - *z1_im * z2_im;
        *z1_im = *z1_re * z2_im + *z1_im*z2_re;
        *z1_re = tmp  + x;
}

#[inline(always)]
pub(crate) fn calc_constants<P>(gamma:P, sigma:P, intensity:P, l:f64) -> (P, P, P, P, P, P) where 
P : Float + FromPrimitive {
    let l = P::from_f64(l).unwrap();
    let sqrt2pi = P::from_f64(SQRT2PI).unwrap();
    let sqrt2 = P::from_f64(SQRT_2).unwrap();
    let _2 = P::from_f64(2.0).unwrap();
    let _4 = P::from_f64(4.0).unwrap();

    let c0 = (intensity / (sqrt2pi * sigma)) as P;
    let mut c1 = _2*l*sigma + sqrt2*gamma;
    c1 = c1*c1;
    let c1 = c1 as P;
    let c2 = (_2*sigma*(_2*l*sigma + sqrt2*gamma)) as P;
    let c3 = (_2*sqrt2*sigma) as P;
    let c4 = (_2*l*l*sigma*sigma - gamma*gamma) as P;
    let c5 = (_4*sqrt2*l*sigma) as P;

    (c0, c1, c2, c3, c4, c5)
}

