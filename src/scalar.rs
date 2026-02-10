use crate::const_parameters::*;
use num_traits::{Float, FromPrimitive};


// Voigt profile: f(x, x₀, γ, σ, A) = A⋅w(z) / (σ √π) where
//     x: independent variable
//     x₀: center parameter
//     γ: Lorentz variance parameter
//     σ: Gauss variance parameter
//     A: area under curve
//     w(z) Faddeeva function evaluated at z = (x + i⋅γ ) / (√2 σ)
// Weidemann's approximation for the Faddeeva function w(z):
//   let 
//     l± ≡ Ln ± i⋅z
//   introduce t and Z 
//     t =   1 / l-
//     Z =  l+ / l-  =  l+ ⋅ t 
//   evaluate the polynom:
//     p = c0 + (c1 + (c2 + ( ...cn*Z)*Z)*Z)*Z)...)*Z)*Z)
//   finally
//     w(z) = (c + 2⋅c⋅p)⋅p where c = 1 / √π
//  
// There are 3 version with n=16, 24 and 32 (inccreasing accuracy)
//
// Expanding complex variables into reals
//     (a+bi)(c+di) = ac - bd + (ad + bc)i
//     1 / (a + bi) = (a - bi) / (a² + b²)
//
//     z = (dx + i γ ) / (√2 σ)
//     iz = (-γ + i⋅x) / (√2 σ) = -c₀⋅γ  +  i⋅c₀⋅x
//     t = 1 / (Ln - iz) = 1 / (Ln + c₀⋅γ - i⋅c₀⋅x) = (Ln + c₀⋅γ + i⋅c₀⋅x) / ((Ln + c₀⋅γ)²  + (c₀⋅x))²)
//     z = (Ln + iz) * t = (Ln - c₀⋅γ + i⋅c₀⋅x) (Ln + c₀⋅γ + i⋅c₀⋅x) / ((Ln + c₀⋅γ)²  + (c₀⋅x))²)
//     where c₀ := 1/(√2 σ)
//
//     Re(1/t) = 2⋅σ⋅(2⋅Ln⋅σ + √2⋅γ)  / (2⋅x² + (2⋅Ln*σ + √2⋅γ)²)
//     Im(1/t) = 2⋅√2⋅x⋅σ            / (2⋅x² + (2⋅Ln*σ + √2⋅γ)²)
//     Re(z) = 2(2⋅Ln²σ² - x² - γ²) /  (2⋅x² + (2⋅Ln*σ + √2⋅γ)²)
//     Im(z) = (4⋅√2⋅Ln⋅x⋅σ)    / (2⋅x² + (2⋅Ln*σ + √2⋅γ)²)


pub(crate) fn weideman_scalar<P>(xvec: &[P], x0:P, gamma:P, sigma:P, intensity:P, approx: &WeidemanParams<P>) -> Vec<P> where 
P : Float + FromPrimitive  + VoigtConstants , // f32 or f64
{   
    let mut y = Vec::with_capacity(xvec.len());
    let coef = calc_constants(gamma, sigma, intensity, approx.l);
    for x in xvec.iter() {        
        y.push(eval_weideman(*x - x0, coef, &approx));
    }
    y
    
}


#[inline(always)]
pub(crate) fn eval_weideman<P>(dx: P, c: (P,P,P,P,P,P), approx: &WeidemanParams<P>) -> P where 
P : Float + FromPrimitive + VoigtConstants, {
        let denominator: P = P::ONE / (P::TWO*dx*dx + c.1);

        // z = (L16 - c₀⋅γ + i⋅c₀⋅x)(L16 + c₀⋅γ - i⋅c₀⋅x)⁻¹
        let z_re = P::TWO*(c.4 - dx*dx) * denominator;
        let z_im = c.5*dx * denominator;

        // eval the polynomial
        let mut p_re = approx.coef[0];
        let mut p_im: P = P::ZERO;
        for w in approx.coef[1..].iter() {
            z1_mul_z2_add_real(&mut p_re, &mut p_im, z_re, z_im, *w); 
        }

        // t = (L16 + c₀⋅γ - i⋅c₀⋅x)⁻¹
        let t_re = c.2 * denominator;
        let t_im = c.3*dx * denominator;

        z1_mul_z2_add_real(&mut p_re, &mut p_im, P::TWO*t_re, P::TWO*t_im, P::RSQRTPI); 

        c.0 * (p_re*t_re - p_im*t_im)
}




// Evaluate z₁ = z₁⋅z₂ + r = 
//    1⋅(Re(z₁)Re(z₂) - Im(z₁)Im(z₂) + r
//  + i⋅(Re(z₁)Im(z₂) + Im(z₁)Re(z₂))
#[inline(always)]
pub(crate) fn z1_mul_z2_add_real<T>(z1_re: &mut T, z1_im: &mut T, z2_re: T, z2_im: T, x: T) 
where T: Float {
        let tmp = *z1_re * z2_re - *z1_im * z2_im;
        *z1_im = *z1_re * z2_im + *z1_im*z2_re;
        *z1_re = tmp  + x;
}


pub(crate) fn calc_constants<P>(gamma:P, sigma:P, intensity:P, l:P) -> (P, P, P, P, P, P) where 
P : Float + FromPrimitive + VoigtConstants{

    let c0 = intensity / (P::SQRT2PI * sigma);
    let c1 = P::TWO*l*sigma + P::SQRT2*gamma;
    let c1 = c1*c1;
    let c2 = P::TWO*sigma*(P::TWO*l*sigma + P::SQRT2*gamma);
    let c3 = P::TWO*P::SQRT2*sigma;
    let c4 = P::TWO*l*l*sigma*sigma - gamma*gamma;
    let c5 = P::FOUR*P::SQRT2*l*sigma;

    (c0, c1, c2, c3, c4, c5)
}

