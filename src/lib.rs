

mod const_parameters;
#[allow(unused_imports)]
pub use crate::const_parameters::*;


mod scalar;
pub use crate::scalar::*;

mod simd;
pub use crate::simd::*;


pub fn fast_voigt16_s(xvec: &[f32], x0:f32, gamma:f32, sigma:f32, intensity:f32) -> Vec<f32>{
    weideman_scalar(xvec, x0, gamma, sigma, intensity, WP16)
}

pub fn fast_voigt16(xvec: &[f64], x0:f64, gamma:f64, sigma:f64, intensity:f64) -> Vec<f64>{
    weideman_scalar(xvec, x0, gamma, sigma, intensity, WP16)
}

pub fn fast_voigt24(xvec: &[f64], x0:f64, gamma:f64, sigma:f64, intensity:f64) -> Vec<f64>{
    weideman_scalar(xvec, x0, gamma, sigma, intensity, WP24)
}

pub fn fast_voigt32(xvec: &[f64], x0:f64, gamma:f64, sigma:f64, intensity:f64) -> Vec<f64>{
    weideman_scalar(xvec, x0, gamma, sigma, intensity, WP32)
}




#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::AddAssign;
    use num_traits::{Float, Signed, abs, FromPrimitive, AsPrimitive};
    use std::collections::HashMap;

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

    fn approx_eq<P>(l: P, r:P, prec:P) -> bool where
    P : Float + Signed {
        abs::<P>(l-r) < prec
    }



    fn evaluate_scalar<T>(approx: fn(&[T], T, T, T, T)->Vec<T>, precision: f64) -> bool where 
    T: Float + FromPrimitive + AsPrimitive<f64> + AddAssign {
        let x = linspace(T::zero(), T::from_f64(5.0).unwrap(), 1024);
        let x0 = T::from_f64(0.0).unwrap();
        let gamma = T::from_f64(0.5).unwrap();
        let sigma = T::from_f64(0.5).unwrap();
        let intensity = T::from_f64(1.0).unwrap();
        
        // accurate (up to double prec) values calculated utilizing scipy.special.wofz()
        let accurate = HashMap::from([
            (0, 4.17418561040735436e-01),
            (1, 4.17409090948306805e-01),
            (7, 4.16954843520911000e-01),
            (63, 3.81870067048370398e-01),
            (127, 2.94541176272260508e-01),
            (255, 1.26062625829457348e-01),
            (1023, 6.49746971953819082e-03),
        ]);

        let y = approx(&x, x0, gamma, sigma, intensity);
        for (idx, accurate_value) in accurate {
            let approx_value = y[idx].as_();
            if !approx_eq(approx_value, accurate_value, precision) {
                return false;
            }
        }
        true
    }


    #[test]
    fn test_scalar_accuarcy() {
        assert!(evaluate_scalar(fast_voigt16_s, 1E-6));
        assert!(evaluate_scalar(fast_voigt16, 1E-8));
        assert!(evaluate_scalar(fast_voigt24, 1E-11));
        assert!(evaluate_scalar(fast_voigt32, 1E-15));
        
    }
}
