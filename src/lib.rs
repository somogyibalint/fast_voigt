mod const_parameters;
pub use crate::const_parameters::*;

mod scalar;
use crate::scalar::*;

mod simd;
pub use crate::simd::*;

mod test_utils;

// mod bckp_old_impl; // TODO REMOVE
// pub use crate::bckp_old_impl::w16_f32_f32scalar; // TODO REMOVE


pub fn fast_voigt16_s(xvec: &[f32], x0:f32, gamma:f32, sigma:f32, intensity:f32) -> Vec<f32>{
    weideman_scalar(xvec, x0, gamma, sigma, intensity, &WP16S)
}

pub fn fast_voigt16(xvec: &[f64], x0:f64, gamma:f64, sigma:f64, intensity:f64) -> Vec<f64>{
    weideman_scalar(xvec, x0, gamma, sigma, intensity, &WP16)
}

pub fn fast_voigt24(xvec: &[f64], x0:f64, gamma:f64, sigma:f64, intensity:f64) -> Vec<f64>{
    weideman_scalar(xvec, x0, gamma, sigma, intensity, &WP24)
}

pub fn fast_voigt32(xvec: &[f64], x0:f64, gamma:f64, sigma:f64, intensity:f64) -> Vec<f64>{
    weideman_scalar(xvec, x0, gamma, sigma, intensity, &WP32)
}




#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::evaluate_accuracy;

    #[test]
    fn test_scalar_accuarcy() {
        assert!(evaluate_accuracy(fast_voigt16_s, 1E-6));
        assert!(evaluate_accuracy(fast_voigt16, 1E-8));
        assert!(evaluate_accuracy(fast_voigt24, 1E-11));
        assert!(evaluate_accuracy(fast_voigt32, 1E-14));
        
    }
}
