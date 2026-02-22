mod const_parameters;
use cfg_if::cfg_if;
pub use crate::const_parameters::*;

mod scalar;
use crate::scalar::*;

mod simd;
#[cfg(any(feature = "avx2", feature = "avx512"))]
pub(crate) use crate::simd::*;

mod test_utils;

// #[cfg(feature = "py_bindings")]
mod py_bindings;
 


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


cfg_if! {
    if #[cfg(feature = "avx2")] {
        use pulp::x86::V3;

        pub fn fast_voigt16s_avx2(xvec: &[f32], x0:f32, gamma:f32, sigma:f32, intensity:f32) -> Vec<f32>{
            let simd = V3::try_new().unwrap();
            weideman_simd(simd, xvec, x0, gamma, sigma, intensity, &WP16S)
        }

        pub fn fast_voigt16d_avx2(xvec: &[f64], x0:f64, gamma:f64, sigma:f64, intensity:f64) -> Vec<f64>{
            let simd = V3::try_new().unwrap();
            weideman_simd(simd, xvec, x0, gamma, sigma, intensity, &WP16)
        }

        pub fn fast_voigt24_avx2(xvec: &[f64], x0:f64, gamma:f64, sigma:f64, intensity:f64) -> Vec<f64>{
            let simd = V3::try_new().unwrap();
            weideman_simd(simd, xvec, x0, gamma, sigma, intensity, &WP24)
        }

        pub fn fast_voigt32_avx2(xvec: &[f64], x0:f64, gamma:f64, sigma:f64, intensity:f64) -> Vec<f64>{
            let simd = V3::try_new().unwrap();
            weideman_simd(simd, xvec, x0, gamma, sigma, intensity, &WP32)
        }
    } 
}

cfg_if! {
    if #[cfg(feature = "avx512")] {
        use pulp::x86::V4;

        pub fn fast_voigt16s_avx512(xvec: &[f32], x0:f32, gamma:f32, sigma:f32, intensity:f32) -> Vec<f32>{
            let simd = V4::try_new().unwrap();
            weideman_simd(simd, xvec, x0, gamma, sigma, intensity, &WP16S)
        }

        pub fn fast_voigt16d_avx512(xvec: &[f64], x0:f64, gamma:f64, sigma:f64, intensity:f64) -> Vec<f64>{
            let simd = V4::try_new().unwrap();
            weideman_simd(simd, xvec, x0, gamma, sigma, intensity, &WP16)
        }

        pub fn fast_voigt24_avx512(xvec: &[f64], x0:f64, gamma:f64, sigma:f64, intensity:f64) -> Vec<f64>{
            let simd = V4::try_new().unwrap();
            weideman_simd(simd, xvec, x0, gamma, sigma, intensity, &WP24)
        }

        pub fn fast_voigt32_avx512(xvec: &[f64], x0:f64, gamma:f64, sigma:f64, intensity:f64) -> Vec<f64>{
            let simd = V4::try_new().unwrap();
            weideman_simd(simd, xvec, x0, gamma, sigma, intensity, &WP32)
        }
    } 
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_accuracy;

    #[test]
    fn test_scalar_accuarcy() {
        assert_accuracy(fast_voigt16_s, 3E-7);
        assert_accuracy(fast_voigt16, 6E-8);
        assert_accuracy(fast_voigt24, 4E-11);
        assert_accuracy(fast_voigt32, 2E-14);
    }

    #[test]
    #[cfg(feature = "avx2")]
    fn test_avx2_accuarcy() {
        assert_accuracy(fast_voigt16s_avx2, 3E-7);
        assert_accuracy(fast_voigt16d_avx2, 6E-8);
        assert_accuracy(fast_voigt24_avx2, 4E-11);
        assert_accuracy(fast_voigt32_avx2, 2E-14);
    }

    #[test]
    #[cfg(feature = "avx512")]
    fn test_avx512_accuarcy() {
        assert_accuracy(fast_voigt16s_avx512, 3E-7);
        assert_accuracy(fast_voigt16d_avx512, 6E-8);
        assert_accuracy(fast_voigt24_avx512, 4E-11);
        assert_accuracy(fast_voigt32_avx512, 2E-14);
    }
}
