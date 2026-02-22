use pyo3;

#[pyo3::pymodule]
mod fast_voigt {
    use numpy::{PyArray1, PyReadonlyArray1};
    use pyo3::{pyfunction, Python, Bound};
    use cfg_if::cfg_if;

    use crate::const_parameters::{WP16S, WP16, WP24, WP32};
    use crate::scalar::weideman_scalar;

    #[cfg(any(feature = "avx2", feature = "avx512"))]
    use crate::simd::weideman_simd;

    #[pyfunction(name = "fast_voigt16_single")]
    fn fast_voigt16s_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f32>,
        x0: f32,
        gamma: f32,
        sigma: f32,
        intensity: f32,
    ) -> Bound<'py, PyArray1<f32>> {
        let x = x.as_slice().unwrap();
        cfg_if! {
            if #[cfg(feature = "avx512")] {
                use pulp::x86::V4;

                let simd = V4::try_new().unwrap();
                PyArray1::from_vec(py,  weideman_simd(simd, x, x0, gamma, sigma, intensity, &WP16S))
            } else if #[cfg(feature = "avx2")] {
                use pulp::x86::V3;

                let simd = V3::try_new().unwrap();
                PyArray1::from_vec(py,  weideman_simd(simd, x, x0, gamma, sigma, intensity, &WP16S))
            } else {
                PyArray1::from_vec(py,  weideman_scalar(x, x0, gamma, sigma, intensity, &WP16S))
            }
        }
    }

    #[pyfunction(name = "fast_voigt16")]
    fn fast_voigt16_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        x0: f64,
        gamma: f64,
        sigma: f64,
        intensity: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let x = x.as_slice().unwrap();
        cfg_if! {
            if #[cfg(feature = "avx512")] {
                use pulp::x86::V4;

                let simd = V4::try_new().unwrap();
                PyArray1::from_vec(py,  weideman_simd(simd, x, x0, gamma, sigma, intensity, &WP16))
            } else if #[cfg(feature = "avx2")] {
                use pulp::x86::V3;

                let simd = V3::try_new().unwrap();
                PyArray1::from_vec(py,  weideman_simd(simd, x, x0, gamma, sigma, intensity, &WP16))
            } else {
                PyArray1::from_vec(py,  weideman_scalar(x, x0, gamma, sigma, intensity, &WP16))
            }
        }
    }

    #[pyfunction(name = "fast_voigt24")]
    fn fast_voigt24_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        x0: f64,
        gamma: f64,
        sigma: f64,
        intensity: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let x = x.as_slice().unwrap();
        cfg_if! {
            if #[cfg(feature = "avx512")] {
                use pulp::x86::V4;

                let simd = V4::try_new().unwrap();
                PyArray1::from_vec(py,  weideman_simd(simd, x, x0, gamma, sigma, intensity, &WP24))
            } else if #[cfg(feature = "avx2")] {
                use pulp::x86::V3;

                let simd = V3::try_new().unwrap();
                PyArray1::from_vec(py,  weideman_simd(simd, x, x0, gamma, sigma, intensity, &WP24))
            } else {
                PyArray1::from_vec(py,  weideman_scalar(x, x0, gamma, sigma, intensity, &WP24))
            }
        }
    }

    #[pyfunction(name = "fast_voigt32")]
    fn fast_voigt32_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        x0: f64,
        gamma: f64,
        sigma: f64,
        intensity: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let x = x.as_slice().unwrap();
        cfg_if! {
            if #[cfg(feature = "avx512")] {
                use pulp::x86::V4;

                let simd = V4::try_new().unwrap();
                PyArray1::from_vec(py,  weideman_simd(simd, x, x0, gamma, sigma, intensity, &WP32))
            } else if #[cfg(feature = "avx2")] {
                use pulp::x86::V3;

                let simd = V3::try_new().unwrap();
                PyArray1::from_vec(py,  weideman_simd(simd, x, x0, gamma, sigma, intensity, &WP32))
            } else {
                PyArray1::from_vec(py,  weideman_scalar(x, x0, gamma, sigma, intensity, &WP32))
            }
        }
    }

}



