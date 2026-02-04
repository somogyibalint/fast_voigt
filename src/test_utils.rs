#![cfg(test)]
use std::ops::AddAssign;
use num_traits::{Float, Signed, abs, FromPrimitive, AsPrimitive};
use std::collections::HashMap;


pub(crate) fn linspace<T>(fr: T, to: T, n: u32) -> Vec<T> where 
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


fn approx_eq<P>(l: P, r:P, norm:P, prec:P) -> bool where
P : Float + Signed {
    abs::<P>(l-r) / norm < prec
}


pub(crate) fn evaluate_accuracy<T>(approx: fn(&[T], T, T, T, T)->Vec<T>, precision: f64) -> bool where 
T: Float + FromPrimitive + AsPrimitive<f64> + AddAssign  {
    let x = linspace(T::zero(), T::from_f64(10.0).unwrap(), 2001);
    let x0 = T::from_f64(0.0).unwrap();
    let gamma = T::from_f64(0.5).unwrap();
    let sigma = T::from_f64(0.5).unwrap();
    let intensity = T::from_f64(1.0).unwrap();
    
    // Accurate (up to f64 machine precision) values calculated utilizing scipy.special.wofz()
    let accurate = HashMap::from([
        (0, 4.17418561040735436e-01),
        (1, 4.17408650320942820e-01),
        (7, 4.16933282958152407e-01),
        (63, 3.80311100325604223e-01),
        (127, 2.90046558879590188e-01),
        (255, 1.20770262633453737e-01),
        (511, 2.64831190930530785e-02),
        (1023, 6.20284807080986877e-03),
        (2000, 1.59956736012200696e-03)
    ]);

    let y = approx(&x, x0, gamma, sigma, intensity);
    let y0 = y[0].as_();
    for (idx, accurate_value) in accurate {
        let approx_value = y[idx].as_();
        if !approx_eq(approx_value, accurate_value, y0, precision) {
            println!("|{} - {}| > {:.2e}", approx_value, accurate_value, precision);
            return false;
        }
    }
    true
}



