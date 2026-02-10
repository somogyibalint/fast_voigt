use std::{hint::black_box};
use std::ops::AddAssign;
use criterion::{criterion_group, criterion_main, Criterion};
use fast_voigt::*;

use num_traits::{Float};
use pulp::x86::V3;

fn linspace<T>(fr: T, to: T, n: u32) -> Vec<T> 
where T: Float + AddAssign 
{  
    let mut x: T = T::zero();
    let dx: T = (to - fr) / ( T::from(n - 1).unwrap() );
    let mut array: Vec<T> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        array.push(x);
        x += dx;
    }
    array
}

fn w16_scalar_benchmarks(c: &mut Criterion) {
    let x = linspace(0.0f32, 5.0f32, 2048);
    c.bench_function("w16 scalar, f32", |b| b.iter(|| w16_f32_f32scalar(&x, 0.0, 0.5, 0.5, black_box(1.0))));
    c.bench_function("w16 scalar, f32", |b| b.iter(|| fast_voigt16_s(&x, 0.0, 0.5, 0.5, black_box(1.0))));
    let x = linspace(0.0f64, 5.0f64, 2048);
    c.bench_function("w16 scalar, f64", |b| b.iter(|| fast_voigt16(&x, 0.0, 0.5, 0.5, black_box(1.0))));
    c.bench_function("w24 scalar, f64", |b| b.iter(|| fast_voigt24(&x, 0.0, 0.5, 0.5, black_box(1.0))));
    c.bench_function("w32 scalar, f64", |b| b.iter(|| fast_voigt32(&x, 0.0, 0.5, 0.5, black_box(1.0))));
}

fn w16_simd_benchmarks(c: &mut Criterion) {
    let simd = V3::try_new().unwrap(); 
    let x = linspace(0.0f32, 5.0f32, 2048);
    c.bench_function("w16 avx2, f32", |b| b.iter(|| weideman16_avx2_f32(simd, &x, 0.0, 0.5, 0.5, black_box(1.0))));
}

criterion_group!(benches, w16_scalar_benchmarks, w16_simd_benchmarks);
criterion_main!(benches);
