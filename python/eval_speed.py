from typing import Callable
from time import perf_counter
import numpy as np
from numpy.random import rand
from matplotlib import pyplot as plt
from scipy.special import voigt_profile
import scalar.fast_voigt as scalar
import avx2.fast_voigt as avx2
import avx512.fast_voigt as avx512

def usecs_per_array(elapsed_sec, n_samples):
    return 1E6*elapsed_sec/float(n_samples)

approximations_single = [
    ('W16s', scalar.fast_voigt16_single),
    ('W16s-AVX2', avx2.fast_voigt16_single),
    ('W16s-AVX512', avx512.fast_voigt16_single),
]

approximations_double = [
    ('W16', scalar.fast_voigt16),
    ('W24', scalar.fast_voigt24),
    ('W32', scalar.fast_voigt32),
    ('W16-AVX2', avx2.fast_voigt16),
    ('W24-AVX2', avx2.fast_voigt24),
    ('W32-AVX2', avx2.fast_voigt32),
    ('W16-AVX512', avx512.fast_voigt16),
    ('W24-AVX512', avx512.fast_voigt24),
    ('W32-AVX512', avx512.fast_voigt32),    
]

n_samples = 100_000
rho = rand(n_samples)

xax = np.linspace(0.0, 20.0, 2048)
xax_single = xax.astype(np.float32)

max_error = {
    'W16s': 6.5446e-07,
    'W16': 1.2459e-07,
    'W24': 7.9514e-11,
    'W32': 3.9417e-14,
}

# reference implementation
start = perf_counter()
for r in rho:
    y =  voigt_profile(xax, r, 1.0-r)
elapsed = {'SGJ' : usecs_per_array(perf_counter() - start, n_samples)}


for k, approx in approximations_single:
    start = perf_counter()
    for r in rho:
        y =  approx(xax_single, 0.0, r, 1.0-r, 1.0)
    elapsed[k] =  usecs_per_array(perf_counter() - start, n_samples)


for k, approx in approximations_double:
    start = perf_counter()
    for r in rho:
        y =  approx(xax, 0.0, r, 1.0-r, 1.0)
    elapsed[k] =  usecs_per_array(perf_counter() - start, n_samples)


_scalar, _avx2, _avx512 = [], [], []

for name, dt in elapsed.items():
    if name == 'SGJ':
        continue
    elif 'AVX512' in name:
        _avx512 += [dt]
    elif 'AVX2' in name:
        _avx2 += [dt]
    else:
        _scalar += [dt]
_scalar, _avx2, _avx512 = np.array(sorted(_scalar)), np.array(sorted(_avx2)), np.array(sorted(_avx512))

fig, axs = plt.subplots(1, 2, figsize = (10, 4))
ax = axs[0]
ax.plot(max_error.values(), sorted(_scalar), 'o', label='scalar', markersize=9)
ax.plot(max_error.values(), sorted(_avx2), 'v', label='AVX2', markersize=9)
ax.plot(max_error.values(), sorted(_avx512), '*', label='AVX512', markersize=9)
ax.hlines(y=elapsed['SGJ'], xmin=1E-14, xmax=2E-6, ls=':', color='red')
ax.set_xlabel('Absolute error: max(|ε|)')
ax.set_ylabel('Δt (μs)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(0.8, 210)
ax.set_xlim(1E-15, 1E-5)
ax.text(2E-15, 105, 'SGJ', color='red', fontsize=16, style='oblique')
ax.legend(loc='lower left')

ax = axs[1]
ax.plot(max_error.values(), elapsed['SGJ'] / _scalar, 'o', label='scalar', markersize=9)
ax.plot(max_error.values(), elapsed['SGJ'] / _avx2, 'v', label='AVX2', markersize=9)
ax.plot(max_error.values(), elapsed['SGJ'] / _avx512, '*', label='AVX512', markersize=9)
ax.set_xlabel('Absolute error: max(|ε|)')
ax.set_ylabel('Speedup')
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_ylim(0, 60)
ax.set_xlim(1E-15, 1E-5)
ax.legend(loc='upper left')
fig.tight_layout()
plt.show()


