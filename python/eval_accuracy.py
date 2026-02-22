from typing import Callable
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import voigt_profile
from fast_voigt import fast_voigt16_single, fast_voigt16, fast_voigt24, fast_voigt32

EPS = 1E-15

def eval_approx(approx: Callable, xax: np.ndarray, rho_ax: np.ndarray):
    ''' Calculate relative and absoltue error of `approx` for 
        given `ρ` and `x` arrays.
            γ = ρ 
            σ = 1-ρ
    '''
    abs_err = np.zeros((len(rho_ax), len(xax)), dtype= float)
    rel_err = np.zeros((len(rho_ax), len(xax)), dtype= float)
    for i, rho in enumerate(rho_ax):
        gamma, sigma = rho, 1.0 - rho
        y_accurate = voigt_profile(xax, sigma, gamma)
        y_approx = approx(xax, 0.0, gamma, sigma, 1.0)
        a = np.abs(y_accurate-y_approx)
        r = a / y_accurate 
        abs_err[i] = a / y_accurate[0]
        rel_err[i] = r
    return abs_err, rel_err

def calc_extent(xax:  np.ndarray, rho_ax:  np.ndarray):
    """ Helper for imshow's extent calculation
    """
    dx = xax[1] - xax[0]
    drho = rho_ax[1] - rho_ax[0]
    return (
        xax.min()-0.5*dx, 
        xax.max()+0.5*dx, 
        rho_ax.min()-0.5*drho, 
        rho_ax.max()+0.5*drho
    )



approximations = (
    ('W16 (f32)',  fast_voigt16_single, (0,0)),
    ('W16 (f64)',  fast_voigt16, (0,1)),
    ('W24 (f64)',  fast_voigt24, (1,0)),
    ('W32 (f64)', fast_voigt32, (1,1)),
)

xax = np.linspace(0.0, 20.0, 1000)
rho_ax = np.linspace(0.0, 1.0, 500)
extent = calc_extent(xax, rho_ax)

fig, axs = plt.subplots(2, 2, figsize = (11, 8))
fig.suptitle('log₁₀(|ε|) for the family of Weideman approximations')
for approx_name, approx, ax_pos in approximations:

    ax: plt.Axes = axs[*ax_pos]
    if 'f32' in approx_name:
        xax_single = xax.astype(np.float32)
        abs_err, rel_err = eval_approx(approx, xax_single, rho_ax)
        vmin = -10
    else:
        abs_err, rel_err = eval_approx(approx, xax, rho_ax)
        vmin = np.log10(EPS)

    n_nan = np.count_nonzero(np.isnan(abs_err)) 
    assert(n_nan == 0)

    abs_err = np.clip(abs_err, a_min=EPS, a_max=None)

    log_abs_err = np.log10(abs_err)
    img = ax.imshow(log_abs_err, origin='lower', extent=extent, vmin=vmin)
    ax.set_title(approx_name)
    ax.set_aspect("auto")
    ax.set_xlabel('x')
    ax.set_ylabel('ρ')
    fig.colorbar(img)

fig.tight_layout()
plt.show()