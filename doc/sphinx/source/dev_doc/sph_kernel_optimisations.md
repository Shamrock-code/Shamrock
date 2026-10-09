# SPH kernel optimisations

This page documents optimisations of the SPH kernel functions defined in
`shammath/sphkernels.hpp` (`shammath::SPHKernelGen`): the underlying choices of optimisations, why
they are valid, and what they cost in accuracy, if anything.

## Column-integrated kernel `Y_3d`

### What it computes

The column renders (`SPHColumnInteg`, exposed as `render_column_integ` and
`render_cartesian_column_integ`) and the azimuthal renders (`SPHAzymuthalInteg`, exposed as
`render_azymuthal_integ`) integrate a field $A$ along rays. Every particle $b$ whose support is
crossed by the ray contributes

$$
\frac{m_b A_b}{\rho_b} \, Y_{3d}(r, h_b),
$$

where $r$ is the distance between the particle and the ray, and $Y_{3d}$ is the kernel integrated
along the line of sight:

$$
Y_{3d}(r, h) = \frac{C_{\rm norm}}{h^2} \int_{-R}^{R} f\left(\sqrt{x^2 + z^2}\right) {\rm d}z,
\qquad x = \frac{r}{h},
$$

with $R$ the kernel support radius (`Rkern`). The integral is a Riemann sum over $2 n_p$ points
$z_k = k \, \Delta z$, $\Delta z = R / n_p$, $k = -n_p, \dots, n_p - 1$. The renders use
$n_p = 4$, and $Y_{3d}$ is evaluated once per (ray, particle) pair, so it dominates the cost of the
render kernels.

### Symmetric sampling

<!-- inlined (not an <img>) so that the figure picks up the theme colours and the light/dark switch -->
```{raw} html
:file: sph_kernel_y3d_sampling.svg
```

Three properties of the sum reduce the number of kernel evaluations from $2 n_p$ to $n_p$:

1. **Symmetry.** $f(\sqrt{x^2 + z^2})$ is even in $z$, and the grid is exactly symmetric around 0
   ($z_k$ is computed as `k * step`, so $z_{-k} = -z_k$ bitwise). Each sample with
   $k = 1, \dots, n_p - 1$ is evaluated once and counted twice.
2. **Compact support.** The $z = -R$ sample is $f(\sqrt{x^2 + R^2})$, whose argument is at least
   $R$, where every kernel is 0. It is dropped.
3. **$z = 0$ needs no square root.** $\sqrt{x^2} = |x|$ exactly in binary floating point (barring
   over/underflow), so the centre sample is `f(|x|)`.

$\Delta z$ is then factored out of the sum:

$$
\sum_{k=-n_p}^{n_p-1} f\left(\sqrt{x^2 + z_k^2}\right) \Delta z
= \Delta z \left[ f(|x|) + 2 \sum_{k=1}^{n_p-1} f\left(\sqrt{x^2 + z_k^2}\right) \right].
$$

For $n_p = 4$ this is 4 kernel evaluations and 3 square roots instead of 8 and 8.

Before, a plain Riemann sum with a runtime `np`:

```cpp
inline static Tscal f3d_integ_z(Tscal x, int np = 32) {
    return integ_riemann_sum<Tscal>(-Rkern, Rkern, Rkern / np, [&](Tscal z) {
        return f(sqrt(x * x + z * z));
    });
}
```

After, `np` is a template parameter, so the loop is fully unrolled and `step` is a constant:

```cpp
template<int np>
inline static Tscal f3d_integ_z(Tscal x) {
    constexpr Tscal step = Rkern / np;

    Tscal xx = x * x;

    Tscal acc = f(sycl::fabs(x)); // z = 0
    for (int k = 1; k < np; k++) {
        Tscal z = k * step;
        acc += 2 * f(sqrt(sycl::fma(z, z, xx)));
    }
    return acc * step;
}
```

### Usage

- `Kernel::template Y_3d<np>(r, h)` is the optimised version, to be used in device kernels (the
  render modules call `Kernel::template Y_3d<4>(rab, h_b)`).
- `Kernel::Y_3d(r, h, np)` and `Kernel::f3d_integ_z(x, np)` keep the plain Riemann sum with a
  runtime `np`. They serve as the reference in the unit tests (including the CPU reference of the
  render tests in `SPHRenderTestCommon.hpp`), and back the Python binding
  `shamrock.math.sphkernel.<kernel>_f3d_integ_z`.

### Accuracy

The result is **not** bitwise identical to the plain Riemann sum: the additions are reordered and
the multiplication by $\Delta z$ is factored out, so each term may differ by a few ulp.

- The unit test `shammath/sphkernels/f3d_integ_z_symmetric` compares both versions for every
  kernel, in `f32` and `f64`, for $n_p \in \{4, 8, 16, 32\}$, with a tolerance of
  $4 \cdot 2 n_p \cdot \varepsilon \cdot Y(0)$.
- On the outputs of `examples/sph/run_sph_rendering.py`, the density renders differ from the
  previous implementation by at most $8 \times 10^{-16}$ relative per pixel (about $4\varepsilon$).
  For the $v_z$ renders, the largest difference is below $10^{-15}$ of the image's largest
  value. The relative difference per pixel only gets larger (up to $10^{-10}$) where positive and
  negative contributions cancel, so the pixel value is close to 0.

:::{note}
Keeping bitwise-identical results is possible (evaluate only the $z \le 0$ samples, then add the
same $2 n_p$ terms in the original order), but it saves fewer evaluations (5 instead of 4 for
$n_p = 4$). It also depends on FMA contraction: the old code compiled `x * x + z * z` with `z` in a
register, which the CUDA backend contracts as `fma(z, z, x * x)`. With a compile-time `z`, the
compiler chose `fma(x, x, z * z)` instead, which rounds differently. And with
`-ffp-contract=fast` on x86 with `-march=native`, sharing the mirrored products changes which
multiply-adds get fused, so even the old code's last bit depends on where it gets inlined. Bitwise
reproducibility across such changes therefore needs explicit `sycl::fma` calls, not just the same
summation order.
:::

### Performance

Timings of the render calls of `examples/sph/run_sph_rendering.py` (100k particles, $1024^2$ rays
per render), before and after the change:

| Render | Hardware | Before | After | Speedup |
| --- | --- | --- | --- | --- |
| Cartesian column $\rho$ | RTX 3070, median | 0.589 s | 0.375 s | 1.57x |
| Azimuthal $\rho$ | RTX 3070, median | 10.08 s | 7.25 s | 1.39x |
| 4 column renders, total | CPU (OpenMP, 4 cores), mean of 3 | 6.63 s | 6.14 s | 1.08x |
| 2 azimuthal renders, total | CPU (OpenMP, 4 cores), mean of 3 | 65.6 s | 60.4 s | 1.09x |

The gain is larger on the GPU because it runs FP64 at a small fraction of its FP32 rate (1/64 on
the RTX 3070), so cutting FP64 kernel evaluations matters more there. On the CPU, the column render
time is likely dominated by other parts of the call (tree build, ray setup, traversal).

The CPU times are the `compute_column_integ took ...` / `compute_azymuthal_integ took ...` lines
logged by the render modules, with the two builds run alternately to limit drift.
