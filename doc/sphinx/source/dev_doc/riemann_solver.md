# Riemann solver implementation notes

This page documents implementation choices made in `shammath`'s Riemann solvers (Rusanov, HLL,
HLLC, dust HLL, Huang-Bai).

## Axis permutation vs projection

This section adds an `_n` suffix to its examples to tell the projection variant apart from the
permutation variant in the discussion below.

Every Riemann solver is defined once as an `_n` variant that takes the face's unit normal `n`
directly:

```cpp
template<class Tprim>
inline constexpr auto riemann_solver_flux_n(
    Tprim primL, Tprim primR, typename Tprim::Tscal gamma, typename Tprim::Tvec n) {
    // ... flux computation using n[0], n[1], n[2] directly ...
}
```

Until recently, each solver also shipped six `_x`/`_y`/`_z`/`_mx`/`_my`/`_mz` wrappers that got to
the same result by **permuting** components to the `+x` axis, calling the `_x` solver, then
permuting the result back — instead of **projecting** through `_n` with the axis's unit vector
directly. E.g. the `-z` wrapper:

```cpp
// axis permutation: rotate the inputs to +x, solve there, rotate the result back
template<class Tprim>
inline constexpr auto riemann_solver_flux_mz(Tprim pL, Tprim pR, typename Tprim::Tscal gamma) {
    return invert_axis(
        riemann_solver_flux_z(prim_invert_axis(pL), prim_invert_axis(pR), gamma));
}
```

`riemann_solver_flux_mz(pL, pR, gamma)` and `riemann_solver_flux_n(pL, pR, gamma, {0, 0, -1})`
are mathematically equivalent, but they are not equivalent *as generated code*.

### Comparing the assembly

{download}`riemann_solver_axis_dispatch_godbolt.cpp` is a minimal, dependency-free reproducer (no
SYCL, just a plain `Vec3`) isolating exactly this comparison for one solver, with two entry
points:

```cpp
Cons via_mz_dispatch(Prim pL, Prim pR) {
    return riemann_solver_flux_mz(pL, pR);
}

Cons via_flux_n(Prim pL, Prim pR) {
    return riemann_solver_flux_n(pL, pR, Vec3{0, 0, -1});
}
```

Try it live on [Compiler Explorer](https://godbolt.org):

```{raw} html
<div id="riemann-godbolt-wrap">
  <button id="riemann-godbolt-toggle" type="button">⤢ Expand</button>
  <iframe id="riemann-godbolt-iframe"
          src="https://godbolt.org/e#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIAOykrgAyeAyYAHI%2BAEaYxBIAzABspAAOqAqETgwe3r4BaRlZAqHhUSyx8VzJtpj2jgJCBEzEBLk%2BfoF2mA7ZTS0EpZExcYkpCs2t7fldk4NhwxWj1UkAlLaoXsTI7BwA9HsA1AC0p4csYXgsYocThujHYYf8xLcIrNFMyADWrgogIAAbj1DhAFABPZC0AHA0wJNzoTbReikBImBLYVYnU4mDQAQQmxC8DkOADUeglDiZ/FZ8Yd6YdEV5kZhDqpSIdwRyAF7o2l4hmHNAMCaYVSpV5Mln01CpOJMIjEEwAVisyoAIhAwgRDngscKJlSaYdiJgCFsGLqqQl1ej1YcNNaAGJsw4gUF4a22m2HLjOzluw7c1Z8o22ukMg0EMUSxlI%2BhUsxJWXyxUqtWa7W6rGC%2BnUiwms0Wq12u0O/2qQNar1lv3ol3gwPB0PU8N41t83F4qMx17k5CUlPEBUkY4QfuUpg53Mz/OF83ES0T/PHJgAOnZJ3XXK3a95/m9BY7%2BJ74r7FMOQ5HSss44vTA5E8O0WneeNpoXS4p%2BfXlesz43Dlt0TAtojXHd125ECAP3Q8wy7U9YyfK9FQAKjvAdDgfONmQTBRpznD9i2XGlf0OVDbiA8DyMorC9xohQOwSI8Dy7LsDmxY43lYa4CCQEB1S8CZlGIK5%2BmjQ4jkEiYPBFcTWQgAhRJYNh0CxU5ji7aMWFSAxo3RNwoSYBQFEOAAVWEAH10WwLtCWJHVpIIESxOaCT8y7QUhLCYBzNhL0/J6azmM8hkpQTYgEFQDzWJCiN6QskFgVoGLDzY2L%2BS0zAdL0zADKMkzAuQYLbPxeySSc2SFHko1%2BS8zIjCKgLEuKzt4pw6VIuimkmLqhkWpNKLktStr23axDXkqgRqrc1lzGTOVhzQu0IHC1kqCxDz2sFLryLLKhRpnekuuSgKTrqGiDrigUjqIxdyL4vBGOuwVj3bDK2PxbTdIVPL4QK0yWpshDpujM9DimuTZoMoGMUvRbr2sCAowhoS2mm%2BSYdhGzEySbCUchmbfqxnocfml9atC46i3uwnMfhWHbNItddv/MCuqo87aGg9mhrqXr0rbdijg0k08GywwGEstAVIENcEFSVJDgQOpFtMr5zTEWhGyEzB0EvS0%2BNZFgoNSBUEE4rKcuJ/6DEKxmuzCWgFiFUHe1RmSMehhnsbh9BLIQcF0GIVBLKobxVEshgICclyWHptxGcZSyJSuDkBoYDk1sOQEGE2nr2rp73E997Bk/DrxVEOsKw4jlmoten1/dTlh69QGjc%2Br%2Bl/Yr1Q29Ostm%2BUtdToo1aU%2BH3aKNzkMXoZO7LR7iPRre77cvyu3AdLx2GGd8JXZFMHYyLm2S9Jv2A6DkPa8rqOY7RuOE6Toe06avPKfG0GOoTXOAoYdMNAqntBRF%2BrdhqqkARqaC/9VRcCATRUBI9%2BawPgf%2BGBFgzDwJARPK4SCUqqkwRqLu85iz%2B0DsHUOvc76gI5JnHOecV6sXxFbH6%2BlbbGUBgaYG%2BInYuwmuZA0ydI5EEstyCAZlBHIHzn1BKgj/bIAmMQ%2BREw25HTLMgNuSjpYqK5gA1BPpjgaN0QQoBWiFEEH7sgiwcCoHqMsfg6xpi57d20RY4xGD9H2iMXzBxkC0rtQXsncxjC2xr1PgDAR01uF4l4fvfhEjprJzCMCVolkmCqCeuIyR0iqaRJFEExRziCluIbgyOxXUzE6J8QFMc3jUDJVnjIkh91lEEBCSw9e7D7Yt2ibE1k8SW7JxbqIyyIjVDiMGakHJ7UzKTJbn9JpitlKYFUWUn0qRNFFKWVcFZ4CLB%2BIChsvZhD/E3XpNstgeD0w2PtGWI5VibnEIubsqxJyan3N8U4ppgTnntK%2BtlVhf1DKb3Mj0jEO8979LduDWZykhnKUsskuIBA0kZIUBMuFUyP5nNBZi%2BZTz5mqLuZsxZhKB4GIgB8xpuSfn4uum9YWnExYSwYFLdAaMA5eEMMASynw8Dy0VqCYyuodRPUOEQdATAdyxBeKyU0LB6liHUjif51s2HAo4biq4vTd58OhbGJgXgiDKy5UYXlTA8A30jtHWFVx4VXGCOnQZoCABKHICDgjlMwNgWqWAAgztMnFKNDXGtzsEP%2BeioHYJbsEK5EDUGWEOOgm5CCcEsFjXslNaD0xvOjcpDNryvm5ODUa9uucXURvjVG%2B1LAXVxv2QmgsyasE1rrZmxtSac0ttdfWk5K9P6HywqW544bB6XwoVa6habHVJo5GG6lA7DQhvblQCtY7yHXyodHV1tC50MBdQusaOKT7qo9V61grJbV%2BpAM/K1I16XtTwFQUEYbrRl0dPNea9C11ww0CGbaNde4BSoMEYhdQlC6mfRAV9BlyyfqTN%2B60bgHT/pxYKJeldgMurA7QCDT6X0MFHfCODSYv3lrfSh3J6GrV7R9IAop4HWT4eg4RijH7SMIfI7Bv9VHAMRwCtQUdibV2HppTTRed76VMKPWE9VESr06shQfUU4Nl0mu5eay1VDxlXprTO3TrrA07XE%2Bps1fLJ3bunVnNNbrxWesYBe31/rsY0i4ByDQ7nWyiekx08JIKFPgp4bquJ%2BrXhqYQKanl5mqFiN06kfTkyD3YuM5%2BIRozQ5iIixp6LEdLLjNASMsZlLgjrBrYV0O4zUgHtWN5oWqrAUb01QF0qMTgtQsPu7cLkXNOTpNhiu18WnWYqS1tHFgT/ZItSekzJWWzMWsnWIgrk2UXTfRfF0rS2GApJW2iyl1XaufTxBxUWRtxUAHd26iFoNzTI0ZTLGvwFQZ9GkuzeUanHAKsdlIJ2XOC66b3fJVU%2B2jKqP3vx/cyswswCQn1YGfaSOB%2BIgeAjwEwSyJtLL4AUGbAgyAEAQA%2B4Nw4hORsFzGyZ2bUX5tUL60TqrB2ocJFcE%2Bw75gYdUDh2STBSPEko7R1ugnmKZ0k6M/PCn3Wcu32jnT2zJF9nuY5McOBB4Gftmh8zqgHB1jQg4MqXgfgOBaFIKgTgbhrD/gUJsbYc1oc8FIAQTQWv1jfBAMkNcZguBJA917z33uzBmH0JwSQ%2BvHfG84Lwf4HmHeG616QOAsAkAy1SHQOI5BKBJ5T/EIyRhbceZoLQaMxB/gQGiKH6IYQWjgk4Hb8vzBiDggAPLRG0D0aPduZZsEEA33eVeY%2BkCwNELwwA3Ba3%2BNwXgWBrhGHEH3/AppejAjH0bsUPQjW7Dt9qOoofnbRGHPXjwWBQ9KSuNX3gKTogZEwOqFlwA96gBj%2BscOTBgAKFJOLM7DevWn5kIIEQYh2BSA/7yBKBqCh66BuYGBGAoDm6WD6B4DRD/CQDrCygNAiicDHAN6UjHATDoB2imCWDWBmCOjHDqikhmC8D1JxCiRYCIEQDrDdC9DOAQCuDTB%2BBuYhALDlCVB6DpC3bZCsE8FFCoFDBcHLC1D1B9BzACFuYMGoH9CtAiEjBVC2BSGeAdB6B3AKGcFKESD0FW47C6GB664h594m4cBsgAAcSQxwSQkgQokBvk0Oa4XAa4joEAuAhAJAiYCQXAqwvA0eWg6wEAieqAOkmeaeyMoRye9AWeDhuefAdAhexepefetele3%2BaR9eTeLeDg3%2BHejABA3e2soeA%2BQ%2BI%2B12Y%2Bduk%2B3KM%2BRuc%2BreeAi%2BoeK%2ByAa%2B3%2Bm%2BOuRuO%2Be%2B4IB%2BuwRux%2Bfq4%2BpA5%2Bl%2B1%2BU%2Bt%2BPkjuj%2BBgL%2Bb%2BmAH%2BX%2Bwx/Av%2BV2AB0gKxwBKg6gfeugAeDh0BBBsBO%2BtByBqQqBY%2BGBWBOBeBMBGCxBpBdulBxA1BmApx4hDRTBLBah%2BQ7BDA6AihSwyhvBxQOQPxbBhQfBJQ2hQJGhdQnxDA8h6MeQEJshkhAwgJ3BMhqhKJGhcwmJywehWwBhvhRheupABuRuZhlh1hth9h3KhwThLhbhHhio3hvh/h0xceSAj2VAERmxf%2B4ggBmxig2x2%2BCA/wEBEpvJZ67AHmtAEpYoSkTAKB2Q4epAkU/wAempMp9mIAHmmpSpw4qpzgHAseOu5JlJFBnA6oT68O7%2BcQhwZuRxBYluxJNuPhNJNhdh2ejhCQzhrhJwmBJwNxNo%2BBVglgRBJwpBZgOcpkzpEZrp%2BhHpfoqgVh3p9JjUTJgZVxIZBAuBYZdxUZJBCOnJD%2BzuIASQ/p/gFhCQNZdZtZ9Z/g/gyoRhweFJoeZhEeIAAeARTuRh5BHZph6pfZNWIxcQmQzgkgQAA%3D%3D">
  </iframe>
</div>

<div id="riemann-godbolt-backdrop"></div>

<style>
#riemann-godbolt-toggle {
  margin-bottom: 6px;
  padding: 4px 12px;
  cursor: pointer;
}
#riemann-godbolt-iframe {
  width: 100%;
  height: 600px;
  zoom: 0.6;
  border: 1px solid #ccc;
  display: block;
}
#riemann-godbolt-iframe.riemann-godbolt-expanded {
  position: fixed;
  top: 3vmin;
  left: 3vmin;
  width: calc(100vw - 6vmin);
  height: calc(100vh - 6vmin);
  zoom: 1;
  z-index: 100000;
}
#riemann-godbolt-backdrop {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  z-index: 99999;
}
#riemann-godbolt-backdrop.riemann-godbolt-visible {
  display: block;
}
</style>

<script>
(function () {
  var btn = document.getElementById("riemann-godbolt-toggle");
  var frame = document.getElementById("riemann-godbolt-iframe");
  var backdrop = document.getElementById("riemann-godbolt-backdrop");
  var expanded = false;

  function setExpanded(value) {
    expanded = value;
    frame.classList.toggle("riemann-godbolt-expanded", expanded);
    backdrop.classList.toggle("riemann-godbolt-visible", expanded);
    btn.textContent = expanded ? "✕ Close" : "⤢ Expand";
  }

  btn.addEventListener("click", function () {
    setExpanded(!expanded);
  });
  backdrop.addEventListener("click", function () {
    setExpanded(false);
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      setExpanded(false);
    }
  });
})();
</script>
```

Compiled with x86-64 clang 23 at `-O3`, both fully inlined into a single leaf function, the pinned
output disassembles to:

::::{grid} 2
:gutter: 2

:::{grid-item}
**Axis permutation** — `via_mz_dispatch`

```text
.LCPI0_0:
        .quad   0x8000000000000000
        .quad   0x8000000000000000
via_mz_dispatch(DustPrimState<Vec3>, DustPrimState<Vec3>):
        mov     rax, rdi
        movapd  xmm1, xmmword ptr [rsp + 8]
        movsd   xmm5, qword ptr [rsp + 32]
        movsd   xmm4, qword ptr [rsp + 64]
        movsd   xmm0, qword ptr [rsp + 24]
        movapd  xmm3, xmmword ptr [rip + .LCPI0_0]
        xorpd   xmm0, xmm3
        movsd   xmm6, qword ptr [rsp + 56]
        xorpd   xmm6, xmm3
        movapd  xmm2, xmm5
        unpcklpd        xmm2, xmm4
        xorpd   xmm2, xmm3
        movhpd  xmm1, qword ptr [rsp + 40]
        mulpd   xmm1, xmm2
        movsd   xmm3, qword ptr [rsp + 16]
        mulsd   xmm3, xmm1
        shufpd  xmm0, xmm2, 2
        mulpd   xmm0, xmm1
        unpcklpd        xmm2, xmm6
        mulpd   xmm2, xmm1
        movapd  xmm6, xmm1
        unpckhpd        xmm6, xmm1
        movsd   xmm7, qword ptr [rsp + 48]
        mulsd   xmm7, xmm6
        xorpd   xmm8, xmm8
        ucomisd xmm8, xmm5
        jbe     .LBB0_4
        ucomisd xmm4, xmm8
        jae     .LBB0_4
        unpcklpd        xmm0, xmm2
        movapd  xmm8, xmm1
        movapd  xmm4, xmm3
.LBB0_3:
        movapd  xmm5, xmm0
        jmp     .LBB0_7
.LBB0_4:
        ucomisd xmm5, xmm8
        jbe     .LBB0_8
        ucomisd xmm4, xmm8
        jbe     .LBB0_8
        unpckhpd        xmm2, xmm0
        movapd  xmm8, xmm6
        movapd  xmm4, xmm7
        movapd  xmm5, xmm2
.LBB0_7:
        xorpd   xmm5, xmmword ptr [rip + .LCPI0_0]
        movsd   qword ptr [rax], xmm8
        movsd   qword ptr [rax + 8], xmm4
        movupd  xmmword ptr [rax + 16], xmm5
        ret
.LBB0_8:
        ucomisd xmm5, xmm8
        seta    cl
        ucomisd xmm4, xmm8
        setb    dl
        ucomisd xmm5, xmm8
        xorpd   xmm5, xmm5
        jae     .LBB0_13
        and     cl, dl
        jne     .LBB0_13
        xorpd   xmm9, xmm9
        ucomisd xmm4, xmm9
        xorpd   xmm4, xmm4
        jbe     .LBB0_7
        shufpd  xmm2, xmm2, 1
        addsd   xmm6, xmm1
        addpd   xmm0, xmm2
        addsd   xmm7, xmm3
        movapd  xmm8, xmm6
        movapd  xmm4, xmm7
        jmp     .LBB0_3
.LBB0_13:
        xorpd   xmm4, xmm4
        jmp     .LBB0_7
```

:::

:::{grid-item}
**Direct projection** — `via_flux_n`

```text
.LCPI1_0:
        .quad   0x8000000000000000
        .quad   0x8000000000000000
via_flux_n(DustPrimState<Vec3>, DustPrimState<Vec3>):
        mov     rax, rdi
        movsd   xmm5, qword ptr [rsp + 32]
        movsd   xmm2, qword ptr [rsp + 64]
        movapd  xmm3, xmmword ptr [rip + .LCPI1_0]
        movsd   xmm0, qword ptr [rsp + 8]
        xorpd   xmm0, xmm3
        movsd   xmm1, qword ptr [rsp + 40]
        xorpd   xmm1, xmm3
        mulsd   xmm0, xmm5
        movsd   xmm7, qword ptr [rsp + 16]
        mulsd   xmm7, xmm0
        movapd  xmm3, xmm0
        unpcklpd        xmm3, xmm0
        mulpd   xmm3, xmmword ptr [rsp + 24]
        mulsd   xmm1, xmm2
        movapd  xmm4, xmm1
        unpcklpd        xmm4, xmm1
        mulpd   xmm4, xmmword ptr [rsp + 56]
        xorpd   xmm6, xmm6
        ucomisd xmm6, xmm5
        unpcklpd        xmm0, xmm7
        jbe     .LBB1_4
        ucomisd xmm2, xmm6
        jae     .LBB1_4
        movapd  xmm5, xmm0
        movapd  xmm2, xmm3
.LBB1_3:
        movupd  xmmword ptr [rax], xmm5
        movupd  xmmword ptr [rax + 16], xmm2
        ret
.LBB1_4:
        lea     rcx, [rsp + 40]
        movapd  xmm7, xmm1
        mulsd   xmm7, qword ptr [rcx + 8]
        ucomisd xmm5, xmm6
        unpcklpd        xmm1, xmm7
        jbe     .LBB1_6
        ucomisd xmm2, xmm6
        ja      .LBB1_10
.LBB1_6:
        ucomisd xmm5, xmm6
        seta    cl
        ucomisd xmm2, xmm6
        setb    dl
        ucomisd xmm5, xmm6
        xorpd   xmm5, xmm5
        jae     .LBB1_12
        and     cl, dl
        jne     .LBB1_12
        ucomisd xmm2, xmm6
        xorpd   xmm2, xmm2
        jbe     .LBB1_3
        addpd   xmm1, xmm0
        addpd   xmm4, xmm3
.LBB1_10:
        movapd  xmm5, xmm1
        movapd  xmm2, xmm4
        movupd  xmmword ptr [rax], xmm5
        movupd  xmmword ptr [rax + 16], xmm2
        ret
.LBB1_12:
        xorpd   xmm2, xmm2
        movupd  xmmword ptr [rax], xmm5
        movupd  xmmword ptr [rax + 16], xmm2
        ret
```

:::
::::

:::{admonition} Analysis according to claude
Both keep the same control-flow shape (the solver's internal branch tree survives inlining
unchanged), but `via_mz_dispatch` does strictly more work for the same result:

- **More sign flips.** `via_flux_n` needs 2 real `xorpd`s (negating the `z` component once per
  side, since `n = {0, 0, -1}`). `via_mz_dispatch` needs 4 — two from rotating the inputs in
  (`prim_invert_axis` + `prim_z_to_x`) that don't fully cancel against the two undoing the
  rotation on the way out (`x_to_z` + `invert_axis`); one extra negation survives all the way to
  the final store with no counterpart in the direct-`n` version.
- **Lane shuffling.** `via_mz_dispatch` uses `unpcklpd`/`unpckhpd`/`shufpd`/`movhpd` to move
  vector components between lanes, a byproduct of routing everything through the `+x` axis.
  `via_flux_n` never permutes lanes: with `n_x = n_y = 0`, the relevant component is used in
  place.
- **More live registers and instructions for an identical result** (`via_mz_dispatch` reaches
  `xmm9`, `via_flux_n` stops at `xmm7`).

The compiler eliminates the literal-zero multiplies coming from `{1, 0, 0}` in both cases, but it
does not fully cancel the round trip's redundant sign flips and lane permutes. Calling `_n`
directly with the target axis's unit vector is not just cleaner source — it is strictly cheaper
codegen at `-O2`.
:::

### On-device benchmark

{download}`riemann_solver_axis_dispatch_sycl_bench.cpp` is the same comparison ported to real
`sycl::vec<double, 3>` and run as an actual kernel launch, over $2\cdot10^7$ randomly generated face
states, on whichever SYCL device you point it at. Inputs and outputs are USM device allocations
(`sycl::malloc_device`) on an in-order queue. It checks that both variants agree exactly before
timing them, then reports the best of 20 timed runs for each:

```text
❯ ACPP_VISIBILITY_MASK=omp ./a.out
Device: AdaptiveCpp OpenMP host device
N = 20000000 elements, 100 repeats per case (best of N reported)

correctness: max |via_mz_dispatch - via_flux_n| = 0.000e+00  (PASS)

via_mz_dispatch  : best of 100 runs =    61.519 ms  (3.076 ns/elem)
via_flux_n       : best of 100 runs =    60.624 ms  (3.031 ns/elem)
❯ ACPP_VISIBILITY_MASK=cuda ./a.out
Device: NVIDIA GeForce RTX 3070
N = 20000000 elements, 100 repeats per case (best of N reported)

correctness: max |via_mz_dispatch - via_flux_n| = 0.000e+00  (PASS)

via_mz_dispatch  : best of 100 runs =     9.680 ms  (0.484 ns/elem)
via_flux_n       : best of 100 runs =     9.682 ms  (0.484 ns/elem)
```

It seems that on CPU the use of projection yield a small gain thanks to the shorten assembly. On GPU the difference seems tiny (probably because it is still mostly memory bound).
