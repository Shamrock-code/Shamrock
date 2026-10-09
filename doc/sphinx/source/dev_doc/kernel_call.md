# `sham::kernel_call`

`sham::kernel_call` submits a SYCL kernel over `n` threads. Inputs and outputs are
wrapped in `MultiRef`; buffer access (`get_read_access` / `get_write_access`) and
event completion are handled automatically.

Header: `shambackends/kernel_call.hpp`.

## Basic usage

C++ cannot expand two parameter packs, so inputs and outputs are passed as
`MultiRef`:

```cpp
sham::kernel_call(
    queue,
    sham::MultiRef{/* inputs */},
    sham::MultiRef{/* outputs (in-out) */},
    n,   // thread count
    [](u32 i, /* input ptrs */, /* output ptrs */) {
        // ...
    });
```

Minimal example:

```cpp
sham::kernel_call(
    q,
    sham::MultiRef{buf_in},
    sham::MultiRef{buf_out},
    n,
    [](u32 i, const T *in, T *out) {
        out[i] = in[i];
    });
```

- Functor first argument is the index (`u32`); then pointers in `MultiRef` order (inputs, then outputs)
- Read buffers → `const T*`; write buffers → `T*`
- Prefer `const sham::DeviceBuffer&` for pure inputs

## `MultiRef`

`MultiRef` holds references to buffer-like objects passed to `kernel_call`.
Each member is what the kernel accesses — it must provide `get_read_access`,
`get_write_access`, and `complete_event_state` (e.g. `DeviceBuffer<T>` or a custom
accessor).

## Under the hood

- `get_read_access` / `get_write_access` on each `MultiRef` member
- launch `n` threads
- `complete_event_state` after the kernel

## Related variants

| API | Role |
|-----|------|
| `kernel_call_u64` | same, `u64` index / count |
| `kernel_call_hndl` | functor returns a SYCL handler lambda |
| [`distributed_data_kernel_call`](distributed_data_kernel_call.md) | per-patch call over distributed data |

Custom accessors (any type with `get_read_access` / `complete_event_state`) and
`MultiRefOpt` for optional buffers are also supported — see Doxygen in
[`kernel_call.hpp`](../../../src/shambackends/include/shambackends/kernel_call.hpp).

## Related files

- [`kernel_call.hpp`](../../../src/shambackends/include/shambackends/kernel_call.hpp)
- [`kernel_call_distrib.hpp`](../../../src/shambackends/include/shambackends/kernel_call_distrib.hpp)
- [`kernel_call_tests.cpp`](../../../src/tests/shambackends/kernel_call_tests.cpp)
