# `sham::distributed_data_kernel_call`

`sham::distributed_data_kernel_call` is the distributed counterpart to
[`sham::kernel_call`](kernel_call.md). It runs the same functor once per patch,
using `DDMultiRef` instead of `MultiRef` and a `DistributedData` of per-patch
thread counts.

Header: `shambackends/kernel_call_distrib.hpp`.

## Basic usage

```cpp
sham::distributed_data_kernel_call(
    dev_sched,
    sham::DDMultiRef{/* distributed inputs */},
    sham::DDMultiRef{/* distributed outputs */},
    thread_counts,   // DistributedData<index_t> of per-patch sizes
    [](u32 i, /* input ptrs */, /* output ptrs */) {
        // ...
    });
```

Minimal example:

```cpp
sham::distributed_data_kernel_call(
    dev_sched,
    sham::DDMultiRef{bufs_in},
    sham::DDMultiRef{bufs_out},
    sizes,
    [](u32 i, const T *in, T *out) {
        out[i] = in[i];
    });
```

- First arg is `DeviceScheduler_ptr` (not a queue) — queue is taken via `dev_sched->get_queue()`
- Inputs/outputs in `DDMultiRef` are any objects that declare `get(id)`; the result must be buffer-like (`get_read_access` / `get_write_access` / `complete_event_state`), e.g. `DistributedData<DeviceBuffer<T>>` or a custom accessor container
- `thread_counts` is `DistributedData<index_t>` keyed by patch id; each value is that patch's `n`
- Functor signature is the same as `kernel_call` (index + pointers)

## `DDMultiRef`

`DDMultiRef` holds references to distributed containers. `.get(id)` builds a
`MultiRef` by calling `.get(id)` on each member. Those per-id objects are what
`kernel_call` then accesses — they must provide the same buffer interface as for
`MultiRef` (`get_read_access`, `get_write_access`, `complete_event_state`).

## Under the hood

- For each patch id in `thread_counts`, build per-patch `MultiRef`s via `DDMultiRef::get(id)`
- Call `sham::kernel_call(queue, …, n, func)` for that patch

## Related variants

| API | Role |
|-----|------|
| `distributed_data_kernel_call_hndl` | same, but uses `kernel_call_hndl` |
| [`kernel_call`](kernel_call.md) | single-buffer (non-distributed) form |

## Related files

- [`kernel_call_distrib.hpp`](../../../src/shambackends/include/shambackends/kernel_call_distrib.hpp)
- [`kernel_call.hpp`](../../../src/shambackends/include/shambackends/kernel_call.hpp)
- [`kernel_call_distribTests.cpp`](../../../src/tests/shambackends/kernel_call_distribTests.cpp)
