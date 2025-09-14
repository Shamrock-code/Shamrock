# Smoothing Length Tolerance in SPH

## Overview

The choice of smoothing length tolerance can greatly impact the performance of the SPH solver. It is important to avoid setting it too large, which would result in extra overhead due to the increased neighbor count, or too low, which would result in too many iterations to converge.

## Parameters

The Shamrock SPH solver implements two tolerance parameters:

- **`htol_up_coarse_cycle`**: Factor applied to the smoothing length for neighbor search and ghost zone size
- **`htol_up_fine_cycle`**: Maximum factor of smoothing length evolution per subcycle

The coarse cycle tolerance must be greater than or equal to the fine cycle tolerance:
```cpp
htol_up_coarse_cycle >= htol_up_fine_cycle
```

### Default Values

```cpp
Tscal htol_up_coarse_cycle = 1.1;  // Default: 1.1
Tscal htol_up_fine_cycle  = 1.1;  // Default: 1.1
```

## Setting the Right Tolerance

This is, in principle, not too difficult. Using `1.1` results in a moderate excess of neighbors while allowing the smoothing length to converge in a single coarse cycle during most simulations. If your simulation has very fast advecting components with large density contrasts, you will see warnings like:
```
Warning: smoothing length is not converged, rerunning the iterator ...                [Smoothinglength][rank=0]
     largest h = 0.8310577409570404 unconverged cnt = 99994
```
in the logs. If this happens, you can try increasing the tolerance to something like `1.15` or `1.2`, which should solve the issue at the cost of a slight performance slowdown.

The tolerance can be set in the runscript using
```py
model.change_htolerances(coarse=1.1, fine=1.05)
```
