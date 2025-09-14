# Smoothing Length Tolerance in SPH

## Overview

The choice of the smoothing length tolerance can greatly impact the performance of the solver 
it is important to avoid setting it too large which would result in extra overhead to to the increased neighbor count or too low which would result in too many iterations to converge.

## Parameters

Shamrock SPH solver implements two tolerance parameters:

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

## Setting the right tolerance

This is in principle not too hard. Using `1.1` result in a moderate excess of neighbor while allowing in most cases the smoothing length to converge in a single coarse cycle during the simulation. If your simulation have very fast advecting components with large density contrast you will see things like
```
Warning: smoothing length is not converged, rerunning the iterator ...                [Smoothinglength][rank=0]
     largest h = 0.8310577409570404 unconverged cnt = 99994  
```
in the logs. If that happens you can try to increase the tolerance to something like `1.15` or `1.2` which should solve the issue at the cost of a slight slowdown.
