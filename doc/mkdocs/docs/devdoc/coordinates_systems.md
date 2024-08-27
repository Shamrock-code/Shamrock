# Coordinate Spaces

## Patch Coordinates

Those represent an abstract coordinate space used by the patch system.
they are vectors of `u64` that evolve in the range `[0,PatchScheduler::max_axis_patch_coord]`

```{warning}
Contrary to other coordinates systems in shamrock they evolve in a closed interval
This is subject to change a some point to avoid inconsistencies in the code
```

They are manipulated using the class `PatchCoord`.

## Object coordinates

Using the Scheduler with the class `SimBox` we can convert Patch coordinates to obj coordinates.
Those are represented on a half open range `[a,b[`

```cpp
// to change the bound of the obj coordinates
sched.set_coord_domain_bound(bmin,bmax);

// to get the current obj coordinates bound

```

### AMR Case

They are represented using half-open intervals of integers
`[0,max_amr[` The only criterion is that `max_amr` must divide `PatchScheduler::max_axis_patch_coord_length`

Using a offset + scalling the AMR integer coordinates can be seen as real floating point coordinates.

### SPH case
