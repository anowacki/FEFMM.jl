# FEFMM.jl

Compute distances (or travel times) on regular grids using the factored
eikonal fast marching method, as described by Treister & Haber (2016).

## Quick example
Here we set up a slowness-squared grid of points (all with value 1)
each 1 distance unit apart, then calculate the travel time between
all points and the source at index (1, 1).

```julia
using FEFMM
k2 = ones(100, 100) # slowness squared
dx = (1.0, 1.0) # grid spacing
x0 = CartesianIndex(1, 1) # source location index
t = fefmm(k2, dx, x0) # travel time grid at slowness-square grid point
```

## User guide
The package exports three things:
- `fefmm` for single computations,
- `fefmm!` for repeat computations on the same grid, and
- `FEFMMGrid` for creating reusable computation grids.

For one-shot cacluations, you can pass the slowness-squared grid,
grid spacing and source index to `fefmm` as in the quick example above.
However, if you need to find the travel time field on the same grid multiple
times—even if the source is in a different location—less memory is allocated
by first creating a `FEFMMGrid`, then passing that to `fefmm!` each time:

```julia
grid = FEFMMGrid(k2, dx)
t = fefmm!(grid, x0)
t_max1 = maximum(t)
fefmm!(grid, (50, 50)) # New source but no more memory needed
maximum(t) # Note new values within `t`
```

**Note:** The `t` returned by `fefmm!` is an array within the `grid`
object, so its contents are updated on each call to `fefmm!`.  Remember to
extract the information you need out of the returned array before calling
`fefmm!` again.

## Acknowledgments
The original version of this code is taken from Jack Muir's
https://github.com/jbmuir/FEFMM.jl, which in turn was adapted from
https://github.com/JuliaInv/FactoredEikonalFastMarching.jl.  I am grateful to
all authors for making the code available under the MIT licence.
Original licence documents are distributed within `LICENCES`.

## References
- Eran Treister and Eldad Haber, 2016,
  A fast marching algorithm for the factored eikonal equation,
  *Journal of Computational Physics*, **324**, 210–225.
