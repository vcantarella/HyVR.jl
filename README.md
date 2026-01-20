# HyVR.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://vcantarella.github.io/HyVR.jl/dev/)
[![Build Status](https://github.com/vcantarella/HyVR.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/vcantarella/HyVR.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/vcantarella/HyVR.jl/graph/badge.svg?token=AP09UVR42M)](https://codecov.io/gh/vcantarella/HyVR.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Julia implementation of the [HyVR](https://github.com/driftingtides/hyvr) (Hydrogeological Virtual Reality) package. This package simulates sedimentary heterogeneity using an object-based approach, creating 3D grids of hydraulic properties (facies, dip, dip direction) for groundwater modeling.


## Features

*   **Performance**: Built with `KernelAbstractions.jl` to support high-performance parallel execution on both CPUs and GPUs.
*   **Object-Based Simulation**: Supports Troughs (Truncated Ellipsoids), Channels (Extruded Parabolas), and Sheets.
*   **Geostatistical Tools**: Includes Ferguson curve generation for meandering channels and Spectral Simulation (`specsim`) for generating random surfaces.

## Installation

```julia
using Pkg
Pkg.add(path=".") # If installing locally
```

## Device Selection (CPU vs GPU)

HyVR.jl uses `KernelAbstractions.jl` to automatically select the compute device based on the input array type.

*   **CPU**: Pass standard Julia `Array`s (e.g., `zeros(100, 100, 100)`). The kernels will run using multi-threading.
*   **GPU**: Pass GPU arrays (e.g., `CuArray` from `CUDA.jl`, `ROCArray` from `AMDGPU.jl`, or `MtlArray` from `Metal.jl`). The kernels will automatically compile and run on the GPU.

### Example: Running on GPU (CUDA)

```julia
using HyVR, CUDA

# Create arrays on the GPU
dims = (100, 100, 50)
f_array = CUDA.zeros(Int, dims)
dip = CUDA.zeros(Float32, dims)
# ... pass these to HyVR functions ...
```

## Usage Example

Here is a simple example generating a Trough (Half Ellipsoid) with internal layering.

```julia
using HyVR
using KernelAbstractions

# 1. Define the Grid (MODFLOW convention or generic)
nx, ny, nz = 100, 100, 50
f_array = fill(-1, (nx, ny, nz))       # Facies array (Int)
dip = zeros(Float64, (nx, ny, nz))     # Dip angle array
dip_dir = zeros(Float64, (nx, ny, nz)) # Dip direction array

# Create coordinate grids
# Note: For efficiency, you can often use ranges, but for these kernels 
# we currently pass explicit coordinate arrays.
x = [i for i in 1:nx, j in 1:ny, k in 1:nz]
y = [j for i in 1:nx, j in 1:ny, k in 1:nz]
z = [k for i in 1:nx, j in 1:ny, k in 1:nz]

# 2. Define Object Parameters
center = (50.0, 50.0, 25.0)   # Center (x, y, z)
dims = (20.0, 10.0, 5.0)      # Dimensions (a, b, c)
azimuth = 45.0                # Rotation angle
facies_ids = [1, 2]           # Facies to alternate

# 3. Generate the Object
# Note: The '!' indicates this function modifies arrays in-place
half_ellipsoid!(
    f_array, dip, dip_dir, 
    x, y, z, 
    center, dims, azimuth, 
    facies_ids; 
    internal_layering=true, 
    bulb=true, 
    layer_dist=2.0
)

# 4. Inspect Results
println("Number of cells modified: ", count(x -> x != -1, f_array))
```

## Testing

Run the test suite to verify object generation and geostatistical tools:

```bash
julia --project=. test/runtests.jl
```