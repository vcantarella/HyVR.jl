module HyVR

using KernelAbstractions

# Submodules
include("utils.jl")
include("tools.jl")
include("grid_interface.jl")

# Objects
include("objects/trough.jl")
include("objects/sheet.jl")
include("objects/channel.jl")

using .Utils
using .Tools
using .Grids
using .Trough
using .Sheet
using .Channel


export half_ellipsoid!, sheet!, channel!
export ferguson_curve, specsim_surface, contact_surface, compute_min_distance!
export HyVRGrid, UniformGrid, RectilinearGrid, StructuredGrid, UnstructuredGrid
export get_xyz, grid_size
end # module
