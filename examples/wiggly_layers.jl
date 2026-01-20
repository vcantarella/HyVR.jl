using HyVR
using Random
using Statistics
using LinearAlgebra
using GLMakie
using Printf
using Distributions

# ==============================================================================
# 1. Grid/Model Creation
# ==============================================================================

# Grid properties
Lx = 900.0  # problem length [m]
Ly = 600.0  # problem width [m]
H = 9.0     # aquifer height [m]
delx = 1.5  # block size x direction
dely = 1.5  # block size y direction
delz = 0.2  # block size z direction

nlay = Int(H / delz)
ncol = Int(Lx / delx)
nrow = Int(Ly / dely)

println("Grid dimensions: ($nlay, $nrow, $ncol)")

# Grid Coordinates (MODFLOW convention: layer, row, col)
# z: decreases from top to bottom
# y: decreases, usually row index 1 is top Y
# x: increases

# Center points
xs = range(delx/2, length = ncol, step = delx)
# Assuming y starts at 0 and goes to Ly, row 1 corresponds to y=0+dely/2
ys = range(dely/2, length = nrow, step = dely)
zs = range(H - delz/2, length = nlay, step = -delz)

# Create 3D arrays (Layer, Row, Col) -> (z, y, x)
# Julia arrays are column-major, but we stick to the logical indexing (k, i, j)
# x_3d[k, i, j] = xs[j]
# y_3d[k, i, j] = ys[i]
# z_3d[k, i, j] = zs[k]

x_3d = zeros(Float64, nlay, nrow, ncol)
y_3d = zeros(Float64, nlay, nrow, ncol)
z_3d = zeros(Float64, nlay, nrow, ncol)

for k = 1:nlay, i = 1:nrow, j = 1:ncol
    x_3d[k, i, j] = xs[j]
    y_3d[k, i, j] = ys[i]
    z_3d[k, i, j] = zs[k]
end

# Arrays for properties
facies = fill(2, nlay, nrow, ncol) # Default facies 7
dip = zeros(Float64, nlay, nrow, ncol)
dip_dir = zeros(Float64, nlay, nrow, ncol)

# ==============================================================================
# 2. Surface Generation
# ==============================================================================

Random.seed!(37893)

# Top Surface
mean_top = H - 1.86
var_top = 0.7
corl_top = [70.0, 60.0]

# specsim_surface expects 2D grid of x and y
# We take the first layer's x and y
x_2d = x_3d[1, :, :]
y_2d = y_3d[1, :, :]

surf_top = specsim_surface(x_2d, y_2d, mean_top, var_top, corl_top)

# Bottom Surface
mean_botm = H - 8.3
var_botm = 0.9
corl_botm = [300.0, 300.0]

surf_botm = specsim_surface(x_2d, y_2d, mean_botm, var_botm, corl_top) # Using corl_top as per python script? Or typo in python? Python says `corl=corl_top`.

for j in axes(z_3d, 1) # number of layers:
    top_index = z_3d[j, :, :] .> surf_top
    local_facies = @view facies[j, :, :]
    local_facies[top_index] .= 1
    bottom_index = z_3d[j, :, :] .< surf_botm
    local_facies[bottom_index] .= 3
end

# ==============================================================================
# 2. Plot surface results
# ==============================================================================
println("Plotting results...")

# 2D Cross Sections
# Middle indices
mid_k = div(nlay, 2)
mid_i = div(nrow, 2)
mid_j = div(ncol, 2)

fig = Figure(size = (1200, 800))

# Planar view (Middle Z)
ax1 = Axis(fig[1, 1], title = "Planar View (Layer $mid_k)", xlabel = "x", ylabel = "y")
hm1 = heatmap!(ax1, xs, ys, transpose(facies[mid_k, :, :]), colormap = :turbo)

# Cross-section (Middle Col -> Y-Z plane)
# facies[:, :, mid_j] is (nlay, nrow) -> (z, y)
# The facies array needs to be transposed for Makie.
ax2 = Axis(fig[1, 2], title = "Cross-section (Col $mid_j)", xlabel = "y", ylabel = "z")
hm2 = heatmap!(ax2, ys, zs, transpose(facies[:, :, mid_j]), colormap = :turbo)

# Longitudinal (Middle Row -> X-Z plane)
ax3 = Axis(fig[2, 1], title = "Longitudinal (Row $mid_i)", xlabel = "x", ylabel = "z")
hm3 = heatmap!(ax3, xs, zs, transpose(facies[:, mid_i, :]), colormap = :turbo)

Colorbar(fig[1, 3], hm1, label = "Facies")

display(fig)
