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
Lx = 50.0  # problem length [m]
Ly = 50.0  # problem width [m]
H = 6.0     # aquifer height [m]
delx = 0.2  # block size x direction
dely = 0.2  # block size y direction
delz = 0.1  # block size z direction

nlay = Int(H / delz)
ncol = Int(Lx / delx)
nrow = Int(Ly / dely)

println("Grid dimensions: ($nlay, $nrow, $ncol)")

# Grid Coordinates (MODFLOW convention: layer, row, col)
# z: decreases from top to bottom
# y: increases (or decreases, usually row index 1 is top Y)
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

x_3d = zeros(Float64, ncol, nrow, nlay)
y_3d = zeros(Float64, ncol, nrow, nlay)
z_3d = zeros(Float64, ncol, nrow, nlay)

for k = 1:nlay, i = 1:nrow, j = 1:ncol
    x_3d[j, i, k] = xs[j]
    y_3d[j, i, k] = ys[i]
    z_3d[j, i, k] = zs[k]
end
grid = RectilinearGrid(xs, ys, zs)

# Arrays for properties
facies = fill(7, ncol, nrow, nlay) # Default facies 7
dip_arr = zeros(Float64, ncol, nrow, nlay)
dip_dir_arr = zeros(Float64, ncol, nrow, nlay)


Random.seed!(37893)

# first 3 layers:

# Bottom Surface
mean_botms = [0.3, 0.6, 0.9]
var_botm = 0.9
corl_botm = [300.0, 900.0]
# specsim_surface expects 2D grid of x and y
# We take the first layer's x and y
x_2d = x_3d[:, :, 1]
y_2d = y_3d[:, :, 1]
surf_botms = [specsim_surface(x_2d, y_2d, mean_botm, var_botm, corl_botm) for mean_botm in mean_botms]

for lay in 1:nlay
    for i in 1:3
        layf = @view facies[:, :, lay]
        layz = @view z_3d[:, :, lay]
        layf[layz .< surf_botms[i] .&& layf .== 7] .= i
    end
end

# rolling back on 3 (to 2)
facies[facies .== 3] .= 2

# now sequence of scour pools and sheets
# events can be accretionary or cut and fill. In any manner cut and fill are objects that
# erode while accretionary just add on top. if we model the accretionary first we are simulating the 
# right 

# another accretionary surfaces:
mean_botms = rand(Uniform(1, 6), 3,)
sort!(mean_botms)
var_botm = 0.9
corl_botm = [300.0, 900.0]
# specsim_surface expects 2D grid of x and y
# We take the first layer's x and y
x_2d = x_3d[:, :, 1]
y_2d = y_3d[:, :, 1]
surf_botms = [specsim_surface(x_2d, y_2d, mean_botm, var_botm, corl_botm) for mean_botm in mean_botms]

for lay in 1:nlay
    for i in 1:3
        layf = @view facies[:, :, lay]
        layz = @view z_3d[:, :, lay]
        layf[layz .< surf_botms[i] .&& layf .== 7] .= 1 #facies GCM(=1)
    end
end

# Now top most layer as well
facies[facies .== 7] .= 1
scour_pools_acc = zeros(size(facies))
# now we simulate scour pools events
spacing = (6-1)/3
noise = spacing/2
evt_zs = 1 .+ [spacing, 2*spacing, 3*spacing] .+
    rand(Uniform(-noise, noise), 3,)
# Clamp to keep inside z domain (roughly 0 to 6)
evt_zs = clamp.(evt_zs, 0.5, 5.5)
sort!(evt_zs)
println("$evt_zs")
for evt_z in evt_zs
    ind = true
    n_vals = size(scour_pools_acc[(evt_z .- 0.3) .< z_3d .< (evt_z .+ 0.3)])[1]
    if n_vals == 0
        println("Warning: No cells in range for evt_z = $evt_z. Skipping.")
        continue
    end
    while (ind)
        x_c = rand(Uniform(0, 50))
        y_c = rand(Uniform(0, 50))
        z_c = evt_z + rand(Uniform(-0.3, 0.3))
        a = rand(Uniform(5, 30))
        b = rand(Uniform(6, 22))
        c = rand(Uniform(1, 3))
        azim = rand(Uniform(-20, 20))
        dip_v = rand(Uniform(4,30))
        dip_dir_v = rand(Uniform(0, 180))
        fs = [3,4,5,6]
        half_ellipsoid!(facies, dip_arr, dip_dir_arr,
            grid, (x_c, y_c, z_c),
            (a, b, c),
            azim,
            fs,
            internal_layering=true,
            layer_dist=rand(Uniform(0.1, 0.3)),
            alternating_facies = true,
            facies_p = [0.6, 0.1, 0.1, 0.2],
            dip = dip_v,
            dip_dir = dip_dir_v,
        )
        # assigning the same object but to scout pool control
        half_ellipsoid!(scour_pools_acc, dip_arr, dip_dir_arr,
            grid, (x_c, y_c, z_c),
            (a, b, c),
            azim,
            fs,
            internal_layering=true,
            layer_dist=0.2,
            alternating_facies = true,
            facies_p = [0.5, 0.1, 0.2, 0.2],
            dip = dip_v,
            dip_dir = dip_dir_v,
        )
        cur_acc = sum(scour_pools_acc[(evt_z .- 0.3) .< z_3d .< (evt_z .+ 0.3)] .!= 0)
        ind = cur_acc/n_vals < 0.5
    end
end


color_facies = Dict(
    1 => "#c2bb7b",
    2 => "#c76481",
    3 => "#ecf80b",
    4 => "#0c9900",
    5 => "#a3fd95",
    6 => "#feba0e",
)

# Create custom colormap
facies_colors = [color_facies[i] for i in 1:6]
cmap = cgrad(facies_colors, 6, categorical=true)

# 2D Cross Sections
idx1 = div(nrow, 4)
idx2 = div(nrow, 3)
idx3 = div(nrow, 2)
idx4 = div(nrow * 3, 4)

fig = Figure(size = (1200, 800))

ax1 = Axis(fig[1, 1], title = "X-Z Cross-section (Row $idx1)", xlabel = "x", ylabel = "z", aspect = DataAspect())
hm1 = heatmap!(ax1, xs, zs, facies[:, idx1, :], colormap = cmap, colorrange=(1, 6))

ax2 = Axis(fig[1, 2], title = "X-Z Cross-section (Row $idx2)", xlabel = "x", ylabel = "z", aspect = DataAspect())
hm2 = heatmap!(ax2, xs, zs, facies[:, idx2, :], colormap = cmap, colorrange=(1, 6))

ax3 = Axis(fig[2, 1], title = "X-Z Cross-section (Row $idx3)", xlabel = "x", ylabel = "z", aspect = DataAspect())
hm3 = heatmap!(ax3, xs, zs, facies[:, idx3, :], colormap = cmap, colorrange=(1, 6))

ax4 = Axis(fig[2, 2], title = "X-Z Cross-section (Row $idx4)", xlabel = "x", ylabel = "z", aspect = DataAspect())
hm4 = heatmap!(ax4, xs, zs, facies[:, idx4, :], colormap = cmap, colorrange=(1, 6))

Colorbar(fig[:, 3], hm1, label = "Facies", ticks = 1:6)


display(fig)

save("heinz_cross_sections_x.png", fig)
println("Saved 2D plots to heinz_cross_sections_x.png")