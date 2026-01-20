using HyVR
using Random
using Statistics
using LinearAlgebra
using GLMakie
using Printf
using Distributions
using CUDA

# Ensure CUDA is functional
if !CUDA.functional()
    error("CUDA is not functional. Please check your GPU and drivers.")
end

println("Running on GPU: ", CUDA.device())

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
xs = range(delx/2, length = ncol, step = delx)
ys = range(dely/2, length = nrow, step = dely)
zs = range(H - delz/2, length = nlay, step = -delz)

# Create 3D arrays on CPU first, then move to GPU
x_3d_cpu = zeros(Float64, nlay, nrow, ncol)
y_3d_cpu = zeros(Float64, nlay, nrow, ncol)
z_3d_cpu = zeros(Float64, nlay, nrow, ncol)

for k = 1:nlay, i = 1:nrow, j = 1:ncol
    x_3d_cpu[k, i, j] = xs[j]
    y_3d_cpu[k, i, j] = ys[i]
    z_3d_cpu[k, i, j] = zs[k]
end

# Move to GPU
x_3d = CuArray(x_3d_cpu)
y_3d = CuArray(y_3d_cpu)
z_3d = CuArray(z_3d_cpu)

# Arrays for properties
facies = CUDA.fill(7, nlay, nrow, ncol) # Default facies 7
dip = CUDA.zeros(Float64, nlay, nrow, ncol)
dip_dir = CUDA.zeros(Float64, nlay, nrow, ncol)

# ==============================================================================
# 2. Surface Generation
# ==============================================================================

Random.seed!(37893)

# Top Surface
mean_top = H - 1.86
var_top = 0.7
corl_top = [70.0, 792.0]

# specsim_surface runs on CPU (FFTW dependency)
x_2d_cpu = x_3d_cpu[1, :, :]
y_2d_cpu = y_3d_cpu[1, :, :]

surf_top = specsim_surface(x_2d_cpu, y_2d_cpu, mean_top, var_top, corl_top)

# Bottom Surface
mean_botm = H - 8.3
var_botm = 0.9
corl_botm = [300.0, 900.0]

surf_botm = specsim_surface(x_2d_cpu, y_2d_cpu, mean_botm, var_botm, corl_top)

# ==============================================================================
# 3. Thickness Sequence
# ==============================================================================

simulated_thickness = mean_top - 1.0
n_layers = 8
min_thick = 0.0
thicknesses = Float64[]

while min_thick < 0.3
    global thicknesses
    zs_gen = rand(Uniform(0, simulated_thickness), n_layers - 1)
    ordered_zs = sort(zs_gen)
    push!(ordered_zs, simulated_thickness)
    pushfirst!(ordered_zs, 0.0)
    thicknesses = diff(ordered_zs)
    global min_thick = minimum(thicknesses)
end

println("Thicknesses: ", thicknesses)
println("Total thickness: ", sum(thicknesses))

# ==============================================================================
# 4. Helper data
# ==============================================================================

# Flattened 2D coordinates for distance calc (Row, Col plane)
x_flat = vec(x_3d[1, :, :])
y_flat = vec(y_3d[1, :, :])

# ==============================================================================
# 5. Sedimentary Structure Modeling
# ==============================================================================

z_0 = 0.0

for (idx, thick) in enumerate(thicknesses)
    println("Processing layer $idx with thickness $thick at z0=$z_0")

    # 5.1 Anastamosing channel pattern
    main_channels = []
    channels = []

    # Generate main channels (on CPU)
    for i = 1:6
        ystart = rand(Uniform(0, 600))
        curve_data = ferguson_curve(
            h = 0.3,
            k = π/200,
            eps_factor = (π/1.5)^2,
            flow_angle = 0.0,
            s_max = 1500.0,
            xstart = -500.0,
            ystart = ystart,
        )
        push!(main_channels, curve_data)

        cx, cy = curve_data[1], curve_data[2]
        indices = randperm(length(cx))[1:4]

        for k in indices
            xp, yp = cx[k], cy[k]
            branch_data = ferguson_curve(
                h = 0.3,
                k = π/200,
                eps_factor = (π/1.5)^2,
                flow_angle = rand(Uniform(-π/18, π/18)),
                s_max = 1000.0,
                xstart = xp,
                ystart = yp,
            )
            push!(channels, branch_data)
        end
    end

    total_channels = vcat(main_channels, channels)

    # Calculate min distance to ANY channel (on GPU)
    min_dist_global = CUDA.fill(Inf, length(x_flat))

    for ch in total_channels
        cx, cy = CuArray(ch[1]), CuArray(ch[2])
        dists = compute_min_distance(cx, cy, x_flat, y_flat)
        min_dist_global .= min.(min_dist_global, dists)
    end

    # 5.2 Primitive Facies
    p = sortperm(min_dist_global)
    num_indices = Int(floor(length(p) * 0.2))
    # Note: slicing CuArray returns CuArray
    selected_indices = p[(end-num_indices+1):end]

    primitive_layer = CUDA.fill(7, nrow, ncol)
    primitive_flat = vec(primitive_layer)
    primitive_flat[selected_indices] .= 6
    primitive_layer = reshape(primitive_flat, nrow, ncol)

    # Assign to facies
    # We iterate over k on CPU, but use GPU broadcast assignment
    for k = 1:nlay
        z_val = zs[k]
        if z_val >= z_0
            facies[k, :, :] .= primitive_layer
        end
    end

    println("  Finished primitive layer")

    # 5.3 Ponds
    p_ponds = 0.0
    z_top_curr = z_0 + thick

    while p_ponds < 0.30
        x_c = rand(Uniform(0, 900))
        y_c = rand(Uniform(0, 600))
        z_c = z_top_curr + rand(Uniform(0, 0.1))

        a = rand(Uniform(50, 80))
        b = rand(Uniform(30, 60))
        c = thick
        azim = rand(Uniform(-20, 20))

        half_ellipsoid!(
            facies,
            dip,
            dip_dir,
            x_3d,
            y_3d,
            z_3d,
            (x_c, y_c, z_c),
            (a, b, c),
            azim,
            2,
            internal_layering = false,
        )

        mask = (z_3d .>= z_0) .& (z_3d .<= z_top_curr)
        count_mask = count(mask)
        if count_mask > 0
            # count works on CuArray (moves result to CPU)
            p_ponds = count((facies .== 2) .& mask) / count_mask
        else
            p_ponds = 1.0
        end
    end
    println("  Finished ponds ($p_ponds)")

    # 5.4 Channels
    for ch in main_channels
        cx, cy = ch[1], ch[2]
        curve_mat = CuArray(hcat(cx, cy))

        channel!(
            facies,
            dip,
            dip_dir,
            x_3d,
            y_3d,
            z_3d,
            z_top_curr,
            curve_mat,
            [30.0, thick],
            4,
        )
    end

    for ch in channels
        cx, cy = ch[1], ch[2]
        curve_mat = CuArray(hcat(cx, cy))

        channel!(
            facies,
            dip,
            dip_dir,
            x_3d,
            y_3d,
            z_3d,
            z_top_curr,
            curve_mat,
            [20.0, thick],
            4,
        )
    end
    println("  Finished channels")

    # 5.5 Peat Lenses
    p_peat = 0.0
    c_peat = thick > 0.4 ? 0.4 : thick

    valid_layers = findall(zs .<= z_top_curr)
    if isempty(valid_layers)
        layer_idx = 1
    else
        layer_idx = valid_layers[1]
    end

    # Sampling coordinates on CPU
    # Masking on GPU first
    mask_water = (facies[layer_idx, :, :] .== 2) .| (facies[layer_idx, :, :] .== 4)
    if count(mask_water) > 0
        # Transfer water body coordinates to CPU for sampling
        xs_water_cpu = Array(x_3d[layer_idx, :, :][mask_water])
        ys_water_cpu = Array(y_3d[layer_idx, :, :][mask_water])

        while p_peat < 0.20
            idx = rand(1:length(xs_water_cpu))
            x_c = xs_water_cpu[idx]
            y_c = ys_water_cpu[idx]
            z_c = z_top_curr

            a = rand(Uniform(30, 60))
            b = rand(Uniform(20, 40))
            azim = rand(Uniform(-20, 20))
            f_code = rand([8, 9])

            half_ellipsoid!(
                facies,
                dip,
                dip_dir,
                x_3d,
                y_3d,
                z_3d,
                (x_c, y_c, z_c),
                (a, b, c_peat),
                azim,
                f_code,
            )

            mask_peat_zone = (z_3d .>= z_top_curr - c_peat) .& (z_3d .<= z_top_curr)
            count_peat_zone = count(mask_peat_zone)
            if count_peat_zone > 0
                p_peat = count((facies .== f_code) .& mask_peat_zone) / count_peat_zone
            else
                p_peat = 1.0
            end
        end
    end
    println("  Finished peat ($p_peat)")

    global z_0 += thick
end

# ==============================================================================
# 6. Final Layer
# ==============================================================================

println("Processing final layer...")
min_height = z_0
facies[z_3d .> min_height] .= 10

heights = range(min_height, stop = maximum(surf_top), step = 0.05)

x_c = -rand(Uniform(200, 300))
y_c = rand(Uniform(200, 600))

gravel_channel = nothing
P_0_x = x_flat
P_0_y = y_flat

while true
    global gravel_channel
    gravel_channel_data = ferguson_curve(
        h = 0.3,
        k = π/200,
        eps_factor = π^2,
        flow_angle = 0.0,
        s_max = 1500.0 - x_c,
        xstart = x_c,
        ystart = y_c,
    )
    cx, cy = CuArray(gravel_channel_data[1]), CuArray(gravel_channel_data[2])
    dists = compute_min_distance(cx, cy, P_0_x, P_0_y)
    count_close = count(dists .< 200)

    if count_close >= 200
        gravel_channel = gravel_channel_data
        break
    end
end

println("  Gravel channel generated")

for h_val in heights
    p_tufa = 0.0
    thick = 0.2

    mask_tufa = (z_3d .>= h_val) .& (z_3d .<= h_val + thick)
    if count(mask_tufa) == 0
        continue
    end

    # Calculate dists for all points (or subset). 
    # To optimize, we can filter points on GPU, then calculate distance.
    # For simplicity, we calculate distance on subset.

    x_tufa = x_3d[mask_tufa]
    y_tufa = y_3d[mask_tufa]

    cx, cy = CuArray(gravel_channel[1]), CuArray(gravel_channel[2])
    dists = compute_min_distance(cx, cy, x_tufa, y_tufa)

    valid_indices = findall(dists .< 200.0)

    if isempty(valid_indices)
        continue
    end

    # Bring valid coordinates to CPU for sampling
    valid_x_cpu = Array(x_tufa[valid_indices])
    valid_y_cpu = Array(y_tufa[valid_indices])

    while p_tufa < 0.90
        idx = rand(1:length(valid_x_cpu))
        xt = valid_x_cpu[idx]
        yt = valid_y_cpu[idx]
        zt = h_val + thick

        a = rand(Uniform(60, 90))
        b = rand(Uniform(40, 50))
        c = rand(Uniform(thick, thick + 0.2))
        azim = rand(Uniform(-20, 20))

        half_ellipsoid!(
            facies,
            dip,
            dip_dir,
            x_3d,
            y_3d,
            z_3d,
            (xt, yt, zt),
            (a, b, c),
            azim,
            11,
        )

        current_facies_subset = facies[mask_tufa]
        relevant_facies = current_facies_subset[valid_indices]
        p_tufa = count(relevant_facies .== 11) / length(relevant_facies)
    end

    curve_mat = CuArray(hcat(gravel_channel[1], gravel_channel[2]))
    channel!(
        facies,
        dip,
        dip_dir,
        x_3d,
        y_3d,
        z_3d,
        h_val + thick,
        curve_mat,
        [25.0, thick + 0.2],
        12,
    )

    println("  Finished layer $h_val")
end

# ==============================================================================
# 7. Masking and Export
# ==============================================================================

# Broadcast surfaces
surf_top_gpu = reshape(CuArray(surf_top), 1, nrow, ncol)
surf_botm_gpu = reshape(CuArray(surf_botm), 1, nrow, ncol)

facies[z_3d .>= surf_top_gpu] .= 21
facies[z_3d .<= surf_botm_gpu] .= 31

println("Model generation complete.")

# ==============================================================================
# 8. Plotting
# ==============================================================================

println("Plotting results...")
# Move to CPU for plotting
facies_cpu = Array(facies)

# 2D Cross Sections
mid_k = div(nlay, 2)
mid_i = div(nrow, 2)
mid_j = div(ncol, 2)

fig = Figure(size = (1200, 800))

ax1 = Axis(fig[1, 1], title = "Planar View (Layer $mid_k)", xlabel = "x", ylabel = "y")
hm1 = heatmap!(ax1, xs, ys, transpose(facies_cpu[mid_k, :, :]), colormap = :turbo)

ax2 = Axis(fig[1, 2], title = "Cross-section (Col $mid_j)", xlabel = "y", ylabel = "z")
hm2 = heatmap!(ax2, ys, zs, transpose(facies_cpu[:, :, mid_j]), colormap = :turbo)

ax3 = Axis(fig[2, 1], title = "Longitudinal (Row $mid_i)", xlabel = "x", ylabel = "z")
hm3 = heatmap!(ax3, xs, zs, transpose(facies_cpu[:, mid_i, :]), colormap = :turbo)

Colorbar(fig[1, 3], hm1, label = "Facies")

save("ammer_facies_2d_gpu.png", fig)
println("Saved 2D plots to ammer_facies_2d_gpu.png")
