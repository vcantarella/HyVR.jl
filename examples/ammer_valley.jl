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

x_3d = zeros(Float64, nlay, nrow, ncol)
y_3d = zeros(Float64, nlay, nrow, ncol)
z_3d = zeros(Float64, nlay, nrow, ncol)

for k = 1:nlay, i = 1:nrow, j = 1:ncol
    x_3d[k, i, j] = xs[j]
    y_3d[k, i, j] = ys[i]
    z_3d[k, i, j] = zs[k]
end

# Arrays for properties
facies = fill(7, nlay, nrow, ncol) # Default facies 7
dip = zeros(Float64, nlay, nrow, ncol)
dip_dir = zeros(Float64, nlay, nrow, ncol)

# ==============================================================================
# 2. Surface Generation
# ==============================================================================

Random.seed!(37893)

# Top Surface
mean_top = H - 1.86
var_top = 0.7
corl_top = [70.0, 792.0]

# specsim_surface expects 2D grid of x and y
# We take the first layer's x and y
x_2d = x_3d[1, :, :]
y_2d = y_3d[1, :, :]

surf_top = specsim_surface(x_2d, y_2d, mean_top, var_top, corl_top)

# Bottom Surface
mean_botm = H - 8.3
var_botm = 0.9
corl_botm = [300.0, 900.0]

surf_botm = specsim_surface(x_2d, y_2d, mean_botm, var_botm, corl_top) # Using corl_top as per python script? Or typo in python? Python says `corl=corl_top`.

# ==============================================================================
# 3. Thickness Sequence
# ==============================================================================

simulated_thickness = mean_top - 1.0
n_layers = 8
min_thick = 0.0
thicknesses = Float64[]

# Generate thicknesses
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
# 4. Helper Function: min_distance
# ==============================================================================

# Flattened 2D coordinates for distance calc (Row, Col plane)
# In Python script: X[0,:,:].ravel() -> flattened 2D plane
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

    # Generate main channels
    for i = 1:6
        ystart = rand(Uniform(0, 600))
        # ferguson_curve returns tuple (x, y, vx, vy, s)
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

        # Branching channels
        cx, cy = curve_data[1], curve_data[2]
        # Random indices
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

    # Calculate min distance to ANY channel
    # This effectively creates a distance map
    # We need the minimum across all channels for each point

    # To save time, we can concat all curves?
    # Python script loops channels and stores in (N_points, N_channels) array, then takes min(axis=1)

    min_dist_global = fill(Inf, length(x_flat))
    dists = similar(x_flat)

    for ch in total_channels
        cx, cy = ch[1], ch[2]
        compute_min_distance!(dists, cx, cy, x_flat, y_flat)
        min_dist_global .= min.(min_dist_global, dists)
    end

    # 5.2 Primitive Facies (Channels vicinity)
    # Select last 20% of sorted indices (largest distances? No, min_distance_array.min(axis=1))
    # Python: sorted_indices = np.argsort(min_arr)
    # num_indices = int(len * 0.2)
    # selected = sorted_indices[-num_indices:] -> These are largest distances (furthest from channels)
    # primitive[selected] = 6

    # Wait, Python script logic:
    # min_arr = min_distance_array.min(axis=1) (Distance to nearest channel)
    # argsort sorts ascending (small dist to large dist)
    # selected_indices = sorted_indices[-num_indices:] (Largest distances)
    # primitive[selected] = 6 (Facies 6 away from channels)
    # primitive defaults to 7 (Facies 7 close to channels?)
    # Python: primitive = np.ones... * 7. primitive[selected] = 6.
    # So: Close to channels = 7, Far = 6.

    p = sortperm(min_dist_global)
    num_indices = Int(floor(length(p) * 0.2))
    selected_indices = p[(end-num_indices+1):end]

    primitive_layer = fill(7, nrow, ncol) # Default 7
    # Flat indexing mapping to 2D
    primitive_flat = vec(primitive_layer)
    primitive_flat[selected_indices] .= 6
    primitive_layer = reshape(primitive_flat, nrow, ncol)

    # Assign to facies array for current z interval
    # z_ is column of z coordinates.
    # We check each layer k.

    # z_3d[k, i, j]. z varies by k.
    for k = 1:nlay
        z_val = zs[k]
        if z_val >= z_0
            facies[k, :, :] .= primitive_layer
        end
    end

    println("  Finished primitive layer")

    # 5.3 Ponds (Troughs)
    p_ponds = 0.0
    # Logic mask for current layer thickness
    # In Julia, we iterate or use logical indexing
    # We need to know which cells are in [z0, z0 + thick]

    # Current top for this sequence
    z_top_curr = z_0 + thick

    while p_ponds < 0.30
        x_c = rand(Uniform(0, 900))
        y_c = rand(Uniform(0, 600))
        z_c = z_top_curr + rand(Uniform(0, 0.1))

        a = rand(Uniform(50, 80))
        b = rand(Uniform(30, 60))
        c = thick
        azim = rand(Uniform(-20, 20))

        # Facies 2 for ponds
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
            2, # Facies ID
            internal_layering = false,
        )

        # Calculate proportion
        # Cells in range
        mask = (z_3d .>= z_0) .& (z_3d .<= z_top_curr)
        count_mask = count(mask)
        if count_mask > 0
            p_ponds = count((facies .== 2) .& mask) / count_mask
        else
            p_ponds = 1.0 # break
        end
    end
    println("  Finished ponds ($p_ponds)")

    # 5.4 Channels
    # Main channels
    for ch in main_channels
        cx, cy = ch[1], ch[2]
        # Convert curve to matrix Nx2
        curve_mat = hcat(cx, cy)

        channel!(
            facies,
            dip,
            dip_dir,
            x_3d,
            y_3d,
            z_3d,
            z_top_curr, # z_top
            curve_mat,
            [30.0, thick], # width, depth
            4, # Facies
        )
    end

    # Branch channels
    for ch in channels
        cx, cy = ch[1], ch[2]
        curve_mat = hcat(cx, cy)

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

    # Identify water bodies (2 or 4) in the *lowest* layer of this thickness interval?
    # Python: layer = np.min(ind[0]) where ind is z <= z0+thick. 
    # z decreases in index? No, Python z 0 is top usually in FLOPY if not flipped.
    # In my grid z[1] is top.
    # We want the layer at z <= z0 + thick.
    # Since zs is descending, we want indices where z <= value.
    # The lowest physical z corresponds to higher index.
    # Python script seems to pick a reference 2D slice to sample coordinates.

    # Let's pick a layer index roughly in the middle of the interval or top.
    # Python: `layer = np.min(ind[0])`. `ind` is where `z <= z0+thick`.
    # Since z is likely bottom-up in that specific python array? 
    # Python script: `z = np.flip(z, axis=0)`. `zs` was `0..20`. Flipped means `20..0`.
    # `ind` is where `z <= current_top`. This includes everything below current top.
    # `min(ind)` would be the index corresponding to `current_top` (since z decreases with index).
    # So `layer` is the top-most layer index of the current sediment package.

    valid_layers = findall(zs .<= z_top_curr)
    if isempty(valid_layers)
        layer_idx = 1
    else
        layer_idx = valid_layers[1] # First index is highest z
    end

    # Get coordinates where facies is 2 or 4
    mask_water = (facies[layer_idx, :, :] .== 2) .| (facies[layer_idx, :, :] .== 4)
    if count(mask_water) > 0
        xs_water = x_3d[layer_idx, :, :][mask_water]
        ys_water = y_3d[layer_idx, :, :][mask_water]

        while p_peat < 0.20
            idx = rand(1:length(xs_water))
            x_c = xs_water[idx]
            y_c = ys_water[idx]
            z_c = z_top_curr

            a = rand(Uniform(30, 60))
            b = rand(Uniform(20, 40))
            # c = c_peat
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

            # Logic for peat
            # z >= z0 + thick - c
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
# 6. Final Layer (Top Deterministic)
# ==============================================================================

println("Processing final layer...")
min_height = z_0
# Assign 10 above min_height
facies[z_3d .> min_height] .= 10

heights = range(min_height, stop = maximum(surf_top), step = 0.05)

# Initial channel
x_c = -rand(Uniform(200, 300))
y_c = rand(Uniform(200, 600))

# We need a channel that has at least 200 points within the grid?
# Python script logic ensures `size_indexes < 200` loop.
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
    # Check coverage
    cx, cy = gravel_channel_data[1], gravel_channel_data[2]
    dists = similar(P_0_x)
    compute_min_distance!(dists, cx, cy, P_0_x, P_0_y)
    count_close = count(dists .< 200) # What distance threshold? Python uses values < 200? No, indices. 
    # Python: indexes = np.where(min_distance_arr < 200). 
    # Wait, 200 is the distance threshold? Or count?
    # "while size_indexes < 200": size_indexes is xs.shape[0].
    # So we want at least 200 grid points within distance 200? Or distance < something?
    # Python: `np.where(min_distance_arr < 200)`. This implies distance threshold is 200 units.

    if count_close >= 200
        gravel_channel = gravel_channel_data
        break
    end
end

println("  Gravel channel generated")

# Loop heights
for h_val in heights
    # Python: "adding peat lenses"
    p_tufa = 0.0
    thick = 0.2

    mask_tufa = (z_3d .>= h_val) .& (z_3d .<= h_val + thick)
    if count(mask_tufa) == 0
        continue
    end

    # Only points close to channel
    # Python re-calculates distance for the subset X[logic_tufa].
    # We can use global pre-calc if optimized, but here we filter coordinates.

    x_tufa = x_3d[mask_tufa]
    y_tufa = y_3d[mask_tufa]

    cx, cy = gravel_channel[1], gravel_channel[2]
    dists = similar(x_tufa)
    compute_min_distance!(dists, cx, cy, x_tufa, y_tufa)

    # Indices in the subset where dist < 200
    valid_indices = findall(dists .< 200.0)

    if isempty(valid_indices)
        continue
    end

    # We need coordinates to sample centers
    valid_x = x_tufa[valid_indices]
    valid_y = y_tufa[valid_indices]

    while p_tufa < 0.90
        idx = rand(1:length(valid_x))
        xt = valid_x[idx]
        yt = valid_y[idx]
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

        # Check proportion in the valid area
        # We need to re-extract facies values for the specific indices
        # Optimization: Just check global count? Python does specific check.

        current_facies_subset = facies[mask_tufa] # Flattened
        relevant_facies = current_facies_subset[valid_indices]
        p_tufa = count(relevant_facies .== 11) / length(relevant_facies)
        # println(p_tufa)
    end

    # Add channel facies
    curve_mat = hcat(cx, cy)
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

# Mask top and bottom
# Python: facies[Z >= surf_top] = 21
# Python: facies[Z <= surf_botm] = 31

# Need to expand surf_top (2D) to 3D check
# z_3d[k,i,j] compared to surf_top[i,j] (Careful with indices!)
# x_3d, y_3d derived from xs, ys.
# surf_top calculated from x_2d, y_2d which matches i,j of 3D grid.

for k = 1:nlay, i = 1:nrow, j = 1:ncol
    if z_3d[k, i, j] >= surf_top[i, j]
        facies[k, i, j] = 21
    end
    if z_3d[k, i, j] <= surf_botm[i, j]
        facies[k, i, j] = 31
    end
end

println("Model generation complete.")

# ==============================================================================
# 8. Plotting
# ==============================================================================

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

save("ammer_facies_2d.png", fig)
println("Saved 2D plots to ammer_facies_2d.png")

# 3D Visualization (Subsample for performance)
# Scatter plot is heavy for 1M+ points.
# Let's plot surface points or a subsample.

fig3d = Figure(size = (800, 600))
ax3d = Axis3(fig3d[1, 1], title = "3D Facies Model")

# Downsample factor
ds = 4
x_sub = vec(x_3d[1:ds:end, 1:ds:end, 1:ds:end])
y_sub = vec(y_3d[1:ds:end, 1:ds:end, 1:ds:end])
z_sub = vec(z_3d[1:ds:end, 1:ds:end, 1:ds:end])
f_sub = vec(facies[1:ds:end, 1:ds:end, 1:ds:end])

# Filter out empty/background if needed, but let's show all
mask_valid = f_sub .!= -1
scatter!(
    ax3d,
    x_sub[mask_valid],
    y_sub[mask_valid],
    z_sub[mask_valid],
    color = f_sub[mask_valid],
    colormap = :turbo,
    markersize = 3,
)

save("ammer_facies_3d.png", fig3d)
println("Saved 3D plot to ammer_facies_3d.png")
