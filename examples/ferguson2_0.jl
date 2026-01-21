using DifferentialEquations
using GLMakie
using LinearAlgebra
using DataInterpolations


function ferguson_curve(;
    h::Float64,
    k::Float64,
    ϵ::Float64,
    θ::Float64,
    sₘ::Float64,
    xstart::Float64,
    ystart::Float64,
    extra_noise::Float64 = 0.0,
)

    function ferguson_drift!(du, u, p, s)
        k, h, ϵ = p

        θ, ω = u
        du[1] = ω
        du[2] = -k^2*θ - 2*h*k*ω
    end
    
    function ferguson_diffusion!(du, u, p, s)
        k, h, ϵ = p
        θ, ω = u
        du[1] = 0.0
        du[2] = ϵ*k^2
    end

    u0 = [0.0, 0.0]
    tspan = (0, sₘ)
    p = (k, h, ϵ)
    prob = SDEProblem(ferguson_drift!, ferguson_diffusion!, u0, tspan, p)
    sol = solve(prob, reltol = 1e-4, abstol = 1e-4)
    # Extract solution
    θs = sol[1,:]
    s = sol.t
    ds = diff(s)
    vx = cos.(θs)
    vy = sin.(θs)
    dx = vx[1:end-1].*ds
    dy = vy[1:end-1].*ds
    x = cumsum(dx)
    y = cumsum(dy)
    x = [[0.0];x]
    y = [[0.0];y]

    # Rotation
    rot_angle = θ
    cos_r = cos(rot_angle)
    sin_r = sin(rot_angle)

    # Rotate coordinates
    x_rot = x .* cos_r .- y .* sin_r
    y_rot = x .* sin_r .+ y .* cos_r

    # Rotate velocities
    vx_rot = vx .* cos_r .- vy .* sin_r
    vy_rot = vx .* sin_r .+ vy .* cos_r

    # Translate
    x_final = x_rot .+ xstart
    y_final = y_rot .+ ystart

    return (x_final, y_final, vx_rot, vy_rot, s)
end

"""
    apply_cutoffs(x, y, s, width)

Scans the channel for neck cutoffs. If the Euclidean distance between two points
is less than `width`, but they are separated by a sufficient path length (arc length),
the intermediate loop is removed.
"""
function apply_cutoffs(x::Vector{Float64}, y::Vector{Float64}, s::Vector{Float64}, width::Float64)
    # Working copies of the arrays
    x_new = copy(x)
    y_new = copy(y)
    s_new = copy(s)
    
    # We must restart the scan after every cut because indices shift
    clean_pass = false
    
    # Threshold for "path distance" to prevent detecting the bend itself as a cutoff.
    # We only care if points are close in space but far apart along the river.
    # A safe heuristic is a few channel widths.
    arc_length_threshold = 5.0 * width 
    
    while !clean_pass
        clean_pass = true
        n = length(x_new)
        
        # Iterate through points
        for i in 1:n-1
            # Optimization: Only check points j that are far enough downstream
            # We can skip ahead based on arc_length_threshold to save time
            # But simple loop is fine for N < 10,000
            for j in (i + 1):n
                
                # 1. Check Path Distance (Arc Length)
                path_dist = s_new[j] - s_new[i]
                if path_dist < arc_length_threshold
                    continue 
                end
                
                # 2. Check Spatial Distance (Neck Width)
                # Using squared distance is faster (avoids sqrt)
                spatial_dist_sq = (x_new[i] - x_new[j])^2 + (y_new[i] - y_new[j])^2
                
                if spatial_dist_sq < (width*2)^2
                    # CUTOFF DETECTED! 
                    # Everything between i and j is the "loop" (oxbow).
                    # We keep 1:i and connect it to j:end.
                    
                    # Update arrays by splicing
                    x_new = [x_new[1:i]; x_new[j:end]]
                    y_new = [y_new[1:i]; y_new[j:end]]
                    
                    # We must also re-calculate 's' (arc length) for the new path
                    # because we just deleted a chunk of distance.
                    # The chunk [j:end] shifts "backwards" in s-space.
                    deleted_length = s_new[j] - s_new[i]
                    s_tail = s_new[j:end] .- deleted_length
                    s_new = [s_new[1:i]; s_tail]
                    
                    clean_pass = false # Restart scan on modified arrays
                    break 
                end
            end
            if !clean_pass break end
        end
    end
    
    return x_new, y_new, s_new
end

# --- 1. Calculate Curvature and Normal Vectors ---
function get_geometry(x, y)
    # Central differences for derivatives
    # dx/ds and dy/ds
    xp = similar(x); yp = similar(y)
    
    # Endpoints (forward/backward difference)
    xp[1] = x[2]-x[1]; yp[1] = y[2]-y[1]
    xp[end] = x[end]-x[end-1]; yp[end] = y[end]-y[end-1]
    
    # Interior (central difference)
    for i in 2:length(x)-1
        xp[i] = (x[i+1] - x[i-1]) / 2.0
        yp[i] = (y[i+1] - y[i-1]) / 2.0
    end

    # Arc length calculation
    ds = sqrt.(xp.^2 .+ yp.^2)
    
    # Curvature (k = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5)
    # We need second derivatives
    xpp = similar(x); ypp = similar(y)
    xpp[1] = xp[2]-xp[1]; ypp[1] = yp[2]-yp[1]
    xpp[end] = xp[end]-xp[end-1]; ypp[end] = yp[end]-yp[end-1]
    
    for i in 2:length(x)-1
        xpp[i] = (xp[i+1] - xp[i-1]) / 2.0
        ypp[i] = (yp[i+1] - yp[i-1]) / 2.0
    end

    curvature = (xp .* ypp .- yp .* xpp) ./ (ds.^3 .+ 1e-12)
    
    # Normal vectors (pointing towards outer bank)
    # If tangent is (dx, dy), normal is (-dy, dx) normalized
    nx = yp ./ ds
    ny =  -xp ./ ds
    
    return curvature, nx, ny, ds
end

# --- 2. Resampling (Crucial for stability) ---
# Keeps points evenly spaced as the channel stretches
function resample_centerline(x, y, target_ds)
    # Calculate cumulative distance along the current path
    dists = sqrt.(diff(x).^2 .+ diff(y).^2)
    s_cum = [0.0; cumsum(dists)]
    total_len = s_cum[end]
    
    # --- SAFETY CHECK ---
    # If the remaining channel is shorter than one step, or collapsed to a point,
    # force at least 2 points to prevent the simulation from crashing.
    if total_len < target_ds
        # Just return the start and end points
        return [x[1], x[end]], [y[1], y[end]]
    end
    # --------------------

    # Create new even s array
    s_new = collect(0.0:target_ds:total_len)
    
    # Ensure the last point is included if it's not exactly on the grid
    # (Optional, but good for keeping the exact end-point)
    if s_new[end] < total_len
        push!(s_new, total_len)
    end
    
    # Interpolate
    itp_x = LinearInterpolation(x, s_cum; extrapolation = ExtrapolationType.Linear)
    itp_y = LinearInterpolation(y, s_cum; extrapolation = ExtrapolationType.Linear)
    
    return itp_x(s_new), itp_y(s_new)
end

# A "Deep" Erodibility Field (Simulating Heterogeneous Floodplain)
# Finotello et al. highlight that substrate variability drives complex shapes.
function get_erodibility(x, y, base_E)
    # Simple Perlin-like noise or random patches
    # High values = Sand (Fast erosion), Low values = Clay/Vegetation (Slow)
    noise = sin(x/100) * cos(y/100) 
    return base_E * (1.0 + 0.5 * noise) 
end

function hk_migrate(x, y, dt, Width, E, Lag_Coeff)
    # 1. Get Geometry
    C, nx, ny, ds_local = get_geometry(x, y)
    
    # 2. Calculate Weighted Curvature (The H&K Lag)
    # R_eff[i] represents the near-bank velocity perturbation
    # We treat Lag_Coeff as the "friction factor" (characteristic lag distance)
    # Typically Lag Distance ~ 3 to 5 Channel Widths
    
    R_eff = zeros(Float64, length(C))
    
    # We iterate downstream to accumulate the lag effect
    for i in 2:length(C)
        dist = sqrt((x[i]-x[i-1])^2 + (y[i]-y[i-1])^2)
        
        # This is the discrete solution to the convolution integral
        # Weight decays exponentially with distance
        decay = exp(-dist / (Lag_Coeff * Width))
        
        # Current Velocity = Upstream Velocity * decay + Local Curvature
        R_eff[i] = R_eff[i-1] * decay + C[i] * (1.0 - decay)
    end
    local_E = get_erodibility.(x, y, E)
    # 3. Calculate Migration Vector
    # Migration = Erodibility * Weighted_Curvature
    # E is combined with dt here
    migration_dist = local_E .* R_eff .* dt
    
    # 4. Move Coordinates
    x_new = x .+ nx .* migration_dist
    y_new = y .+ ny .* migration_dist
    
    # Pin the inlet point (boundary condition)
    x_new[1] = x[1]
    y_new[1] = y[1]
    
    return x_new, y_new
end

# Physical Parameters
W = 4.0                # Channel Width
target_ds = 6.0        # Resampling resolution (W/2)

# Migration Parameters
dt = 1.0                # Reduced timestep for stability (was 5.0)
n_steps = 400           # More steps to compensate for lower dt
Base_Erodibility = 30.0 # Migration rate
Lag_Factor = 2.5        # 2.5 Widths (Classic value for Howard model)

# Initialize (Ferguson)
data = ferguson_curve(h=.3, k=π/(12*W), ϵ=5.0, θ=0.0, sₘ=5000.0, xstart=0.0, ystart=0.0)
x, y = data[1], data[2]
x, y, _ = apply_cutoffs(x, y, data[5], W)
x, y = resample_centerline(x, y, target_ds)

# Visualization
fig = Figure(size=(1000, 600))
ax = Axis(fig[1, 1], title="Corrected Kinematic Migration")
lines!(ax, x, y, color=(:black, 0.3), label="Initial")

# Run Simulation
for t in 1:n_steps
    global x, y
    
    # 1. Migrate (using corrected Outer Bank logic)
    x, y = hk_migrate(x, y, dt, W, Base_Erodibility, Lag_Factor)
    
    # 2. Cutoffs
    # (Assuming you calculate s approx)
    s_approx = [0.0; cumsum(sqrt.(diff(x).^2 .+ diff(y).^2))]
    x, y, _ = apply_cutoffs(x, y, s_approx, W)
    
    # 3. Resample (Prevents node collapse)
    x, y = resample_centerline(x, y, target_ds)
    
    if t % 50 == 0
        lines!(ax, x, y, color=(:blue, t/n_steps), linewidth=1.5)
    end
end

lines!(ax, x, y, color=:red, linewidth=3, label="Final")
axislegend(ax)
display(fig)

gravel_channel_data = ferguson_curve(
        h = 0.3,
        k = π/200,
        ϵ = π^2,
        θ = 0.0,
        sₘ = 1500.0,
        xstart = 0.0,
        ystart = 0.0,
    )

# 2. Define Physical Scale
channel_width = 40.0 # Based on W ≈ L/10
x = gravel_channel_data[1]
y = gravel_channel_data[2]
s = gravel_channel_data[5]
# 3. Apply Cutoff Logic
x_cut, y_cut, s_cut = apply_cutoffs(x, y, s, channel_width)

fig = Figure()
ax = Axis(fig[1, 1], title="Planar View - Topology", xlabel="x", ylabel="y")

lines!(x, y, color = (:grey, 0.4), 
    linewidth = 1, 
    linestyle = :dash, 
    label = "Pre-cutoff (Oxbows)")

lines!(ax, x_cut, y_cut, 
    color = :blue, 
    linewidth = 3, 
    label = "Active Channel")

# Plot the cutoff threshold for scale
scatter!(ax, [x_cut[end]], [y_cut[end]], 
    markersize = channel_width, 
    color = (:red, 0.2), 
    strokewidth = 1, 
    strokecolor = :red,
    label = "Channel Width ($channel_width)")
axislegend(ax)
fig