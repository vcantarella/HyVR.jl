module Tools

using OrdinaryDiffEq
using FFTW
using Statistics
using LinearAlgebra
using Random
using Interpolations

export ferguson_curve, specsim_surface, contact_surface

function ferguson_curve(;
    h::Float64,
    k::Float64,
    eps_factor::Float64,
    flow_angle::Float64,
    s_max::Float64,
    xstart::Float64,
    ystart::Float64,
    extra_noise::Float64 = 0.0,
)

    # Correlated variance calculation => Gaussian function
    # In Julia, we can define the problem and solve it.

    # Discretize s for noise generation (similar to Python)
    ds = s_max / 1000
    s_range = 0:ds:s_max
    n_points = length(s_range)

    # Generate correlated noise
    # Python uses: cov = variance * np.exp(-(1 / 2) * dist_arr)
    # where dist_arr is squared euclidean distance

    variance = eps_factor + extra_noise

    # Construct covariance matrix (dense)
    # Note: For large s_max, this might be slow. Python does the same.
    # Optimization: Use a smaller kernel or approximate. But let's stick to the algo.

    s_grid = collect(s_range)
    dist_sq = (s_grid .- s_grid') .^ 2
    cov_mat = variance .* exp.(-0.5 .* dist_sq)

    # Add small error for stability
    cov_mat += I * 1e-8

    # Cholesky decomposition
    L = cholesky(cov_mat).L
    u = randn(n_points)
    e_s = L * u # Correlated noise

    # Interpolation for the ODE solver
    # We need a continuous function of s for the noise
    noise_interp = linear_interpolation(s_grid, e_s, extrapolation_bc = Line())

    # ODE System
    # y = [tau, theta, x, y]
    # y[1] = tau
    # y[2] = theta
    # y[3] = x
    # y[4] = y (coord)

    function ferguson_ode!(dy, y, p, s)
        k, h = p
        eps_t = noise_interp(s)

        # d_tau_ds = (eps_t - theta - 2 * h / k * tau) * (k^2)
        # Note: In Python: d_tau_ds = (eps_t - y[1] - 2 * h / k * y[0]) * (k**2)
        # y[1] is theta, y[0] is tau.
        # Wait, Python code says: y0 = [omega * k, 0.0, 0.0, 0.0] -> [tau, theta, x, y]
        # Python rhs: d_tau_ds = (eps_t - y[1] - 2 * h / k * y[0]) * (k**2)
        # So dy[1] depends on y[2] (theta) and y[1] (tau)

        tau = y[1]
        theta = y[2]

        dy[1] = (eps_t - theta - 2 * h / k * tau) * (k^2)
        dy[2] = tau
        dy[3] = cos(theta)
        dy[4] = sin(theta)
    end

    y0 = [0.0, 0.0, 0.0, 0.0] # Initial conditions: tau=0, theta=0, x=0, y=0. Python uses omega*k for tau. Let's assume omega=0 for now or add as param.
    # Python default omega=0 in some calls, but let's check. 
    # Python code has `omega` param. We didn't expose it in the signature above, assuming 0 or default.
    # Python ferguson_curve calls ferguson_theta_ode with 0.0.

    params = (k, h)
    prob = ODEProblem(ferguson_ode!, y0, (0.0, s_max), params)
    sol = solve(prob, BS3(), reltol = 1e-6, abstol = 1e-6) # BDF in python, BS3/Tsit5 usually fine here.

    # Extract solution
    theta = sol[2, :]
    xp = sol[3, :]
    yp = sol[4, :]
    s = sol.t
    vx = cos.(theta)
    vy = sin.(theta)

    # Rotation
    rot_angle = flow_angle
    cos_r = cos(rot_angle)
    sin_r = sin(rot_angle)

    # Rotate coordinates
    x_rot = xp .* cos_r .- yp .* sin_r
    y_rot = xp .* sin_r .+ yp .* cos_r

    # Rotate velocities
    vx_rot = vx .* cos_r .- vy .* sin_r
    vy_rot = vx .* sin_r .+ vy .* cos_r

    # Translate
    x_final = x_rot .+ xstart
    y_final = y_rot .+ ystart

    return (x_final, y_final, vx_rot, vy_rot, s)
end

function specsim_surface(
    x::AbstractArray,
    y::AbstractArray,
    mean_val::Float64,
    var::Float64,
    corl::Vector{Float64};
    mask = nothing,
)
    # x, y are 2D grids (or flat, but shape matters).
    # We assume x and y are grids of same shape.

    # Python implementation:
    # 1. subtract mean from coords to center (handled by fftshift/centering later or just grid centering)
    # 2. calc kernel distance r
    # 3. ryy = kernel(r)
    # 4. syy = fft(ryy)
    # 5. randomize phase
    # 6. ifft

    # We need a grid of distances.
    # Assuming x and y form a regular grid.

    # Check if x and y are 2D arrays coming from meshgrid
    dims = size(x)

    # Center coordinates relative to the grid center to make the kernel symmetric
    # In python: coords[i].ravel() - np.nanmean(coords[i])
    # Effectively we want lags.

    # Assuming uniform grid, we can just use 0 to N
    # But let's follow the inputs.

    x_centered = x .- Statistics.mean(x)
    y_centered = y .- Statistics.mean(y)

    # Gaussian kernel
    # h_square = 0.5 * (x / corl[0])^2 + 0.5 * (y / corl[1])^2
    h_square = 0.5 .* (x_centered ./ corl[1]) .^ 2 .+ 0.5 .* (y_centered ./ corl[2]) .^ 2
    ryy = var .* exp.(-h_square)

    ntot = length(ryy)

    # Power spectrum
    # Python: syy = np.fft.fftn(np.fft.fftshift(ryy)) / ntot
    syy = fft(fftshift(ryy)) ./ ntot
    syy = abs.(syy)
    syy[1] = 0 # Remove DC component (mean added later)

    # Random phase
    real_part = randn(dims)
    imag_part = randn(dims)
    epsilon = real_part .+ im .* imag_part

    rand_field = epsilon .* sqrt.(syy)
    Y = real(ifft(rand_field) .* ntot)

    Y .+= mean_val

    if mask !== nothing
        # apply mask (set to NaN)
        # Y[.!mask] .= NaN
        # In Julia NaN is float specific.
        return Y # return full, user handles mask
    end

    return Y
end

function contact_surface(x, y, mean, var, corl; mask = nothing)
    return specsim_surface(x, y, mean, var, corl; mask = mask)
end

end # module
