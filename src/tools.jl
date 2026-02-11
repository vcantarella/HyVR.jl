module Tools

using StochasticDiffEq
using FFTW
using Statistics
using LinearAlgebra
using Random
using Interpolations

export ferguson_curve, specsim_surface, contact_surface


function ferguson_curve(;
    h::Real,
    k::Real,
    ϵ::Real,
    θ::Real,
    sₘ::Real,
    xstart::Real,
    ystart::Real,
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
    sol = solve(prob, EM(), reltol = 1e-4, abstol = 1e-4, dt = 4)
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


function specsim_surface(
    x::AbstractArray,
    y::AbstractArray,
    mean_val::Real,
    var::Real,
    corl::AbstractVector;
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
    fill!(view(syy, 1:1), 0.0) # Remove DC component (mean added later)

    # Random phase
    real_part = similar(x)
    randn!(real_part)
    
    imag_part = similar(x)
    randn!(imag_part)
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
