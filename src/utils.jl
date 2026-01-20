module Utils

using KernelAbstractions
using LinearAlgebra
using StaticArrays

export coterminal_angle,
    rotation_matrix_x,
    rotation_matrix_z,
    is_point_inside_ellipsoid,
    normal_plane_from_dip_dip_dir,
    dip_dip_dir_bulbset,
    get_alternating_facies,
    min_distance_kernel,
    min_distance_sq_kernel,
    compute_min_distance!

"""
    coterminal_angle(angle)

Ensure the angle is in the range [0, 2π) radians.
Input angle is in degrees.
"""
@inline function coterminal_angle(angle)
    normalized_angle = mod(angle, 360)
    return deg2rad(normalized_angle)
end

@inline function rotation_matrix_x(alpha)
    return @SMatrix [
        1.0 0.0 0.0;
        0.0 cos(alpha) -sin(alpha);
        0.0 sin(alpha) cos(alpha)
    ]
end

@inline function rotation_matrix_z(alpha)
    return @SMatrix [
        cos(alpha) -sin(alpha) 0.0;
        sin(alpha) cos(alpha) 0.0;
        0.0 0.0 1.0
    ]
end

"""
    is_point_inside_ellipsoid(x, y, z, x_c, y_c, z_c, a, b, c, alpha)

Check if point is inside ellipsoid. 
"""
@inline function is_point_inside_ellipsoid(x, y, z, x_c, y_c, z_c, a, b, c, alpha)
    dx = x - x_c
    dy = y - y_c
    dz = z - z_c

    # Using StaticArrays for rotation
    matrix = rotation_matrix_z(alpha)
    point = @SVector [dx, dy, dz]
    rotated_points = matrix * point

    rx = rotated_points[1]
    ry = rotated_points[2]
    rz = rotated_points[3]

    distance_vector = (rx^2 / a^2) + (ry^2 / b^2) + (rz^2 / c^2)
    return distance_vector <= 1.0
end

@inline function normal_plane_from_dip_dip_dir(dip, dip_dir)
    # dip: degrees
    # dip_dir: degrees
    dip_rad = deg2rad(dip)
    dip_dir_rad = deg2rad(dip_dir + 90)

    nx = -sin(dip_rad) * cos(dip_dir_rad)
    ny = sin(dip_rad) * sin(dip_dir_rad)
    nz = cos(dip_rad)
    return (nx, ny, nz)
end

@inline function dip_dip_dir_bulbset(x, y, z, x_c, y_c, z_c, a, b, c, alpha, dip_limit)
    dx = x - x_c
    dy = y - y_c
    dz = z - z_c

    cos_a = cos(alpha)
    sin_a = sin(alpha)

    normalized_dx = (dx * cos_a + dy * sin_a) / a
    normalized_dy = (-dx * sin_a + dy * cos_a) / b
    normalized_dz = dz / c

    aaa = normalized_dx / a
    bbb = normalized_dy / b
    ccc = normalized_dz / c

    len_normvec = sqrt(aaa^2 + bbb^2 + ccc^2)

    # Avoid division by zero
    if len_normvec < 1e-12
        return 0.0, 0.0, 0.0
    end

    normvec_x = (aaa * cos_a + bbb * sin_a) / len_normvec
    normvec_y = (-aaa * sin_a + bbb * cos_a) / len_normvec
    normvec_z = ccc / len_normvec

    len_normvec_xy = sqrt(normvec_x^2 + normvec_y^2)
    norm_distance = sqrt(normalized_dx^2 + normalized_dy^2 + normalized_dz^2)

    dip_bulb = -acos(abs(normvec_z)) * sign(normvec_x * normvec_z) * 180 / π

    dip_dir_counterclockwise = -atan(sign(normvec_x) * normvec_y, abs(normvec_x)) * 180 / π
    dip_dir_bulb = mod(-dip_dir_counterclockwise + 90, 360)

    dip_out = dip_bulb
    dip_dir_out = dip_dir_bulb

    if len_normvec_xy < 1e-12
        dip_out = 0.0
        dip_dir_out = 0.0
    end

    # Adjust for geological application
    if dip_out < 0
        if dip_dir_out < 180
            dip_dir_out += 180
        else
            dip_dir_out -= 180
        end
    end

    dip_out = abs(dip_out)
    if dip_out > dip_limit
        dip_out = dip_limit
    end

    return deg2rad(dip_out), deg2rad(dip_dir_out), norm_distance
end

function get_alternating_facies(facies::AbstractArray, n_layers::Int, alternating::Bool)
    if alternating
        # Repeat facies pattern
        len = length(facies)
        result = Vector{eltype(facies)}(undef, n_layers)
        for i = 1:n_layers
            idx = mod1(i, len)
            result[i] = facies[idx]
        end
        return result
    else
        # Random choice
        return rand(facies, n_layers)
    end
end

"""
    min_distance_kernel(x, y, curve_x, curve_y)

Finds the minimum squared distance and the index of the closest point on the curve.
This is meant to be called within a kernel for a single point (x,y).
"""
@inline function min_distance_sq_kernel(x, y, curve_x, curve_y)
    min_dist_sq = Inf
    min_idx = 1

    # We iterate over the curve points
    # NOTE: In a GPU kernel, curve_x and curve_y should be in GPU memory.
    # Be careful about the size of the curve.

    @inbounds for i in eachindex(curve_x)
        d2 = (x - curve_x[i])^2 + (y - curve_y[i])^2
        if d2 < min_dist_sq
            min_dist_sq = d2
            min_idx = i
        end
    end
    return min_dist_sq, min_idx
end

@kernel function min_distance_kernel!(d_array, px, py, curve_x, curve_y)
    i = @index(Global)
    dist_sq, _ = min_distance_sq_kernel(px[i], py[i], curve_x, curve_y)
    d_array[i] = sqrt(dist_sq)
end

function compute_min_distance!(d_array, curve_x, curve_y, px, py)
    backend = get_backend(px)
    kernel! = min_distance_kernel!(backend)
    kernel!(d_array, px, py, curve_x, curve_y, ndrange = length(px))
    KernelAbstractions.synchronize(backend)
    return d_array
end

end # module
