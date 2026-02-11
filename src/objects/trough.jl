module Trough

using KernelAbstractions
using ..Utils

export half_ellipsoid!

"""
    half_ellipsoid!(f_array, dip_array, dip_dir_array, x, y, z, center_coords, dims, azim, facies; ...)

In-place modification of arrays for trough object.
"""
@kernel function half_ellipsoid_kernel!(
    f_array,
    dip_array,
    dip_dir_array,
    @Const(x),
    @Const(y),
    @Const(z),
    @Const(x_c),
    @Const(y_c),
    @Const(z_c),
    @Const(a),
    @Const(b),
    @Const(c),
    @Const(alpha),
    @Const(facies_val),
    @Const(internal_layering),
    @Const(alternating_facies),
    @Const(bulb),
    @Const(dip_limit),
    @Const(dip_dir_val),
    @Const(layer_dist),
    @Const(box_min),
    @Const(box_max)
)
    I = @index(Global, Cartesian)

    # 1. Check if point is inside
    xi = x[I]
    yi = y[I]
    zi = z[I]

    # Quick bounding box check (optional optimization, but we usually launch strictly or rely on fast fail)
    # The kernel launch range should ideally be the bounding box.
    if xi ≥ box_min[1] || xi ≤ box_max[1] ||
       yi ≥ box_min[2] || yi ≤ box_max[2] ||
       zi ≥ box_min[3] || zi ≤ box_max[3]
        

        # Semi-ellipsoid check: z <= z_c
        if zi <= z_c
            inside = is_point_inside_ellipsoid(xi, yi, zi, x_c, y_c, z_c, a, b, c, alpha)

            if inside
                # Calculate properties
                if bulb
                    dip_out, dip_dir_out, norm_dist = dip_dip_dir_bulbset(
                        xi,
                        yi,
                        zi,
                        x_c,
                        y_c,
                        z_c,
                        a,
                        b,
                        c,
                        alpha,
                        dip_limit,
                    )

                    # Assign Facies
                    if internal_layering
                        # This requires facies array logic which is hard in kernel without alloc
                        # Simplification: Calculate index and map to facies value if facies is passed as scalar or tuple
                        # Since facies can be an array, we pass it.
                        # Note: accessing arrays in kernels is fine.

                        # n_layers calculation
                        # Python: n_layers = int(ceil(max(norm_dist) * c / layer_dist))
                        # We can't know max(norm_dist) for the whole object here easily without a separate pass.
                        # However, max norm_dist is 1.0 at boundary.
                        # So max_dist approx c (in Z) or scaled.
                        # Actually norm_distance is normalized. Max is 1.

                        # Facies calculation
                        # ns = floor(norm_dist * c / layer_dist)

                        # We need to handle alternating facies logic.
                        # If we just mod index, it works.

                        # Assuming facies is a 1D array on GPU/CPU

                        # idx = floor(Int, norm_dist * c / layer_dist) + 1
                        # val = facies[mod1(idx, length(facies))]
                        # f_array[I] = val

                        # Placeholder for exact logic:
                        # Python logic uses 'get_alternating_facies' pre-calc.
                        # We can emulate modulo.

                        layer_idx = floor(Int, norm_dist * c / layer_dist) + 1
                        # Alternating logic usually just cycles through the available facies
                        # If not alternating, it chooses random. Random in kernel is tricky.
                        # We assume alternating=True or just deterministic mapping for reproduction.

                        f_idx = mod1(layer_idx, length(facies_val)) # facies_val is the array
                        f_array[I] = facies_val[f_idx]

                    else
                        # Homogeneous
                        # facies_val might be array, take first or if scalar
                        f_array[I] = facies_val[1]
                    end

                    dip_array[I] = dip_out
                    dip_dir_array[I] = dip_dir_out

                else
                    # Massive or planar internal
                    if internal_layering
                        nx, ny, nz = normal_plane_from_dip_dip_dir(dip_limit, dip_dir_val)
                        # shift = layer_dist + nx*x_c + ny*y_c + nz*z_c
                        shift = layer_dist + nx*x_c + ny*y_c + nz*z_c
                        plane_dist = xi*nx + yi*ny + zi*nz - shift

                        # ns = floor(plane_dist / layer_dist) + (n_layers // 2)
                        # We approximate n_layers center.
                        # Just use plane_dist / layer_dist

                        layer_idx = floor(Int, abs(plane_dist) / layer_dist) + 1
                        f_idx = mod1(layer_idx, length(facies_val))
                        f_array[I] = facies_val[f_idx]
                    else
                        f_array[I] = facies_val[1]
                    end

                    dip_rad = deg2rad(dip_limit)
                    dip_dir_rad = coterminal_angle(dip_dir_val)
                    dip_array[I] = dip_rad
                    dip_dir_array[I] = dip_dir_rad
                end
            end
        end
    end
end

function half_ellipsoid!(
    f_array,
    dip_array,
    dip_dir_array,
    x,
    y,
    z,
    center_coords,
    dims,
    azim,
    facies;
    internal_layering = false,
    alternating_facies = false,
    bulb = false,
    dip = 0,
    dip_dir = 0,
    layer_dist = 0,
    ϵ_bbox = 1e-4
)

    # Backend determination
    backend = get_backend(f_array)

    # Unpack
    x_c, y_c, z_c = center_coords
    a, b, c = dims
    alpha = coterminal_angle(azim)

    # Bounding box calculation on CPU to limit kernel execution domain
    # Rotation math for BBox
    sin_a = sin(alpha)
    cos_a = cos(alpha)

    # Half-widths
    dx_len = sqrt((a*cos_a)^2 + (b*sin_a)^2)
    dy_len = sqrt((a*sin_a)^2 + (b*cos_a)^2)
    dz_len = c

    # Physical bounds
    b_xmin, b_xmax = x_c - dx_len, x_c + dx_len
    b_ymin, b_ymax = y_c - dy_len, y_c + dy_len
    b_zmin, b_zmax = z_c - dz_len, z_c # Trough is usually top-down, but cover full c extent

   # Add a tiny buffer (epsilon) to be safe
    box_min = (x_c - dx_len - ϵ_bbox, y_c - dy_len - ϵ_bbox, z_c - dz_len - ϵ_bbox)
    box_max = (x_c + dx_len + ϵ_bbox, y_c + dy_len + ϵ_bbox, z_c + dz_len + ϵ_bbox)

    # Extents
    # We want to find the range of indices in x, y, z that cover the ellipsoid.
    # This assumes x, y, z are grid arrays (meshgrid).
    # If they are ranges or vectors, we handle differently.
    # Assuming x, y, z are 3D arrays matching f_array size (MODFLOW style or standard).

    # Actually, calculating the exact indices might be complex if the grid is irregular.
    # If regular, we can find min/max coordinates and map to indices.

    # For now, we launch over the whole grid for correctness, assuming grid size isn't massive or GPU handles it.
    # Optimization: If x,y,z are coordinate arrays, we could find indices.

    # Launch kernel
    kernel = half_ellipsoid_kernel!(backend)
    ndrange = size(f_array)

    # Ensure facies is a Tuple (bitstype) to be passed to kernel
    facies_arr = isa(facies, Number) ? (facies,) : Tuple(facies)

    kernel(
        f_array,
        dip_array,
        dip_dir_array,
        x,
        y,
        z,
        x_c,
        y_c,
        z_c,
        a,
        b,
        c,
        alpha,
        facies_arr,
        internal_layering,
        alternating_facies,
        bulb,
        dip,
        dip_dir,
        layer_dist,
        box_min,
        box_max;
        ndrange = ndrange,
    )

    KernelAbstractions.synchronize(backend)
end

end # module
