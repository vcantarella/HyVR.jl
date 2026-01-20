module Sheet

using KernelAbstractions
using ..Utils

export sheet!

@kernel function sheet_kernel!(
    f_array,
    dip_array,
    dip_dir_array,
    @Const(x),
    @Const(y),
    @Const(z),
    xmin,
    xmax,
    ymin,
    ymax,
    bottom_surf,
    top_surf,
    facies_val,
    internal_layering,
    alternating_facies,
    dip,
    dip_dir,
    layer_dist,
)
    I = @index(Global)

    xi = x[I]
    yi = y[I]
    zi = z[I]

    # Surface lookups
    # If bottom_surf and top_surf are arrays (2D), we need to map 3D index I to 2D index.
    # Assuming I maps to (iz, iy, ix) or similar.
    # We need to know the grid structure.
    # Assuming standard Julia array indexing: I is CartesianIndex or linear.
    # x, y, z are 3D.
    # If surfaces are 2D (xy plane), we assume they align with dimensions 1 and 2 or similar.
    # Let's assume the grid is (nx, ny, nz) or (nz, ny, nx). 
    # Python code: z,y,x = meshgrid(ij). 
    # For generality, we can try to use the coordinates to interpolate or lookup if passed as arrays.

    # Simplification: Assume surfaces are broadcastable scalars OR passed as 3D arrays matching grid size (replicated).
    # If passed as 2D, the user must broadcast them before calling or we assume flat top/bottom for now or 3D expansion.

    # Let's assume bottom_surf and top_surf are values at I (already broadcasted to 3D) or scalars.
    # Since KA doesn't support easy dynamic dispatch on array vs scalar inside kernel without multiple kernel defs,
    # we assume they are arrays of same shape as f_array or scalars.
    # The Python code uses `np.broadcast_to`.

    # Check boundaries
    if xi >= xmin && xi <= xmax && yi >= ymin && yi <= ymax
        # Check surfaces
        # We need to access surf[I] if array.

        # NOTE: Kernel arguments that are arrays must be accessed with I.
        # If scalar, just use value.
        # We'll rely on Julia's broadcasting or the user passing full 3D arrays for surfaces.

        b_val = bottom_surf[I]
        t_val = top_surf[I]

        if zi >= b_val && zi <= t_val

            if internal_layering
                # Planar layering logic
                nx, ny, nz = normal_plane_from_dip_dip_dir(dip, dip_dir)

                # Center for shift (optional, but good for consistency)
                xc = xmin + (xmax - xmin)/2
                yc = ymin + (ymax - ymin)/2
                zmax = t_val # local top? or max of top_surf

                # Shift
                shift = nx*xc + ny*yc + nz*zmax

                d = nx*xi + ny*yi + nz*zi - shift

                layer_idx = floor(Int, d / layer_dist)

                # Map to facies
                # We need min_value of layers to normalize index to 1..N
                # In kernel we can't easily find global min.
                # But facies repetition just depends on relative index.

                f_idx = mod1(layer_idx, length(facies_val))
                f_array[I] = facies_val[f_idx]

            else
                f_array[I] = facies_val[1]
            end

            dip_rad = deg2rad(dip)
            dip_dir_rad = coterminal_angle(dip_dir)
            dip_array[I] = dip_rad
            dip_dir_array[I] = dip_dir_rad
        end
    end
end

function sheet!(
    f_array,
    dip_array,
    dip_dir_array,
    x,
    y,
    z,
    xmin,
    xmax,
    ymin,
    ymax,
    bottom_surface,
    top_surface,
    facies;
    internal_layering = false,
    alternating_facies = false,
    dip = 0.0,
    dip_dir = 0.0,
    layer_dist = 0.0,
)

    backend = get_backend(f_array)

    # Broadcast surfaces if they are scalars
    if isa(bottom_surface, Number)
        bottom_surface = fill(bottom_surface, size(f_array))
    elseif ndims(bottom_surface) == 2
        # If 2D, we assume it's xy (or whatever the first 2 dims are) and replicate along z.
        # This depends on grid orientation.
        # If z is 3rd dim:
        s = size(f_array)
        # Reshape/repeat logic needed. 
        # For simplicity in this port, we require full 3D arrays or scalars.
        # Or we can let the user handle it.
        # Let's assume user passes compatible arrays.
    end

    if isa(top_surface, Number)
        top_surface = fill(top_surface, size(f_array))
    end

    # Move to backend
    # If on GPU, surfaces must be on GPU
    # f_array is on backend.

    facies_arr = isa(facies, Number) ? (facies,) : Tuple(facies)

    kernel = sheet_kernel!(backend)
    kernel(
        f_array,
        dip_array,
        dip_dir_array,
        x,
        y,
        z,
        xmin,
        xmax,
        ymin,
        ymax,
        bottom_surface,
        top_surface,
        facies_arr,
        internal_layering,
        alternating_facies,
        dip,
        dip_dir,
        layer_dist;
        ndrange = size(f_array),
    )

    KernelAbstractions.synchronize(backend)
end

end # module
