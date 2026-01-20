module Channel

using KernelAbstractions
using ..Utils

export channel!

@kernel function channel_kernel!(
    f_array,
    dip_array,
    dip_dir_array,
    @Const(x),
    @Const(y),
    @Const(z),
    z_top,
    @Const(curve_x),
    @Const(curve_y),
    width,
    depth,
    facies_val,
    internal_layering,
    alternating_facies,
    dip,
    layer_dist,
)
    I = @index(Global)

    xi = x[I]
    yi = y[I]
    zi = z[I]

    dz = zi - z_top

    # Check z range
    if dz <= 0 && dz >= -depth
        # Check vicinity (Bounding Box of curve)
        # This can be passed as args or computed.
        # For kernel simplicity, we do the check.
        # But `min_distance_kernel` is expensive, so checking bounds of curve helps.
        # We assume `curve_x` limits are handled outside or we just run min_dist.

        # Calculate distance to curve
        dist_sq, idx = min_distance_sq_kernel(xi, yi, curve_x, curve_y)

        # Check parabola condition
        # width^2 / 4 + width^2 * dz / (4 * depth)

        threshold = (width^2 / 4) + (width^2 * dz) / (4 * depth)

        if dist_sq <= threshold
            # Inside
            if internal_layering
                # Calculate curve length up to idx
                # This requires cumulative sum of curve segments.
                # Passing pre-calculated dist_curve array is better.
                # Assuming `curve_s` (distance along curve) is passed or approximating.
                # For now, simplistic or placeholder.

                # If we want exact port, we need `dist_curve` array corresponding to curve points.
                # Let's assume we implement without for now, or just massive fill if data missing.
                f_array[I] = facies_val[1]
            else
                f_array[I] = facies_val[1]
            end

            # Massive structure defaults
            dip_array[I] = 0.0
            dip_dir_array[I] = 0.0
        end
    end
end

function channel!(
    f_array,
    dip_array,
    dip_dir_array,
    x,
    y,
    z,
    z_top,
    curve,
    parabola_pars,
    facies;
    internal_layering = false,
    alternating_facies = false,
    dip = 0.0,
    layer_dist = 0.0,
)

    backend = get_backend(f_array)

    width, depth = parabola_pars
    curve_x = curve[:, 1]
    curve_y = curve[:, 2]

    # Ensure curve arrays are on the backend
    # If f_array is CuArray, curve_x/y should be too.
    # We assume user handles or we convert (not easy generically without adapt).
    # KernelAbstractions `get_backend` returns the backend.
    # Adapting is usually done via `KernelAbstractions.allocate` or Adapt.jl.
    # For this prototype, we assume consistent types.

    facies_arr = isa(facies, Number) ? (facies,) : Tuple(facies)

    kernel = channel_kernel!(backend)
    kernel(
        f_array,
        dip_array,
        dip_dir_array,
        x,
        y,
        z,
        z_top,
        curve_x,
        curve_y,
        width,
        depth,
        facies_arr,
        internal_layering,
        alternating_facies,
        dip,
        layer_dist;
        ndrange = size(f_array),
    )

    KernelAbstractions.synchronize(backend)
end

end # module
