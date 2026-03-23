import re

def rewrite_file(path, replacements):
    with open(path, 'r') as f:
        content = f.read()
    for old, new in replacements:
        if isinstance(old, re.Pattern):
            content = old.sub(new, content)
        else:
            content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)

base_replacements = [
    (re.compile(r'zeros\(([^,]+),\s*nlay,\s*nrow,\s*ncol\)'), r'zeros(\1, ncol, nrow, nlay)'),
    (re.compile(r'fill\(([^,]+),\s*nlay,\s*nrow,\s*ncol\)'), r'fill(\1, ncol, nrow, nlay)'),
    (re.compile(r'CUDA\.fill\(([^,]+),\s*nlay,\s*nrow,\s*ncol\)'), r'CUDA.fill(\1, ncol, nrow, nlay)'),
    (re.compile(r'CUDA\.zeros\(([^,]+),\s*nlay,\s*nrow,\s*ncol\)'), r'CUDA.zeros(\1, ncol, nrow, nlay)'),

    (re.compile(r'for k\s*=\s*1:nlay,\s*i\s*=\s*1:nrow,\s*j\s*=\s*1:ncol\s*\n\s*x_3d\[k, i, j\] = xs\[j\]\n\s*y_3d\[k, i, j\] = ys\[i\]\n\s*z_3d\[k, i, j\] = zs\[k\]\nend'),
     r'''for k = 1:nlay, i = 1:nrow, j = 1:ncol
    x_3d[j, i, k] = xs[j]
    y_3d[j, i, k] = ys[i]
    z_3d[j, i, k] = zs[k]
end
grid = RectilinearGrid(xs, ys, zs)'''),

    (re.compile(r'\[1, :, :\]'), r'[:, :, 1]'),
    (re.compile(r'\[layer_idx, :, :\]'), r'[:, :, layer_idx]'),
    (re.compile(r'facies\[k, :, :\]'), r'facies[:, :, k]'),
    (re.compile(r'facies\[lay, :, :\]'), r'facies[:, :, lay]'),
    (re.compile(r'z_3d\[lay, :, :\]'), r'z_3d[:, :, lay]'),
    
    (re.compile(r'x_3d,\s*y_3d,\s*z_3d,\s*\(x_c,\s*y_c,\s*z_c\)'), r'grid, (x_c, y_c, z_c)'),
    (re.compile(r'x_3d,\s*y_3d,\s*z_3d,\s*z_top_curr,'), r'grid, z_top_curr,'),
    (re.compile(r'x_3d,\s*y_3d,\s*z_3d,\s*\(xt,\s*yt,\s*zt\)'), r'grid, (xt, yt, zt)'),
    (re.compile(r'x_3d,\s*y_3d,\s*z_3d,\s*h_val\s*\+\s*thick,'), r'grid, h_val + thick,'),

    ("transpose(facies[mid_k, :, :])", "facies[:, :, mid_k]"),
    ("transpose(facies_cpu[mid_k, :, :])", "facies_cpu[:, :, mid_k]"),
    ("transpose(facies[:, :, mid_j])", "facies[mid_j, :, :]"),
    ("transpose(facies_cpu[:, :, mid_j])", "facies_cpu[mid_j, :, :]"),
    ("transpose(facies[:, mid_i, :])", "facies[:, mid_i, :]"),
    ("transpose(facies_cpu[:, mid_i, :])", "facies_cpu[:, mid_i, :]"),

    (re.compile(r'primitive_layer = fill\(7, nrow, ncol\)'), r'primitive_layer = fill(7, ncol, nrow)'),
    (re.compile(r'primitive_layer = CUDA\.fill\(7, nrow, ncol\)'), r'primitive_layer = CUDA.fill(7, ncol, nrow)'),
    (re.compile(r'reshape\(primitive_flat, nrow, ncol\)'), r'reshape(primitive_flat, ncol, nrow)'),

    (re.compile(r'for k = 1:nlay, i = 1:nrow, j = 1:ncol\n\s*if z_3d\[k, i, j\] >= surf_top\[i, j\]\n\s*facies\[k, i, j\] = 21\n\s*end\n\s*if z_3d\[k, i, j\] <= surf_botm\[i, j\]\n\s*facies\[k, i, j\] = 31\n\s*end\nend'),
     r'''for k = 1:nlay, i = 1:nrow, j = 1:ncol
    if z_3d[j, i, k] >= surf_top[j, i]
        facies[j, i, k] = 21
    end
    if z_3d[j, i, k] <= surf_botm[j, i]
        facies[j, i, k] = 31
    end
end'''),

    # heinz_main_discharge_area.jl specific
    (re.compile(r'for j in axes\(z_3d, 1\).*?local_facies\[bottom_index\] \.= 3\nend', re.DOTALL),
     r'''for j in axes(z_3d, 3) # number of layers
    top_index = z_3d[:, :, j] .> surf_top
    local_facies = @view facies[:, :, j]
    local_facies[top_index] .= 1
    bottom_index = z_3d[:, :, j] .< surf_botm
    local_facies[bottom_index] .= 3
end'''),

    # wiggly_layers.jl specific
    (re.compile(r'layf\[layz \.< surf_botms\[i\] \.&& layf \.== 7\] \.= i'),
     r'layf[layz .< surf_botms[i] .&& layf .== 7] .= i'),
    (re.compile(r'scour_pools_acc = zeros\(size\(facies\)\)'),
     r'scour_pools_acc = zeros(size(facies))'),
]

for script in [
    'C:/Users/vcant/Documents/HyVR.jl/examples/ammer_valley.jl',
    'C:/Users/vcant/Documents/HyVR.jl/examples/heinz_main_discharge_area.jl',
    'C:/Users/vcant/Documents/HyVR.jl/examples/wiggly_layers.jl'
]:
    rewrite_file(script, base_replacements)

gpu_replacements = [
    ("x_3d = repeat(reshape(xs_gpu, 1, 1, ncol), nlay, nrow, 1)", 
     "x_3d = repeat(reshape(xs_gpu, ncol, 1, 1), 1, nrow, nlay)\n\n# Create Grid\ngrid = RectilinearGrid(xs_gpu, ys_gpu, zs_gpu)"),
    ("y_3d = repeat(reshape(ys_gpu, 1, nrow, 1), nlay, 1, ncol)", 
     "y_3d = repeat(reshape(ys_gpu, 1, nrow, 1), ncol, 1, nlay)"),
    ("z_3d = repeat(reshape(zs_gpu, nlay, 1, 1), 1, nrow, ncol)", 
     "z_3d = repeat(reshape(zs_gpu, 1, 1, nlay), ncol, nrow, 1)"),
    ("x_2d_cpu = zeros(Float32, nrow, ncol)", "x_2d_cpu = zeros(Float32, ncol, nrow)"),
    ("y_2d_cpu = zeros(Float32, nrow, ncol)", "y_2d_cpu = zeros(Float32, ncol, nrow)"),
    ("x_2d_cpu[i, j] = xs[j]", "x_2d_cpu[j, i] = xs[j]"),
    ("y_2d_cpu[i, j] = ys[i]", "y_2d_cpu[j, i] = ys[i]"),
    ("surf_top_gpu = reshape(CuArray(surf_top), 1, nrow, ncol)", "surf_top_gpu = reshape(CuArray(surf_top), ncol, nrow, 1)"),
    ("surf_botm_gpu = reshape(CuArray(surf_botm), 1, nrow, ncol)", "surf_botm_gpu = reshape(CuArray(surf_botm), ncol, nrow, 1)"),
] + base_replacements

rewrite_file('C:/Users/vcant/Documents/HyVR.jl/examples/ammer_valley_gpu.jl', gpu_replacements)
