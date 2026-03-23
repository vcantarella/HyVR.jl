import os
import re

files = [
    'examples/ammer_valley.jl', 
    'examples/ammer_valley_gpu.jl', 
    'examples/heinz_main_discharge_area.jl', 
    'examples/wiggly_layers.jl'
]

for file in files:
    with open(file, 'r') as f:
        content = f.read()

    # Change zeros calls
    content = re.sub(r'\bdip = zeros\(', 'dip_arr = zeros(', content)
    content = re.sub(r'\bdip_dir = zeros\(', 'dip_dir_arr = zeros(', content)
    
    content = re.sub(r'\bdip = CUDA\.zeros\(', 'dip_arr = CUDA.zeros(', content)
    content = re.sub(r'\bdip_dir = CUDA\.zeros\(', 'dip_dir_arr = CUDA.zeros(', content)

    # Change half_ellipsoid! calls
    content = re.sub(r'half_ellipsoid\!\(\s*facies,\s*dip,\s*dip_dir,', 'half_ellipsoid!(facies, dip_arr, dip_dir_arr,', content)
    content = re.sub(r'half_ellipsoid\!\(\s*scour_pools_acc,\s*dip,\s*dip_dir,', 'half_ellipsoid!(scour_pools_acc, dip_arr, dip_dir_arr,', content)
    
    # Change channel! calls
    content = re.sub(r'channel\!\(\s*facies,\s*dip,\s*dip_dir,', 'channel!(facies, dip_arr, dip_dir_arr,', content)

    with open(file, 'w') as f:
        f.write(content)
