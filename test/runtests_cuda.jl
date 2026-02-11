using Test
using HyVR
using HyVR.Utils
using HyVR.Tools
using LinearAlgebra
using Statistics
using CUDA
using KernelAbstractions

@testset "HyVR CUDA Tests" begin

    if !CUDA.functional()
        @warn "CUDA is not functional. Skipping GPU tests."
    else
        @info "Running tests on CUDA GPU..."

        @testset "Trough (Half Ellipsoid) - GPU" begin
            # 1. Create Grid on CPU first
            x_cpu = [i for i = 0.0:1.0:10.0, j = 0.0:1.0:10.0, k = 0.0:1.0:5.0]
            y_cpu = [j for i = 0.0:1.0:10.0, j = 0.0:1.0:10.0, k = 0.0:1.0:5.0]
            z_cpu = [k for i = 0.0:1.0:10.0, j = 0.0:1.0:10.0, k = 0.0:1.0:5.0]

            # 2. Move to GPU
            x = CuArray(x_cpu)
            y = CuArray(y_cpu)
            z = CuArray(z_cpu)

            # Initialize output arrays on GPU
            f_array = CUDA.fill(-1, size(x))
            dip = CUDA.zeros(size(x))
            dip_dir = CUDA.zeros(size(x))

            # 3. Setup Parameters
            center = (5.0, 5.0, 5.0)
            dims = (4.0, 2.0, 2.0)
            azim = 0.0
            facies = 1

            # 4. Run Simulation (Kernel launches on GPU automatically based on array type)
            half_ellipsoid!(f_array, dip, dip_dir, x, y, z, center, dims, azim, facies)
            
            # Synchronize to ensure kernel is finished (KA usually handles this, but good practice in tests)
            KernelAbstractions.synchronize(CUDA.CUDABackend())

            # 5. Bring results back to CPU for testing
            f_array_cpu = Array(f_array)

            # Check that some points are filled
            @test any(f_array_cpu .== 1)

            # Check center point (indices 6,6,6 for coords 5.0,5.0,5.0)
            @test f_array_cpu[6, 6, 6] == 1
        end

        @testset "Sheet - GPU" begin
            # Create Grid
            x_cpu = [i for i = 1:5, j = 1:5, k = 1:5]
            y_cpu = [j for i = 1:5, j = 1:5, k = 1:5]
            z_cpu = [k for i = 1:5, j = 1:5, k = 1:5]

            x = CuArray(Float64.(x_cpu))
            y = CuArray(Float64.(y_cpu))
            z = CuArray(Float64.(z_cpu))

            f_array = CUDA.fill(-1, size(x))
            dip = CUDA.zeros(size(x))
            dip_dir = CUDA.zeros(size(x))

            # Sheet from z=2 to z=4
            sheet!(f_array, dip, dip_dir, x, y, z, -Inf, Inf, -Inf, Inf, 2.0, 4.0, 2)
            
            KernelAbstractions.synchronize(CUDA.CUDABackend())

            # Fetch result
            f_array_cpu = Array(f_array)

            @test f_array_cpu[3, 3, 3] == 2  # z=3 inside
            @test f_array_cpu[3, 3, 1] == -1 # z=1 outside
            @test f_array_cpu[3, 3, 5] == -1 # z=5 outside
        end

        @testset "Tools: Specsim - GPU" begin
            # Specsim uses FFT. If HyVR uses generic FFT interfaces, 
            # passing CuArrays should trigger cuFFT.
            
            xs = 0:1.0:100.0
            ys = 0:1.0:100.0
            
            # Broadcast to create 2D arrays, convert to CuArray
            x_cpu = [i for i in xs, j in ys]
            y_cpu = [j for i in xs, j in ys]
            
            x = CuArray(x_cpu)
            y = CuArray(y_cpu)

            mean_val = 10.0
            var_val = 4.0
            corl = [5.0, 5.0]

            # Run Specsim on GPU
            field_gpu = specsim_surface(x, y, mean_val, var_val, corl)

            # Bring back to CPU
            field_cpu = Array(field_gpu)

            @test size(field_cpu) == size(x_cpu)

            sample_mean = mean(field_cpu)
            sample_var = var(field_cpu)

            println("GPU Specsim: Input Mean=$mean_val, Sample Mean=$sample_mean")
            println("GPU Specsim: Input Var=$var_val, Sample Var=$sample_var")

            @test isapprox(sample_mean, mean_val, atol = 0.5)
            @test isapprox(sample_var, var_val, rtol = 0.5)
        end
    end
    
    # NOTE: "Utils" and "Ferguson Curve" are primarily CPU logic (scalar math or path generation)
    # and don't strictly benefit from GPU testing unless rewritten as kernels. 
    # They are omitted here to focus on the heavy array operations.
end