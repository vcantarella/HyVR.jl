using Test
using HyVR
using HyVR.Utils
using HyVR.Tools
using LinearAlgebra
using Statistics
using CUDA
using KernelAbstractions
using HyVR.Grids

@testset "HyVR CUDA Tests" begin

    if !CUDA.functional()
        @warn "CUDA is not functional. Skipping GPU tests."
    else
        @info "Running tests on CUDA GPU..."

        @testset "Trough (Half Ellipsoid) - GPU" begin
            # 1. Create Grid (CPU object is fine!)
            grid = UniformGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (11, 11, 6))

            # 2. Initialize Data Arrays on GPU
            # We only need to allocate the data we write to.
            # Coordinates are implicit in the grid object.
            dims = grid_size(grid)
            f_array = CUDA.fill(-1, dims)
            dip = CUDA.zeros(dims)
            dip_dir = CUDA.zeros(dims)

            # 3. Setup Parameters
            center = (5.0, 5.0, 5.0)
            obj_dims = (4.0, 2.0, 2.0)
            azim = 0.0
            facies = 1

            # 4. Run Simulation
            # Pass 'grid' directly. It works on GPU kernels automatically.
            half_ellipsoid!(f_array, dip, dip_dir, grid, center, obj_dims, azim, facies)
            
            KernelAbstractions.synchronize(CUDA.CUDABackend())

            # 5. Bring results back
            f_array_cpu = Array(f_array)

            @test any(f_array_cpu .== 1)
            @test f_array_cpu[6, 6, 6] == 1
        end

        @testset "Sheet - GPU" begin
            grid = UniformGrid((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (5, 5, 5))

            f_array = CUDA.fill(-1, (5, 5, 5))
            dip = CUDA.zeros((5, 5, 5))
            dip_dir = CUDA.zeros((5, 5, 5))

            sheet!(f_array, dip, dip_dir, grid, -Inf, Inf, -Inf, Inf, 2.0, 4.0, 2)
            
            KernelAbstractions.synchronize(CUDA.CUDABackend())

            f_array_cpu = Array(f_array)

            @test f_array_cpu[3, 3, 3] == 2 
            @test f_array_cpu[3, 3, 1] == -1 
        end

        @testset "Tools: Specsim - GPU" begin
            # Specsim still uses arrays
            xs = 0:1.0:100.0
            ys = 0:1.0:100.0
            
            x_cpu = [i for i in xs, j in ys]
            y_cpu = [j for i in xs, j in ys]
            
            x = CuArray(x_cpu)
            y = CuArray(y_cpu)

            mean_val = 10.0
            var_val = 4.0
            corl = [5.0, 5.0]

            field_gpu = specsim_surface(x, y, mean_val, var_val, corl)
            field_cpu = Array(field_gpu)

            @test size(field_cpu) == size(x_cpu)
            @test isapprox(mean(field_cpu), mean_val, atol = 0.5)
        end
    end
end