using Test
using HyVR
using HyVR.Utils
using HyVR.Tools
using LinearAlgebra
using Statistics

@testset "HyVR Tests" begin

    @testset "Utils" begin
        # Test coterminal_angle
        @test isapprox(coterminal_angle(360), 0.0)
        @test isapprox(coterminal_angle(90), π/2)

        # Test normal_plane
        nx, ny, nz = normal_plane_from_dip_dip_dir(0, 0)
        @test isapprox(nz, 1.0) # Horizontal plane
    end

    @testset "Trough (Half Ellipsoid)" begin
        # 1. Create Grid (Uniform)
        # Old: x = 0:1:10 (11 points), y = 0:1:10 (11 points), z = 0:1:5 (6 points)
        origin = (0.0, 0.0, 0.0)
        spacing = (1.0, 1.0, 1.0)
        dims = (11, 11, 6)
        
        grid = UniformGrid(origin, spacing, dims)

        # Initialize Data Arrays
        # Note: dims in UniformGrid are (nx, ny, nz), so we allocate size(dims)
        f_array = fill(-1, dims)
        dip = zeros(dims)
        dip_dir = zeros(dims)

        center = (5.0, 5.0, 5.0)
        obj_dims = (4.0, 2.0, 2.0)
        azim = 0.0
        facies = 1

        # 2. Call with Grid Object (No x, y, z arrays!)
        half_ellipsoid!(f_array, dip, dip_dir, grid, center, obj_dims, azim, facies)

        # Check that some points are filled
        @test any(f_array .== 1)

        # Check center point (should be inside)
        # Index (6, 6, 6) corresponds to 5.0, 5.0, 5.0
        @test f_array[6, 6, 6] == 1
    end

    @testset "Sheet" begin
        # Grid: 5x5x5
        grid = UniformGrid((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (5, 5, 5))

        f_array = fill(-1, (5, 5, 5))
        dip = zeros((5, 5, 5))
        dip_dir = zeros((5, 5, 5))

        # Sheet from z=2 to z=4
        # Note: bounds are passed as scalars
        sheet!(f_array, dip, dip_dir, grid, -Inf, Inf, -Inf, Inf, 2.0, 4.0, 2)

        @test f_array[3, 3, 3] == 2 # z=3 inside
        @test f_array[3, 3, 1] == -1 # z=1 outside
        @test f_array[3, 3, 5] == -1 # z=5 outside
    end

    @testset "Tools: Ferguson Curve" begin
        x, y, vx, vy, s = ferguson_curve(
            h = 0.1,
            k = π/60,
            ϵ = 0.01,
            θ = 0.0,
            sₘ = 100.0,
            xstart = 0.0,
            ystart = 0.0,
        )

        @test length(x) > 0
        @test length(x) == length(y)
        @test x[1] == 0.0
        @test y[1] == 0.0
    end

    @testset "Tools: Specsim" begin
        # Specsim still uses 2D arrays because it relies on FFT
        xs = 0:1.0:100.0
        ys = 0:1.0:100.0
        x = [i for i in xs, j in ys]
        y = [j for i in xs, j in ys]

        mean_val = 10.0
        var_val = 4.0 
        corl = [5.0, 5.0]

        field = specsim_surface(x, y, mean_val, var_val, corl)

        @test size(field) == size(x)
        sample_mean = mean(field)
        sample_var = var(field)

        println("Specsim: Input Mean=$mean_val, Sample Mean=$sample_mean")
        println("Specsim: Input Var=$var_val, Sample Var=$sample_var")

        @test isapprox(sample_mean, mean_val, atol = 0.5)
        @test isapprox(sample_var, var_val, rtol = 0.5)
    end
end