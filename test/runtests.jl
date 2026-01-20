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
        # Create grid
        x = [i for i = 0.0:1.0:10.0, j = 0.0:1.0:10.0, k = 0.0:1.0:5.0]
        y = [j for i = 0.0:1.0:10.0, j = 0.0:1.0:10.0, k = 0.0:1.0:5.0]
        z = [k for i = 0.0:1.0:10.0, j = 0.0:1.0:10.0, k = 0.0:1.0:5.0]

        f_array = fill(-1, size(x))
        dip = zeros(size(x))
        dip_dir = zeros(size(x))

        center = (5.0, 5.0, 5.0)
        dims = (4.0, 2.0, 2.0)
        azim = 0.0
        facies = 1

        half_ellipsoid!(f_array, dip, dip_dir, x, y, z, center, dims, azim, facies)

        # Check that some points are filled
        @test any(f_array .== 1)

        # Check center point (should be inside)
        # Index corresponding to 5.0, 5.0, 5.0 -> indices (6, 6, 6) since 0-based grid + 1
        # z max index is 6 (0 to 5)
        @test f_array[6, 6, 6] == 1
    end

    @testset "Sheet" begin
        x = zeros(5, 5, 5) # Dummy
        # Real grid
        x = [i for i = 1:5, j = 1:5, k = 1:5]
        y = [j for i = 1:5, j = 1:5, k = 1:5]
        z = [k for i = 1:5, j = 1:5, k = 1:5]

        f_array = fill(-1, size(x))
        dip = zeros(size(x))
        dip_dir = zeros(size(x))

        # Sheet from z=2 to z=4
        sheet!(f_array, dip, dip_dir, x, y, z, -Inf, Inf, -Inf, Inf, 2.0, 4.0, 2)

        @test f_array[3, 3, 3] == 2 # z=3 inside
        @test f_array[3, 3, 1] == -1 # z=1 outside
        @test f_array[3, 3, 5] == -1 # z=5 outside
    end

    @testset "Tools: Ferguson Curve" begin
        # Test generation
        x, y, vx, vy, s = ferguson_curve(
            h = 0.1,
            k = π/60,
            eps_factor = 0.01,
            flow_angle = 0.0,
            s_max = 100.0,
            xstart = 0.0,
            ystart = 0.0,
        )

        @test length(x) > 0
        @test length(x) == length(y)
        @test x[1] == 0.0
        @test y[1] == 0.0
    end

    @testset "Tools: Specsim" begin
        # 2D Grid
        xs = 0:1.0:100.0
        ys = 0:1.0:100.0
        x = [i for i in xs, j in ys]
        y = [j for i in xs, j in ys]

        mean_val = 10.0
        var_val = 4.0 # std = 2.0
        corl = [5.0, 5.0] # Correlation length smaller than domain for ergodicity

        # We perform multiple realizations to check ensemble statistics better, 
        # or just one large one. Let's do one large one.

        field = specsim_surface(x, y, mean_val, var_val, corl)

        @test size(field) == size(x)

        # Check statistics (allow for some sample variance)
        sample_mean = mean(field)
        sample_var = var(field)

        println("Specsim: Input Mean=$mean_val, Sample Mean=$sample_mean")
        println("Specsim: Input Var=$var_val, Sample Var=$sample_var")

        @test isapprox(sample_mean, mean_val, atol = 0.5)
        # Variance converges slower, use rough check
        @test isapprox(sample_var, var_val, rtol = 0.5)
    end

end
