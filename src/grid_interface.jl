module Grids
        
    abstract type HyVRGrid end

    export HyVRGrid, UniformGrid, RectilinearGrid, StructuredGrid, UnstructuredGrid
    export get_xyz, grid_size

    # ==============================================================================
    # 1. UNIFORM GRID (Regular spacing)
    # ==============================================================================
    struct UniformGrid{T<:AbstractFloat} <: HyVRGrid
        origin::NTuple{3, T}
        spacing::NTuple{3, T}
        dims::NTuple{3, Int}
    end

    # Usage: UniformGrid((0.0,0.0,0.0), (1.0,1.0,1.0), (100,100,50))

    @inline function get_xyz(g::UniformGrid, I::CartesianIndex{3})
        # Math-based calculation: 0 memory loads
        # I[1]=i (x), I[2]=j (y), I[3]=k (z) - Adjust order if your convention differs
        x = g.origin[1] + (I[1] - 1) * g.spacing[1]
        y = g.origin[2] + (I[2] - 1) * g.spacing[2]
        z = g.origin[3] + (I[3] - 1) * g.spacing[3]
        return x, y, z
    end

    grid_size(g::UniformGrid) = g.dims

    # ==============================================================================
    # 2. RECTILINEAR GRID (Variable spacing, orthogonal)
    # ==============================================================================
    struct RectilinearGrid{T, V<:AbstractVector{T}} <: HyVRGrid
        x::V # Cell centers or nodes in X
        y::V # Cell centers or nodes in Y
        z::V # Cell centers or nodes in Z
    end

    @inline function get_xyz(g::RectilinearGrid, I::CartesianIndex{3})
        # 3 independent memory loads (Fast)
        return g.x[I[1]], g.y[I[2]], g.z[I[3]]
    end

    grid_size(g::RectilinearGrid) = (length(g.x), length(g.y), length(g.z))

    # ==============================================================================
    # 3. STRUCTURED GRID (Deformed Z / Stratigraphic)
    # ==============================================================================
    struct StructuredGrid{T, V<:AbstractVector{T}, A<:AbstractArray{T, 3}} <: HyVRGrid
        x::V # Regular X coordinates
        y::V # Regular Y coordinates
        z::A # Full 3D array for Z (e.g., from wiggly_layers)
    end

    @inline function get_xyz(g::StructuredGrid, I::CartesianIndex{3})
        # Hybrid: Math/Lookup for X/Y, Full Lookup for Z
        return g.x[I[1]], g.y[I[2]], g.z[I]
    end

    grid_size(g::StructuredGrid) = size(g.z)

    # ==============================================================================
    # 4. UNSTRUCTURED GRID (Point Cloud / Mesh Centers)
    # ==============================================================================
    struct UnstructuredGrid{T, V<:AbstractVector{T}} <: HyVRGrid
        x::V
        y::V
        z::V
        # potentially cell_volumes, etc.
    end

    # --- The Magic: Handling Linear Indices ---

    # Case A: Kernel is launched with 1D range -> I is Int
    @inline function get_xyz(g::UnstructuredGrid, i::Int)
        return g.x[i], g.y[i], g.z[i]
    end

    # Case B: Kernel is launched with 1D range -> I is CartesianIndex{1}
    @inline function get_xyz(g::UnstructuredGrid, I::CartesianIndex{1})
        i = I[1]
        return g.x[i], g.y[i], g.z[i]
    end

    # Case C: User tries to access with 3D index (Error, or map if you had a 3D->1D map)
    # Intentionally undefined to throw error

    grid_size(g::UnstructuredGrid) = (length(g.x),)
end