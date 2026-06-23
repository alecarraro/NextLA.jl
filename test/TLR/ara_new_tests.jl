using NextLA
using Test
using LinearAlgebra
using Random
using KernelAbstractions

include("helpers.jl")

@testset "ARA Improvements and Fixes" begin
    # 1. Low-rank exact matrix batch
    @testset "Low-rank exact matrix" begin
        m, n = 32, 32
        k = 4
        U_true = randn(Float64, m, k)
        V_true = randn(Float64, n, k)
        A = U_true * V_true'

        layout = NextLA.TileMap(NextLA.TileColMajor(1, 1), m, n, m, n)

        # Test with DirectDenseSamplingState (new path)
        U = zeros(Float64, m, 16, 1)
        V = zeros(Float64, n, 16, 1)
        ranks = zeros(Int, 1)

        NextLA.TLRmodule.ara_batched!(U, V, ranks, A, layout, 16, 8, 1e-10; required_samples=5)

        @test ranks[1] >= k
        A_rec = U[:, 1:ranks[1], 1] * V[:, 1:ranks[1], 1]'
        @test norm(A - A_rec) / norm(A) < 1e-8
    end

    # 2. Varying ranks in same batch
    @testset "Varying ranks in batch" begin
        b = 32
        # Tile 1: rank 2, Tile 2: rank 8, Tile 3: rank 0 (or near 0)
        A1 = randn(b, 2) * randn(b, 2)'
        A2 = randn(b, 8) * randn(b, 8)'
        A3 = zeros(b, b)

        M = zeros(Float64, b, b, 3)
        M[:, :, 1] .= A1
        M[:, :, 2] .= A2
        M[:, :, 3] .= A3

        U = zeros(Float64, b, 16, 3)
        V = zeros(Float64, b, 16, 3)
        ranks = zeros(Int, 3)

        NextLA.TLRmodule.ara_batched!(U, V, ranks, M, 16, 4, 1e-8; required_samples=4)

        @test ranks[1] >= 2
        @test ranks[2] >= 8
        @test ranks[3] == 0

        for i in 1:2
            Mi = M[:, :, i]
            U_i = U[:, 1:ranks[i], i]
            V_i = V[:, 1:ranks[i], i]
            @test norm(Mi - U_i * V_i') / norm(Mi) < 1e-6
        end
    end

    # 3. Block size independence
    @testset "Block size independence" begin
        m, n = 32, 32
        A = randn(m, 4) * randn(n, 4)'
        layout = NextLA.TileMap(NextLA.TileColMajor(1, 1), m, n, m, n)

        results = []
        for bs in [4, 8, 16]
            U = zeros(Float64, m, 16, 1)
            V = zeros(Float64, n, 16, 1)
            ranks = zeros(Int, 1)
            Random.seed!(42)
            NextLA.TLRmodule.ara_batched!(U, V, ranks, A, layout, 16, bs, 1e-8; required_samples=5)
            push!(results, (bs=bs, rank=ranks[1]))
        end
        # Ranks should be similar regardless of block_size
        @test all(r.rank >= 4 for r in results)
        @test all(abs(results[1].rank - r.rank) <= 2 for r in results)
    end

    # 4. Max-rank behavior
    @testset "Max-rank behavior" begin
        m, n = 16, 16
        A = randn(m, n) # Full rank
        layout = NextLA.TileMap(NextLA.TileColMajor(1, 1), m, n, m, n)

        max_rank = 8
        U = zeros(Float64, m, max_rank, 1)
        V = zeros(Float64, n, max_rank, 1)
        ranks = zeros(Int, 1)

        NextLA.TLRmodule.ara_batched!(U, V, ranks, A, layout, max_rank, 4, 1e-12)
        @test ranks[1] == max_rank
    end

    # 5. Complex eltype
    @testset "Complex eltype" begin
        m, n = 16, 16
        k = 4
        A = randn(ComplexF64, m, k) * randn(ComplexF64, n, k)'
        layout = NextLA.TileMap(NextLA.TileColMajor(1, 1), m, n, m, n)

        U = zeros(ComplexF64, m, 8, 1)
        V = zeros(ComplexF64, n, 8, 1)
        ranks = zeros(Int, 1)
        NextLA.TLRmodule.ara_batched!(U, V, ranks, A, layout, 8, 4, 1e-8)

        @test ranks[1] >= k
        A_rec = U[:, 1:ranks[1], 1] * V[:, 1:ranks[1], 1]'
        @test norm(A - A_rec) / norm(A) < 1e-6
    end

    # 6. Dense A boundary tiling
    @testset "Dense A boundary tiling" begin
        # N = 70, b = 32 => 3x3 tiles
        # Tiles: 32x32 (internal), 32x6 (right), 6x32 (bottom), 6x6 (corner)
        N = 70
        b = 32
        A = randn(N, N)
        layout = NextLA.TileMap(NextLA.TileColMajor(3, 3), b, b, N, N)

        max_rank = 16
        U = zeros(Float64, b, max_rank, 9)
        V = zeros(Float64, b, max_rank, 9)
        ranks = zeros(Int, 9)

        NextLA.TLRmodule.ara_batched!(U, V, ranks, A, layout, max_rank, 8, 1e-8)

        # Verify reconstruction of each tile
        for ti in 1:3, tj in 1:3
            p = NextLA.TLRmodule.tile_linear_index(layout.order, ti, tj)
            p0, q0 = NextLA.TLRmodule.tile_origin_coords(layout, ti, tj)
            tm, tn = NextLA.TLRmodule.tile_sizes(layout, ti, tj)

            Atile = A[p0:p0+tm-1, q0:q0+tn-1]
            Utile = U[1:tm, 1:ranks[p], p]
            Vtile = V[1:tn, 1:ranks[p], p]

            if ranks[p] > 0
                @test norm(Atile - Utile * Vtile') / norm(Atile) < 1e-6
            end
        end
    end

    # 7. No packed storage regression
    @testset "No packed storage regression" begin
        m, n = 32, 32
        backend = CPU()
        ptr_type = Ptr{Float64}
        ws = NextLA.TLRmodule._allocate_ara_workspace(backend, ptr_type, Float64, Int, m, n, 1, 16, 8)

        @test !(:Mcompact in fieldnames(typeof(ws)))
        @test !(:Ucompact in fieldnames(typeof(ws)))
    end
end
