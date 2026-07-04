using Test
using LinearAlgebra
using Random

# Reconstruct one tile from an ARA factor batch: A_b ≈ Q_b[:,1:r] * B_b[:,1:r]'.
function ara_reconstruct(Q, B, ranks, b)
    r = ranks[b]
    r == 0 && return zeros(eltype(Q), size(Q, 1), size(B, 1))
    return Array(Q)[:, 1:r, b] * Array(B)[:, 1:r, b]'
end

# Dense m×n tile with prescribed singular values.
function tile_with_svals(::Type{T}, m, n, σ; seed) where {T}
    rng = MersenneTwister(seed)
    k = length(σ)
    U = Matrix(qr(randn(rng, T, m, k)).Q)
    V = Matrix(qr(randn(rng, T, n, k)).Q)
    return U * Diagonal(T.(σ)) * V'
end

@testset "ARA ara_compress on CPU" begin
    @testset "single dense tile of known low rank" begin
        Random.seed!(2024)
        m = n = 48
        r0 = 6
        A = tile_with_svals(Float64, m, n, range(2.0, 1.0; length=r0); seed=7)
        Abatch = reshape(A, m, n, 1)
        bs = 4
        ε = 1e-8

        Q, B, ranks, failed = NextLA.ara_compress(Abatch, bs, ε, 32)

        @test !failed[1]
        # rank ≈ r0 within block + one detection block of oversampling slack
        @test r0 <= ranks[1] <= r0 + 2 * bs
        relerr = norm(ara_reconstruct(Q, B, ranks, 1) - A) / norm(A)
        @test relerr <= ε
        # Q columns are orthonormal.
        Qr = Array(Q)[:, 1:ranks[1], 1]
        @test norm(Qr' * Qr - I) <= 1e-8
    end

    @testset "slowly decaying spectrum: rank grows, error tracks ε" begin
        # σ_i = 1/i decays slowly, so the ARA convergence signal (a spectral-norm
        # proxy) drops roughly like ε ⇒ the returned rank grows ~1/ε and the
        # reconstruction error shrinks monotonically as ε tightens.
        m = n = 64
        kfull = min(m, n)
        A = tile_with_svals(Float64, m, n, [1.0 / i for i in 1:kfull]; seed=13)
        Abatch = reshape(A, m, n, 1)
        bs = 4
        r_max = 48

        ranks_seen = Int[]
        errs_seen = Float64[]
        for ε in (1e-1, 3e-2, 1e-2)
            Random.seed!(1000)
            Q, B, ranks, failed = NextLA.ara_compress(Abatch, bs, ε, r_max)
            @test !failed[1]
            push!(ranks_seen, ranks[1])
            push!(errs_seen, norm(ara_reconstruct(Q, B, ranks, 1) - A) / norm(A))
        end
        # Tighter ε ⇒ more retained rank and smaller error.
        @test issorted(ranks_seen)
        @test ranks_seen[end] > ranks_seen[1]
        @test issorted(errs_seen; rev=true)
        @test errs_seen[end] < errs_seen[1]

        # A tight tolerance drives the rank up to fill r_max.
        Random.seed!(1000)
        _, _, ranks_tight, _ = NextLA.ara_compress(Abatch, bs, 1e-4, r_max)
        @test ranks_tight[1] >= r_max - bs
    end

    @testset "ill-conditioned sketch engages Cholesky shift without crashing" begin
        Random.seed!(4242)
        m = n = 32
        # Rank-1 tile: every sketch block Y = A·Ω is rank-1, so the CholQR Gram
        # matrix is singular and the diagonal shift is what keeps it factorable.
        A = tile_with_svals(Float64, m, n, [3.0]; seed=5)
        Abatch = reshape(A, m, n, 1)
        bs = 8
        ε = 1e-8

        Q, B, ranks, failed = NextLA.ara_compress(Abatch, bs, ε, 16)
        @test !failed[1]
        @test ranks[1] >= 1
        Qr = Array(Q)[:, 1:ranks[1], 1]
        @test norm(Qr' * Qr - I) <= 1e-6          # still a valid orthonormal basis
        relerr = norm(ara_reconstruct(Q, B, ranks, 1) - A) / norm(A)
        @test relerr <= ε
    end

    @testset "batch matches loop-of-single-tile reference" begin
        m = n = 40
        bs = 4
        ε = 1e-8
        r_max = 32
        ranks0 = [3, 7, 5, 10]
        tiles = [tile_with_svals(Float64, m, n, range(2.0, 1.0; length=r0); seed=100 + i)
                 for (i, r0) in enumerate(ranks0)]
        Abatch = cat(tiles...; dims=3)

        Random.seed!(555)
        Qb, Bb, ranks_b, failed_b = NextLA.ara_compress(Abatch, bs, ε, r_max)

        for (i, A) in enumerate(tiles)
            # Loop reference: same routine, one tile at a time.
            Random.seed!(555 + i)
            Qs, Bs, ranks_s, failed_s =
                NextLA.ara_compress(reshape(A, m, n, 1), bs, ε, r_max)

            @test !failed_b[i]
            @test !failed_s[1]
            relerr_b = norm(ara_reconstruct(Qb, Bb, ranks_b, i) - A) / norm(A)
            relerr_s = norm(ara_reconstruct(Qs, Bs, ranks_s, 1) - A) / norm(A)
            @test relerr_b <= ε                       # batching did not corrupt this tile
            @test relerr_s <= ε
            @test abs(ranks_b[i] - ranks_s[1]) <= bs  # same rank up to block granularity
        end
    end

    @testset "vector-of-matrices input" begin
        Random.seed!(31)
        m = n = 36
        tiles = [tile_with_svals(Float64, m, n, range(2.0, 1.0; length=r0); seed=200 + r0)
                 for r0 in (4, 8)]
        Q, B, ranks, failed = NextLA.ara_compress(tiles, 4, 1e-8, 24)
        for (i, A) in enumerate(tiles)
            @test !failed[i]
            @test norm(ara_reconstruct(Q, B, ranks, i) - A) / norm(A) <= 1e-8
        end
    end
end

@testset "ARA ara_compress! into TLRMatrix on CPU" begin
    fixture = canonical_dense_fixture(Float64)
    A_tlr = NextLA.TLRMatrix(fixture.A, fixture.b, 16)
    NextLA.ara_compress!(A_tlr, fixture.A; tol=1e-8, bs=4)

    relerr = norm(reconstruct_tlr(A_tlr) - fixture.A) / norm(fixture.A)
    @test relerr <= 1e-6
    assert_tile_rank_and_error(A_tlr, 1, 2, 8, fixture.offdiag12; atol_rank=4, rtol_error=1e-6)

    boundary = boundary_dense_fixture(Float64)
    A_panel = NextLA.TLRMatrix(boundary.A, 4, 3)
    NextLA.ara_compress!(A_panel, boundary.A; tol=1e-8, bs=2)
    relerr_panel = norm(reconstruct_tlr(A_panel) - boundary.A) / norm(boundary.A)
    @test relerr_panel <= 1e-6
end

@testset "ARA ara_compress on GPU" begin
    for (backend_name, ArrayType, synchronize) in available_backends()
        backend_name in ("CUDA", "AMDGPU") || continue
        @testset "$backend_name" begin
            Random.seed!(7)
            m = n = 48
            r0 = 5
            A = tile_with_svals(Float32, m, n, range(2.0f0, 1.0f0; length=r0); seed=17)
            tiles = [A, tile_with_svals(Float32, m, n, range(2.0f0, 1.0f0; length=9); seed=18)]
            Abatch = ArrayType(cat(tiles...; dims=3))
            bs = 4
            ε = 1f-4

            Q, B, ranks, failed = NextLA.ara_compress(Abatch, bs, ε, 32)
            synchronize(Q)

            for (i, tile) in enumerate(tiles)
                @test !failed[i]
                relerr = norm(ara_reconstruct(Q, B, ranks, i) - tile) / norm(tile)
                @test relerr <= 5f-3
                Qr = Array(Q)[:, 1:ranks[i], i]
                @test norm(Qr' * Qr - I) <= 1f-2
            end
        end
    end
end
