using Test
using Random

@testset "TLR dense ARA helpers" begin
    A = Float32.(reshape(1:35, 5, 7))
    mt = 4
    nt = 3
    order = NextLA.TileColMajor(cld(size(A, 1), mt), cld(size(A, 2), nt))
    ndiag = min(size(order)...)
    offdiag_batch_size = prod(size(order)) - ndiag

    @testset "pack!" begin
        for (backend_name, AT, sync) in backends
            @testset "$backend_name" begin
                A_dev = _to_backend(AT, A)
                M = _to_backend(AT, fill(-1.0f0, mt, nt, offdiag_batch_size))
                D = _to_backend(AT, fill(-2.0f0, mt, nt, ndiag))
                layout = NextLA.TileMap(order, mt, nt, size(A, 1), size(A, 2))
                storage = NextLA.PackedTileStorage(M, D, layout)
                NextLA.pack!(storage, A_dev)
                sync(M)
                sync(D)
                M_host = Array(M)
                D_host = Array(D)
                offdiag_batch = 0

                for batch in 1:prod(size(order))
                    tile_i, tile_j = NextLA.inverse_tile_index(order, batch)
                    p0 = (tile_i - 1) * mt + 1
                    q0 = (tile_j - 1) * nt + 1
                    tile_m = min(mt, size(A, 1) - p0 + 1)
                    tile_n = min(nt, size(A, 2) - q0 + 1)

                    if tile_i == tile_j
                        @test D_host[1:tile_m, 1:tile_n, tile_i] == A[p0:(p0 + tile_m - 1), q0:(q0 + tile_n - 1)]
                        if tile_m < mt
                            @test all(iszero, D_host[(tile_m + 1):end, :, tile_i])
                        end
                        if tile_n < nt
                            @test all(iszero, D_host[:, (tile_n + 1):end, tile_i])
                        end
                    else
                        offdiag_batch += 1
                        @test M_host[1:tile_m, 1:tile_n, offdiag_batch] == A[p0:(p0 + tile_m - 1), q0:(q0 + tile_n - 1)]
                        if tile_m < mt
                            @test all(iszero, M_host[(tile_m + 1):end, :, offdiag_batch])
                        end
                        if tile_n < nt
                            @test all(iszero, M_host[:, (tile_n + 1):end, offdiag_batch])
                        end
                    end
                end
            end
        end
    end

    @testset "sample_MtU_packed!" begin
        Random.seed!(1234)
        M = zeros(Float32, mt, nt, offdiag_batch_size)
        D = zeros(Float32, mt, nt, ndiag)
        layout = NextLA.TileMap(order, mt, nt, size(A, 1), size(A, 2))
        storage = NextLA.PackedTileStorage(M, D, layout)
        NextLA.pack!(storage, A)
        U = randn(Float32, mt, 2, offdiag_batch_size)
        V = fill(-99.0f0, nt, 2, offdiag_batch_size)

        NextLA.sample_MtU_packed!(V, M, U)

        for batch in 1:offdiag_batch_size
            @test V[:, :, batch] ≈ M[:, :, batch]' * U[:, :, batch]
        end
    end

end

@testset "TLR ara_batched!" begin
    rng = MersenneTwister(1234)
    mt = 8
    nt = 8
    batch_size = 4
    true_rank = 2
    max_rank = 4
    block_size = 2
    eps = 1.0f-3

    M_host = Array{Float32}(undef, mt, nt, batch_size)
    for batch in 1:batch_size
        L = randn(rng, Float32, mt, true_rank)
        R = randn(rng, Float32, nt, true_rank)
        M_host[:, :, batch] = L * R'
    end

    for (backend_name, AT, sync) in backends
        backend_name == "CPU" && continue

        @testset "$backend_name" begin
            M = _to_backend(AT, M_host)
            U = _to_backend(AT, zeros(Float32, mt, max_rank, batch_size))
            V = _to_backend(AT, zeros(Float32, nt, max_rank, batch_size))
            ranks = AT(zeros(Int32, batch_size))

            NextLA.ara_batched!(
                U,
                V,
                ranks,
                M,
                max_rank,
                block_size,
                eps,
            )
            sync(U)

            U_host = Array(U)
            V_host = Array(V)
            ranks_host = Array(ranks)

            @test all((0 .<= ranks_host) .& (ranks_host .<= max_rank))

            for batch in 1:batch_size
                r = Int(ranks_host[batch])
                M_ref = M_host[:, :, batch]
                M_approx = r == 0 ? zeros(Float32, mt, nt) : U_host[:, 1:r, batch] * V_host[:, 1:r, batch]'
                relerr = opnorm(M_ref - M_approx) / max(opnorm(M_ref), eps)
                @test relerr <= 10f0 * eps
            end
        end
    end
end

@testset "TLR compress wrappers" begin
    rng = MersenneTwister(2026)
    b = 8
    max_rank = 4
    true_rank = 2
    block_size = 2
    eps = 1.0f-3

    A_host = zeros(Float32, 2b, 2b)
    A_host[1:b, 1:b] .= randn(rng, Float32, b, b)
    A_host[(b + 1):end, (b + 1):end] .= randn(rng, Float32, b, b)

    L12 = randn(rng, Float32, b, true_rank)
    R12 = randn(rng, Float32, b, true_rank)
    L21 = randn(rng, Float32, b, true_rank)
    R21 = randn(rng, Float32, b, true_rank)
    A_host[1:b, (b + 1):end] .= L12 * R12'
    A_host[(b + 1):end, 1:b] .= L21 * R21'

    for (backend_name, AT, sync) in backends
        backend_name == "CPU" && continue

        @testset "$backend_name" begin
            A_dev = _to_backend(AT, A_host)

            for (label, compress_fn) in (
                ("packed", NextLA.compress!),
                ("hybrid", NextLA.compress_hybrid!),
            )
                @testset "$label" begin
                    A_tlr = NextLA.TLRMatrix(A_dev, b, max_rank; compress_diag=false)
                    compress_fn(A_tlr, A_dev; block_size, eps)

                    sync(A_tlr.U)
                    sync(A_tlr.V)
                    sync(A_tlr.diag)

                    U_host = Array(A_tlr.U)
                    V_host = Array(A_tlr.V)
                    D_host = Array(A_tlr.diag)
                    ranks_host = Array(A_tlr.ranks)

                    @test D_host[:, :, 1] == A_host[1:b, 1:b]
                    @test D_host[:, :, 2] == A_host[(b + 1):end, (b + 1):end]

                    for (tile_i, tile_j, tile_ref) in (
                        (1, 2, A_host[1:b, (b + 1):end]),
                        (2, 1, A_host[(b + 1):end, 1:b]),
                    )
                        batch = NextLA.tile_rank_index(A_tlr, tile_i, tile_j)
                        r = Int(ranks_host[batch])
                        tile_approx = r == 0 ? zeros(Float32, b, b) : U_host[:, 1:r, batch] * V_host[:, 1:r, batch]'
                        relerr = opnorm(tile_ref - tile_approx) / max(opnorm(tile_ref), eps)
                        @test r == true_rank
                        @test relerr <= 10f0 * eps
                    end
                end
            end
        end
    end
end
