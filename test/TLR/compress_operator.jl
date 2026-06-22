using Test
using LinearAlgebra

function _reconstruct_tlr(A_tlr::NextLA.TLRMatrix)
    T = eltype(A_tlr)
    A = zeros(T, size(A_tlr))

    for linear in 1:prod(size(A_tlr.layout))
        tile_i, tile_j, p0, q0, tile_m, tile_n = NextLA.tile_geometry(A_tlr.layout, linear)
        rows = p0:(p0 + tile_m - 1)
        cols = q0:(q0 + tile_n - 1)

        tile = if !A_tlr.compress_diag && tile_i == tile_j
            @view A_tlr.diag[1:tile_m, 1:tile_n, tile_i]
        else
            batch = A_tlr.compress_diag ? linear : NextLA.tile_rank_index(A_tlr, tile_i, tile_j)
            r = Int(A_tlr.ranks[batch])
            r == 0 ? zeros(T, tile_m, tile_n) :
                Matrix(@view(A_tlr.U[1:tile_m, 1:r, batch])) *
                Matrix(adjoint(@view(A_tlr.V[1:tile_n, 1:r, batch])))
        end

        A[rows, cols] .= tile
    end

    return A
end

@testset "TLR operator compression" begin
    Uglob = Float32[1 2; 3 1; 0 1; 2 -1; 1 0; -2 1; 1 -1; 0 2; 1 1; -1 0]
    Vglob = Float32[2 1; -1 1; 1 0; 0 2; 1 -1; 2 0; -1 2; 1 1; 0 1; 2 -1]
    A = Uglob * Vglob'
    op = NextLA.TLRLinearOperator(
        Float32,
        size(A, 1),
        size(A, 2),
        (Y, X) -> mul!(Y, A, X),
        (Y, X) -> mul!(Y, adjoint(A), X),
    )

    for tile_order in (NextLA.TileColMajor, NextLA.TileRowMajor)
        @testset "compress_diag=false ($(tile_order))" begin
            A_tlr = NextLA.TLRMatrix(A, 4, 2; compress_diag=false, tile_order)
            NextLA.compress!(A_tlr, op; block_size=2, eps=1f-5, seed=0x1234)

            A_reconstructed = _reconstruct_tlr(A_tlr)
            @test norm(A_reconstructed - A) / norm(A) ≤ 5f-4
            @test A_tlr.diag[1:4, 1:4, 1] ≈ A[1:4, 1:4]
            @test A_tlr.diag[1:4, 1:4, 2] ≈ A[5:8, 5:8]
            @test A_tlr.diag[1:2, 1:2, 3] ≈ A[9:10, 9:10]
        end

        @testset "compress_diag=true ($(tile_order))" begin
            A_tlr = NextLA.TLRMatrix(A, 4, 2; compress_diag=true, tile_order)
            NextLA.compress!(A_tlr, op; block_size=2, eps=1f-5, seed=0x1234)

            A_reconstructed = _reconstruct_tlr(A_tlr)
            @test norm(A_reconstructed - A) / norm(A) ≤ 5f-4
        end
    end
end
