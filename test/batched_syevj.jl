using Test
using LinearAlgebra
using Random

# Build a batch of Hermitian n×n matrices with known spectra as a [n,n,nb] array.
function _hermitian_batch(::Type{T}, n, nb; seed=11) where {T}
    rng = MersenneTwister(seed)
    A = Array{T}(undef, n, n, nb)
    for b in 1:nb
        M = randn(rng, T, n, n)
        A[:, :, b] = (M + M') / 2
    end
    return A
end

@testset "batched syevj" begin
    for (name, AT, sync) in backends
        @testset "$name" begin
            for T in (Float32, Float64)
                # cuSOLVER's batched Jacobi caps the order at 32.
                for n in (6, 16, 32)
                    nb = 4
                    A = _hermitian_batch(T, n, nb)
                    refs = [eigen(Hermitian(A[:, :, b])) for b in 1:nb]

                    # jobz = 'V': eigenvalues (ascending) + eigenvectors in place
                    Adev = _to_backend(AT, copy(A))
                    W, V = NextLA.syevj_batched!('V', 'U', Adev)
                    @test V === Adev                     # decomposition is in place
                    Wh = Array(W)
                    Vh = Array(Adev)

                    rtol = T == Float32 ? 1f-3 : 1e-9
                    for b in 1:nb
                        # eigenvalues match the reference, ascending
                        @test issorted(Wh[:, b])
                        @test isapprox(Wh[:, b], refs[b].values; rtol=rtol,
                                       atol=rtol * max(1, maximum(abs, refs[b].values)))
                        # eigenvectors reconstruct the matrix: V Λ Vᴴ ≈ A
                        Vb = Vh[:, :, b]
                        recon = Vb * Diagonal(Wh[:, b]) * Vb'
                        @test isapprox(recon, A[:, :, b]; rtol=rtol, atol=rtol)
                    end

                    # jobz = 'N': eigenvalues only, input triangle untouched
                    Adev2 = _to_backend(AT, copy(A))
                    Wn, _ = NextLA.syevj_batched!('N', 'U', Adev2)
                    Wnh = Array(Wn)
                    for b in 1:nb
                        @test isapprox(Wnh[:, b], refs[b].values; rtol=rtol,
                                       atol=rtol * max(1, maximum(abs, refs[b].values)))
                    end
                end
            end
        end
    end
end
