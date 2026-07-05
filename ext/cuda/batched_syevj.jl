# cuSOLVER exposes the batched Jacobi eigensolver as `syevjBatched!` for real
# symmetric matrices and `heevjBatched!` for complex Hermitian ones; both return
# `(W, A)` for jobz == 'V' and just `W` for jobz == 'N', with eigenvalues in
# ascending order. Supported for matrix order n ≤ 32.
@inline _cusolver_evj!(::Type{<:Real})    = CUSOLVER.syevjBatched!
@inline _cusolver_evj!(::Type{<:Complex}) = CUSOLVER.heevjBatched!

function NextLA.syevj_batched!(jobz::Char, uplo::Char,
                              A::CUDA.StridedCuArray{T,3}) where {T}
    res = _cusolver_evj!(T)(jobz, uplo, A)
    return jobz == 'V' ? res : (res, A)
end
