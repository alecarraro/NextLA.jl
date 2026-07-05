export syevj_batched!

"""
    syevj_batched!(jobz, uplo, A) -> (W, A)

Compute an in-place batched Hermitian eigendecomposition of `A` with the
one-sided Jacobi method.

`A` must be a three-dimensional array whose slices `A[:, :, b]` are decomposed
independently. Each slice is overwritten by its eigenvectors when `jobz == 'V'`
(and left untouched when `jobz == 'N'`). Eigenvalues are returned in `W`, an
`n × batch` array whose column `b` holds the eigenvalues of `A[:, :, b]` in
**ascending** order. `uplo` (`'U'` or `'L'`) selects which triangle of each
slice is referenced.

This entry point is reserved for backend batched-library dispatch. CPU arrays
fall back to a LAPACK/`eigen` loop; the CUDA and AMDGPU extensions override it
with native batched Jacobi solvers (`cusolverDn{S,D}syevjBatched` /
`cusolverDn{C,Z}heevjBatched` and `rocsolver_{s,d}syevj_strided_batched` /
`rocsolver_{c,z}heevj_strided_batched`). Every backend returns `(W, A)`.

!!! note
    cuSOLVER's batched Jacobi solver supports matrices of order `n ≤ 32` only.
"""
function syevj_batched!(jobz::Char, uplo::Char, A::AbstractArray{T,3}) where {T}
    backend = KernelAbstractions.get_backend(A)
    backend isa KernelAbstractions.CPU || throw(ArgumentError("NextLA.syevj_batched! has no generic non-CPU implementation; use a backend wrapper"))
    (jobz == 'N' || jobz == 'V') || throw(ArgumentError("jobz must be 'N' or 'V'"))
    (uplo == 'U' || uplo == 'L') || throw(ArgumentError("uplo must be 'U' or 'L'"))
    n = LinearAlgebra.checksquare(view(A, :, :, 1))
    RT = real(T)
    W = similar(A, RT, n, size(A, 3))
    ul = uplo == 'U' ? (:U) : (:L)
    @inbounds for b in axes(A, 3)
        Ab = view(A, :, :, b)
        F = eigen(Hermitian(Matrix(Ab), ul))   # ascending eigenvalues
        W[:, b] .= F.values
        if jobz == 'V'
            Ab .= F.vectors
        end
    end
    return W, A
end
