@inline _rocsolver_syevj_strided_batched_fname(::Type{Float32})    = rocSOLVER.rocsolver_ssyevj_strided_batched
@inline _rocsolver_syevj_strided_batched_fname(::Type{Float64})    = rocSOLVER.rocsolver_dsyevj_strided_batched
@inline _rocsolver_syevj_strided_batched_fname(::Type{ComplexF32}) = rocSOLVER.rocsolver_cheevj_strided_batched
@inline _rocsolver_syevj_strided_batched_fname(::Type{ComplexF64}) = rocSOLVER.rocsolver_zheevj_strided_batched

function _syevj_batched_amdgpu!(jobz::Char, uplo::Char,
                                A::AMDGPU.StridedROCArray{T,3}) where {T}
    n = LinearAlgebra.checksquare(@view A[:, :, 1])
    lda = max(1, stride(A, 2))
    strideA = stride(A, 3)
    batch_count = size(A, 3)
    RT = real(T)

    W        = similar(A, RT, n, batch_count)     # eigenvalues, strideW = n
    residual = similar(A, RT, batch_count)
    n_sweeps = similar(A, Int32, batch_count)
    info     = similar(A, Int32, batch_count)

    esort = rocSOLVER.rocblas_esort_ascending
    evect = jobz == 'V' ? rocSOLVER.rocblas_evect_original : rocSOLVER.rocblas_evect_none
    abstol   = RT(eps(RT))
    max_sweeps = Int32(100)

    fname = _rocsolver_syevj_strided_batched_fname(T)
    fname(rocBLAS.handle(), esort, evect, uplo, n, A, lda, strideA,
          abstol, residual, max_sweeps, n_sweeps, W, n, info, batch_count)
    return W, A
end

function NextLA.syevj_batched!(jobz::Char, uplo::Char,
                              A::AMDGPU.StridedROCArray{T,3}) where {T}
    return _syevj_batched_amdgpu!(jobz, uplo, A)
end
