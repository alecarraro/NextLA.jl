@inline function _allocate_packed_uniform_tiles(A_tlr::TLRMatrix{<:Any,T}) where {T}
    layout = A_tlr.layout
    M = allocate(A_tlr.backend, T, A_tlr.b, A_tlr.b, noffdiag_tiles(layout))
    D = allocate(A_tlr.backend, T, A_tlr.b, A_tlr.b, ndiag_tiles(layout))
    return PackedTileStorage(M, D, layout)
end

@inline function _require_compressible_tlr(A_tlr::TLRMatrix{<:Any,T},
                                           A::AbstractMatrix{T}) where {T}
    get_backend(A) == A_tlr.backend || throw(ArgumentError("dense matrix and TLRMatrix must have the same backend"))
    size(A) == size(A_tlr) || throw(DimensionMismatch("dense matrix size must match the TLRMatrix"))
    return A
end

@inline function _require_compressible_tlr(A_tlr::TLRMatrix{<:Any,T}, op) where {T}
    applicable(size, op) || throw(ArgumentError("operator must define size(op)"))
    applicable(eltype, op) || throw(ArgumentError("operator must define eltype(op)"))
    applicable(adjoint, op) || throw(ArgumentError("operator must define adjoint(op)"))
    applicable(LinearAlgebra.mul!, allocate(A_tlr.backend, T, size(A_tlr, 1), 1), op, allocate(A_tlr.backend, T, size(A_tlr, 2), 1)) ||
        throw(ArgumentError("operator must support mul!(Y, op, X)"))
    applicable(LinearAlgebra.mul!, allocate(A_tlr.backend, T, size(A_tlr, 2), 1), adjoint(op), allocate(A_tlr.backend, T, size(A_tlr, 1), 1)) ||
        throw(ArgumentError("operator must support mul!(Y, adjoint(op), X)"))

    size(op) == size(A_tlr) || throw(DimensionMismatch("operator size must match the TLRMatrix"))
    eltype(op) == T || throw(ArgumentError("operator element type must match the TLRMatrix element type"))
    return op
end

"""
    compress!(A_tlr, A; block_size=min(A_tlr.maxrank, 32), eps, ROWS_PER_WORKGROUP=256, NWORKGROUPS=256)

Pack the dense matrix `A` into the tile layout owned by `A_tlr`, run
`ara_batched!` on the stored tile batch, and write the resulting factors
directly into `A_tlr`.
"""
function compress!(
    A_tlr::TLRMatrix{<:Any,T,RankT},
    A::AbstractMatrix{T};
    block_size::Int=min(A_tlr.maxrank, 32),
    eps,
    ROWS_PER_WORKGROUP::Int=256,
    NWORKGROUPS::Int=256,
) where {T,RankT<:Integer}
    _require_compressible_tlr(A_tlr, A)

    if A_tlr.compress_diag
        tiles = allocate(A_tlr.backend, T, A_tlr.b, A_tlr.b, prod(size(A_tlr.layout)))
        pack!(
            tiles,
            A,
            A_tlr.layout;
            ROWS_PER_WORKGROUP,
            NWORKGROUPS,
            backend=A_tlr.backend,
        )

        if size(tiles, 3) == 0
            fill!(A_tlr.ranks, zero(RankT))
            return A_tlr
        end

        ara_batched!(
            A_tlr.U,
            A_tlr.V,
            A_tlr.ranks,
            tiles,
            A_tlr.maxrank,
            block_size,
            eps,
        )
        return A_tlr
    end

    storage = _allocate_packed_uniform_tiles(A_tlr)
    pack!(
        storage,
        A;
        ROWS_PER_WORKGROUP,
        NWORKGROUPS,
        backend=A_tlr.backend,
    )
    copyto!(A_tlr.diag, storage.diag)

    noffdiag = size(storage.offdiag, 3)
    if noffdiag == 0
        fill!(A_tlr.ranks, zero(RankT))
        return A_tlr
    end

    ara_batched!(
        A_tlr.U,
        A_tlr.V,
        A_tlr.ranks,
        storage.offdiag,
        A_tlr.maxrank,
        block_size,
        eps,
    )

    return A_tlr
end

"""
    compress!(A_tlr, op; block_size=min(A_tlr.maxrank, 32), eps, seed=0x123456789abcdef0)

Compress a matrix-free operator `op` into `A_tlr`. The operator must support
`size(op)`, `eltype(op)`, and in-place block application via `mul!(Y, op, X)`.

This implementation uses the operator directly during adaptive randomized
sampling and only materializes exact diagonal tiles when `compress_diag=false`.
"""
function compress!(
    A_tlr::TLRMatrix{<:Any,T,RankT},
    op;
    block_size::Int=min(A_tlr.maxrank, 32),
    eps,
    seed::Integer=0x123456789abcdef0,
) where {T,RankT<:Integer}
    _require_compressible_tlr(A_tlr, op)

    if !A_tlr.compress_diag
        _materialize_diag_tiles!(A_tlr.diag, op, A_tlr.layout; backend=A_tlr.backend)
    end

    if length(A_tlr.ranks) == 0
        fill!(A_tlr.ranks, zero(RankT))
        return A_tlr
    end

    ara_batched_operator!(
        A_tlr.U,
        A_tlr.V,
        A_tlr.ranks,
        op,
        A_tlr.layout,
        A_tlr.maxrank,
        block_size,
        eps;
        seed,
        compress_diag=A_tlr.compress_diag,
        backend=A_tlr.backend,
    )

    return A_tlr
end
