"""
    compress_hybrid!(A_tlr, A; block_size=min(A_tlr.maxrank, 32), eps, ...)

Like [`compress!`](@ref), but keep the packed off-diagonal tiles only for the
final `M' * U` step while using `sample_range_rademacher!` on the dense matrix
for the forward sampling stage inside ARA.
"""
function compress_hybrid!(
    A_tlr::TLRMatrix{<:Any,T,RankT},
    A::AbstractMatrix{T};
    block_size::Int=min(A_tlr.maxrank, 32),
    eps,
    ROWS_PER_WORKGROUP::Int=256,
    NWORKGROUPS::Int=256,
    seed::Integer=0x123456789abcdef0,
    nthreads::Int=256,
    ROWS_CTA::Int=128,
    BK::Int=16,
    SBLK::Int=32,
    CPW::Int=4,
    use_batch_in_rng::Bool=true,
) where {T,RankT<:Integer}
    _require_compressible_tlr(A_tlr, A)

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

    ara_batched_dense!(
        A_tlr.U,
        A_tlr.V,
        A_tlr.ranks,
        storage.offdiag,
        A,
        storage.layout,
        A_tlr.maxrank,
        block_size,
        eps;
        seed,
        nthreads,
        ROWS_CTA,
        BK,
        SBLK,
        CPW,
        use_batch_in_rng,
    )

    return A_tlr
end
