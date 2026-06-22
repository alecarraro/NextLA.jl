#=
This file contains an experimental path with a hybrid packed-fused ara_batched kernel
=#
"""
    sample_MtU_packed!(V, M, U)

Compute the packed batched adjoint product `V = M' * U`, where `M` stores the
compact off-diagonal tile batch.
"""
function sample_MtU_packed!(
    V::AbstractArray{T,3},
    M::AbstractArray{T,3},
    U::AbstractArray{T,3},
) where {T}
    size(M, 1) == size(U, 1) || throw(DimensionMismatch("M and U must agree in their row dimension"))
    size(M, 2) == size(V, 1) || throw(DimensionMismatch("M columns must match the rows of V"))
    size(M, 3) == size(U, 3) && size(U, 3) == size(V, 3) ||
        throw(DimensionMismatch("M, U, and V must have the same batch size"))

    gemm_batched!('T', 'N', one(T), M, U, zero(T), V)
    return V
end

"""
    sample_MtU!(V, A, U, layout; ...)

Materialize the packed off-diagonal tiles of `A` and compute `V = M' * U`.
This is a convenience wrapper around [`pack!`](@ref) and
[`sample_MtU_packed!`](@ref).
"""
function sample_MtU!(
    V::AbstractArray{T,3},
    A::AbstractMatrix{T},
    U::AbstractArray{T,3},
    layout::TileMap;
    ROWS_PER_WORKGROUP::Int=256,
    NWORKGROUPS::Int=256,
    backend=KernelAbstractions.get_backend(A),
) where {T}
    noffdiag = size(V, 3)
    M = allocate(backend, T, layout.tile_m, layout.tile_n, noffdiag)
    D = allocate(backend, T, layout.tile_m, layout.tile_n, ndiag_tiles(layout))

    pack!(
        PackedTileStorage(M, D, layout),
        A;
        ROWS_PER_WORKGROUP,
        NWORKGROUPS,
        backend,
    )
    return sample_MtU_packed!(V, M, U)
end

function sample_MtU!(
    V::AbstractArray{T,3},
    A::AbstractMatrix{T},
    U::AbstractArray{T,3},
    tile_order::TileOrder,
    mt::Int,
    nt::Int;
    ROWS_PER_WORKGROUP::Int=256,
    NWORKGROUPS::Int=256,
    backend=KernelAbstractions.get_backend(A),
) where {T}
    layout = TileMap(tile_order, mt, nt, size(A, 1), size(A, 2))
    return sample_MtU!(
        V,
        A,
        U,
        layout;
        ROWS_PER_WORKGROUP,
        NWORKGROUPS,
        backend,
    )
end

function ara_batched_dense!(
    U::AbstractArray{T,3},
    V::AbstractArray{T,3},
    ranks::AbstractVector{RankT},
    M::AbstractArray{T,3},
    A::AbstractMatrix{T},
    layout::TileMap,
    max_rank::Int,
    block_size::Int,
    eps;
    seed::Integer=0x123456789abcdef0,
    nthreads::Int=256,
    ROWS_CTA::Int=128,
    BK::Int=16,
    SBLK::Int=32,
    CPW::Int=4,
    use_batch_in_rng::Bool=true,
) where {T,RankT<:Integer}

    get_backend(U) == get_backend(V) == get_backend(M) == get_backend(A) ||
        throw(ArgumentError("U, V, M, and A must have the same backend"))

    backend = get_backend(M)
    batch_size = size(M, 3)
    expected_batch_size = noffdiag_tiles(layout)

    batch_size == expected_batch_size ||
        throw(DimensionMismatch("packed off-diagonal batch size must be $expected_batch_size"))
    size(U, 1) == layout.tile_m || throw(DimensionMismatch("U must have $(layout.tile_m) rows"))
    size(V, 1) == layout.tile_n || throw(DimensionMismatch("V must have $(layout.tile_n) rows"))
    size(U, 2) >= max_rank || throw(DimensionMismatch("U must have at least $max_rank columns"))
    size(V, 2) >= max_rank || throw(DimensionMismatch("V must have at least $max_rank columns"))
    size(U, 3) == batch_size || throw(DimensionMismatch("U batch size must match M"))
    size(V, 3) == batch_size || throw(DimensionMismatch("V batch size must match M"))
    length(ranks) == batch_size || throw(DimensionMismatch("ranks length must match the batch size"))

    max_rank > 0 || throw(ArgumentError("max_rank must be positive"))
    block_size > 0 || throw(ArgumentError("block_size must be positive"))

    RT = typeof(real(zero(T)))
    eps_rt = RT(eps)^2
    required_small = block_size

    detected_rank = allocate(backend, RankT, batch_size)
    small_sample_count = allocate(backend, Int32, batch_size)
    active_batches = allocate(backend, Bool, batch_size)
    sample_scale = allocate(backend, RT, batch_size)
    active_idx_dev = allocate(backend, Int, batch_size)

    fill!(detected_rank, zero(RankT))
    fill!(small_sample_count, Int32(0))
    fill!(active_batches, true)
    fill!(sample_scale, zero(RT))
    copyto!(active_idx_dev, collect(1:batch_size))

    active_host = trues(batch_size)

    max_block = min(block_size, max_rank)
    Y = allocate(backend, T, layout.tile_m, max_block, batch_size)
    Z = allocate(backend, T, max_rank, max_block, batch_size)
    G = allocate(backend, T, max_block, max_block, batch_size)
    potrf_status = allocate(backend, Int32, batch_size)

    convergence_kernel! = convergence_ara_kernel!(backend)
    initialize_scale_kernel! = initialize_ara_scale_kernel!(backend)
    finalize_kernel! = finalize_ara_kernel!(backend)
    zero_inactive_kernel! = zero_inactive_ara_kernel!(backend)
    regularize_inactive_gram_kernel! = regularize_inactive_gram_ara_kernel!(backend)

    j = 0

    while j < max_rank && any(active_host)
        sample_size = min(block_size, max_rank - j)
        cols = (j + 1):(j + sample_size)
        j_before = j

        Y_block = @view Y[:, 1:sample_size, :]
        G_block = @view G[1:sample_size, 1:sample_size, :]

        sample_range_rademacher!(
            Y_block,
            A,
            layout;
            S=sample_size,
            seed=_ara_iteration_seed(seed, j_before),
            nthreads,
            ROWS_CTA,
            BK,
            SBLK,
            CPW,
            use_batch_in_rng,
            offdiag_only=true,
            backend,
        )

        zero_inactive_kernel!(
            Y_block,
            active_batches,
            active_idx_dev,
            batch_size;
            ndrange=(layout.tile_m, sample_size, batch_size),
        )

        for reorth_pass in 1:2
            if j > 0
                U_prev = @view U[:, 1:j, :]
                Z_prev = @view Z[1:j, 1:sample_size, :]

                gemm_batched!('T', 'N', one(T), U_prev, Y_block, zero(T), Z_prev)
                gemm_batched!('N', 'N', -one(T), U_prev, Z_prev, one(T), Y_block)
            end

            syrk_batched!('L', 'T', one(T), Y_block, zero(T), G_block)

            if j_before == 0 && reorth_pass == 1
                initialize_scale_kernel!(
                    sample_scale,
                    G_block,
                    active_idx_dev,
                    sample_size,
                    batch_size;
                    ndrange=batch_size,
                )
            end

            convergence_kernel!(
                active_batches,
                detected_rank,
                small_sample_count,
                sample_scale,
                G_block,
                active_idx_dev,
                eps_rt,
                j_before,
                sample_size,
                required_small,
                batch_size;
                ndrange=batch_size,
            )

            regularize_inactive_gram_kernel!(
                G_block,
                active_batches,
                active_idx_dev,
                batch_size;
                ndrange=(sample_size, sample_size, batch_size),
            )

            zero_inactive_kernel!(
                Y_block,
                active_batches,
                active_idx_dev,
                batch_size;
                ndrange=(layout.tile_m, sample_size, batch_size),
            )

            potrf_batched!('L', G_block, potrf_status)

            trsm_batched!('R', 'L', 'T', 'N', G_block, Y_block)
        end

        copyto!(@view(U[:, cols, :]), Y_block)

        j += sample_size
        active_host .= Array(active_batches)
    end

    finalize_kernel!(
        ranks,
        active_batches,
        detected_rank,
        j,
        batch_size;
        ndrange=batch_size,
    )

    if j > 0
        U_used = @view U[:, 1:j, :]
        V_used = @view V[:, 1:j, :]
        gemm_batched!('T', 'N', one(T), M, U_used, zero(T), V_used)
    end

    return U, V, ranks
end

function ara_batched_dense!(
    U::AbstractArray{T,3},
    V::AbstractArray{T,3},
    ranks::AbstractVector{RankT},
    M::AbstractArray{T,3},
    A::AbstractMatrix{T},
    tile_order::TileOrder,
    mt::Int,
    nt::Int,
    max_rank::Int,
    block_size::Int,
    eps;
    kwargs...,
) where {T,RankT<:Integer}
    layout = TileMap(tile_order, mt, nt, size(A, 1), size(A, 2))
    return ara_batched_dense!(U, V, ranks, M, A, layout, max_rank, block_size, eps; kwargs...)
end
