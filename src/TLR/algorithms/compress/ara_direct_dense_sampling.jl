using Random

"""Direct sampling state from a dense matrix without packing tiles."""
struct DirectDenseSamplingState{MT<:AbstractMatrix, LM<:TileMap}
    A::MT
    layout::LM
end

@kernel function build_dense_A_ptrs_kernel!(
    ptrs,
    base_ptr,
    stride1::Int,
    stride2::Int,
    layout,
    active_idx,
    n_active::Int,
)
    p = @index(Global)
    if p <= n_active
        tile_linear = Int(@inbounds active_idx[p])
        tile_i, tile_j = inverse_tile_index(layout, tile_linear)
        p0, q0 = tile_origin_coords(layout, tile_i, tile_j)

        # Julia is 1-indexed, column major
        offset = ((q0 - 1) * stride2 + (p0 - 1) * stride1) * sizeof(eltype(base_ptr))
        @inbounds ptrs[p] = convert(eltype(ptrs), reinterpret(UInt, base_ptr) + UInt(offset))
    end
end

@kernel function filter_wave_indices_kernel!(
    wave_active_idx,
    wave_slots,
    wave_count_ptr,
    active_idx,
    layout,
    tm::Int,
    tn::Int,
    n_active::Int,
)
    p = @index(Global)
    if p <= n_active
        tile_linear = Int(@inbounds active_idx[p])
        ti, tj = inverse_tile_index(layout, tile_linear)
        stm, stn = tile_sizes(layout, ti, tj)
        if stm == tm && stn == tn
            # Atomic increment for count
            # idx is 1-based because atomic_add! returns the value BEFORE increment
            # and we initialized wave_count to 0.
            # So the first thread gets 0. We need 1.
            idx = KernelAbstractions.Extras.atomic_add!(wave_count_ptr, 1, 1) + 1
            @inbounds wave_active_idx[idx] = tile_linear
            @inbounds wave_slots[idx] = p
        end
    end
end

function _sample_range!(
    Y_current::AbstractArray{T,3},
    source::DirectDenseSamplingState,
    ws::ARAWorkspace,
    active_idx,
    n_active::Int,
    sample_size::Int,
    dense::Bool,
    backend,
) where {T}
    Omega_active = @view ws.Omega[:, 1:sample_size, 1:n_active]
    Random.randn!(Omega_active)

    # Group by tile shape: bxb, bxr, rxb, rxr
    # Since we need to know the shape for each tile, we might need to partition active_idx
    # but for a first implementation, we can just iterate over waves.

    # Actually, the user suggested 3-4 waves based on unique shapes.
    # Let's find unique shapes among ALL tiles in the layout first.

    # For a TileMap, shapes are determined by tile_i, tile_j.
    # (tile_m, tile_n) can only be (b_m, b_n), (b_m, r_n), (r_m, b_n), or (r_m, r_n).

    b_m, b_n = source.layout.tile_m, source.layout.tile_n
    r_m = source.layout.m % b_m
    r_n = source.layout.n % b_n

    shapes = Tuple{Int,Int}[]
    push!(shapes, (b_m, b_n))
    r_n > 0 && push!(shapes, (b_m, r_n))
    r_m > 0 && push!(shapes, (r_m, b_n))
    (r_m > 0 && r_n > 0) && push!(shapes, (r_m, r_n))

    unique_shapes = unique(shapes)

    # Temporary buffers for wave filtering
    wave_active_idx = ws.next_active_idx # Reuse next_active_idx as scratch
    wave_slots = ws.survivor_pos # Reuse survivor_pos as scratch

    for (tm, tn) in unique_shapes
        fill!(ws.wave_count, zero(Int32))

        filter_wave_indices_kernel!(backend)(
            wave_active_idx, wave_slots, ws.wave_count, active_idx, source.layout, tm, tn, n_active;
            ndrange=n_active,
        )

        n_wave = Int(Array(ws.wave_count)[1])
        if n_wave == 0
            continue
        end

        wave_active_idx_view = @view wave_active_idx[1:n_wave]
        wave_slots_view = @view wave_slots[1:n_wave]

        # Build ptrs for this wave
        build_dense_A_ptrs_kernel!(backend)(
            ws.A_ptrs, pointer(source.A), stride(source.A, 1), stride(source.A, 2), source.layout, wave_active_idx_view, n_wave;
            ndrange=n_wave,
        )

        _setup_ptrs!(ws.Omega_ptrs, Omega_active, wave_slots_view, n_wave, backend; scattered=true)
        _setup_ptrs!(ws.Y_ptrs, Y_current, wave_slots_view, n_wave, backend; scattered=true)

        if backend isa CPU
            # Host access to filtered indices
            host_active_idx = Array(@view wave_active_idx[1:n_wave])
            host_slots = Array(@view wave_slots[1:n_wave])
            for i in 1:n_wave
                p = Int(host_slots[i])
                tile_linear = Int(host_active_idx[i])
                ti, tj = inverse_tile_index(source.layout, tile_linear)
                p0, q0 = tile_origin_coords(source.layout, ti, tj)
                Ap = @view source.A[p0:(p0+tm-1), q0:(q0+tn-1)]
                Op = @view Omega_active[1:tn, :, p]
                Yp = @view Y_current[1:tm, :, p]
                BLAS.gemm!('N', 'N', T(one(T)), Ap, Op, T(zero(T)), Yp)
            end
        else
            # On GPU, we need a reference tile for A
            # We can't easily get a view from pointers on host without knowing p0, q0.
            # But we only need Aref for its leading dimension and type.
            # Any tile with shape (tm, tn) will do.
            # For simplicity, we can use a dummy view if we know the stride.

            # Use the first tile in the wave to get p0, q0 for Aref
            first_tile = Int(Array(@view wave_active_idx[1:1])[1])
            ti1, tj1 = inverse_tile_index(source.layout, first_tile)
            p0_1, q0_1 = tile_origin_coords(source.layout, ti1, tj1)
            Aref = @view source.A[p0_1:(p0_1+tm-1), q0_1:(q0_1+tn-1)]

            gemm_batched_ptrs!(
                'N', 'N', one(T),
                ws.A_ptrs, Aref,
                ws.Omega_ptrs, @view(Omega_active[1:tn, :, 1]),
                zero(T),
                ws.Y_ptrs, @view(Y_current[1:tm, :, 1]),
                n_wave,
            )
        end
    end

    return Y_current
end

function _sample_corange!(
    V::AbstractArray{T,3},
    source::DirectDenseSamplingState,
    U::AbstractArray{T,3},
    ranks,
    backend,
) where {T}
    transchar = T <: Real ? 'T' : 'C'
    batch_size = size(V, 3)
    rank_cols = size(U, 2)

    b_m, b_n = source.layout.tile_m, source.layout.tile_n
    r_m = source.layout.m % b_m
    r_n = source.layout.n % b_n
    shapes = Tuple{Int,Int}[]
    push!(shapes, (b_m, b_n))
    r_n > 0 && push!(shapes, (b_m, r_n))
    r_m > 0 && push!(shapes, (r_m, b_n))
    (r_m > 0 && r_n > 0) && push!(shapes, (r_m, r_n))
    unique_shapes = unique(shapes)

    # Use a dummy workspace or allocate temporary ptrs if needed for GPU.
    # Since this is once per call, we can afford small overhead, but let's try to stay batched.
    # For now, implemented as shape-grouped waves.

    for (tm, tn) in unique_shapes
        active_tiles = Int[]
        for batch in 1:batch_size
            ti, tj = inverse_tile_index(source.layout, batch)
            if tile_sizes(source.layout, ti, tj) == (tm, tn)
                push!(active_tiles, batch)
            end
        end
        isempty(active_tiles) && continue

        n_wave = length(active_tiles)

        # We need pointer arrays. If we don't have ws, we might have to use views loop
        # or a temporary. Given the constraints, a view loop by shape is still better than scalar.
        if backend isa CPU
            for batch in active_tiles
                ti, tj = inverse_tile_index(source.layout, batch)
                p0, q0 = tile_origin_coords(source.layout, ti, tj)
                Ap = @view source.A[p0:(p0+tm-1), q0:(q0+tn-1)]
                Up = @view U[1:tm, 1:rank_cols, batch]
                Vp = @view V[1:tn, 1:rank_cols, batch]
                BLAS.gemm!(transchar, 'N', T(one(T)), Ap, Up, T(zero(T)), Vp)
            end
        else
            # For GPU without workspace, we fall back to a loop of gemm_batched! on views if possible,
            # or just a loop of gemm!.
            for batch in active_tiles
                ti, tj = inverse_tile_index(source.layout, batch)
                p0, q0 = tile_origin_coords(source.layout, ti, tj)
                Ap = @view source.A[p0:(p0+tm-1), q0:(q0+tn-1)]
                Up = @view U[1:tm, 1:rank_cols, batch]
                Vp = @view V[1:tn, 1:rank_cols, batch]
                gemm_batched!(transchar, 'N', one(T), reshape(Ap, tm, tn, 1), reshape(Up, tm, rank_cols, 1), zero(T), reshape(Vp, tn, rank_cols, 1))
            end
        end
    end

    return V
end

"""
    ara_batched!(U, V, ranks, A, layout, max_rank, block_size, eps)

Compress a dense matrix `A` into low-rank factors `U` and `V` using `layout`.
Sampling is done directly from `A` without packing tiles.
"""
function ara_batched!(
    U::AbstractArray{T,3},
    V::AbstractArray{T,3},
    ranks::AbstractVector{RankT},
    A::AbstractMatrix{T},
    layout::TileMap,
    max_rank::Int,
    block_size::Int,
    eps;
    required_samples::Int=10,
) where {T,RankT<:Integer}
    get_backend(U) == get_backend(V) == get_backend(A) ||
        throw(ArgumentError("U, V, and A must have the same backend"))

    source = DirectDenseSamplingState(A, layout)
    return _ara_batched_impl!(
        U,
        V,
        ranks,
        source,
        layout.tile_m,
        layout.tile_n,
        prod(size(layout)),
        max_rank,
        block_size,
        eps,
        nothing;
        required_samples,
        backend=get_backend(A),
    )
end
