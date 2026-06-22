struct OperatorARASource{T,Op,L<:TileMap} <: AbstractARASource{T}
    op::Op
    layout::L
    compress_diag::Bool
end

@inline _batch_linear_index(layout::TileMap, batch::Int, compress_diag::Bool) =
    compress_diag ? batch : offdiag_linear_index(layout, batch)

@inline function _batch_tile_coords(layout::TileMap, batch::Int, compress_diag::Bool)
    return inverse_tile_index(layout.order, _batch_linear_index(layout, batch, compress_diag))
end

@inline function _operator_batch_seed(seed::Integer, batch::Int)
    mixed = UInt64(seed) ⊻ (UInt64(batch) * 0x9e3779b97f4a7c15)
    return Int(mod(mixed, UInt64(typemax(Int))))
end

function _materialize_diag_tiles!(diag::AbstractArray{T,3},
                                  op,
                                  layout::TileMap;
                                  backend) where {T}
    size(diag, 1) == layout.tile_m ||
        throw(DimensionMismatch("diag tiles must have first dimension $(layout.tile_m)"))
    size(diag, 2) == layout.tile_n ||
        throw(DimensionMismatch("diag tiles must have second dimension $(layout.tile_n)"))
    size(diag, 3) == ndiag_tiles(layout) ||
        throw(DimensionMismatch("diag tiles must have third dimension $(ndiag_tiles(layout))"))

    fill!(diag, zero(T))

    X = allocate(backend, T, layout.n, layout.tile_n)
    Y = allocate(backend, T, layout.m, layout.tile_n)
    X_host = zeros(T, layout.n, layout.tile_n)

    for tile_d in 1:ndiag_tiles(layout)
        p0 = (tile_d - 1) * layout.tile_m + 1
        q0 = (tile_d - 1) * layout.tile_n + 1
        tile_m = min(layout.tile_m, layout.m - p0 + 1)
        tile_n = min(layout.tile_n, layout.n - q0 + 1)

        fill!(X_host, zero(T))
        @inbounds for local_col in 1:tile_n
            X_host[q0 + local_col - 1, local_col] = one(T)
        end

        copyto!(X, X_host)
        mul!(Y, op, X)
        copyto!(@view(diag[1:tile_m, 1:tile_n, tile_d]), @view(Y[p0:(p0 + tile_m - 1), 1:tile_n]))
    end

    return diag
end

# Group tiles by the single tile support touched in one global operator apply.
# The groups are emitted in first-appearance order along the active batch
# traversal, so row-major vs col-major layouts keep their configured tile order.
function _collect_range_groups(source::OperatorARASource, active_batches::AbstractVector{<:Integer})
    layout = source.layout
    nkeys = size(layout)[2]
    seen = falses(nkeys)
    order = Int[]
    groups = [Tuple{Int,Int,Int}[] for _ in 1:nkeys]

    for p in eachindex(active_batches)
        batch = Int(active_batches[p])
        tile_i, tile_j = _batch_tile_coords(layout, batch, source.compress_diag)
        seen[tile_j] || (push!(order, tile_j); seen[tile_j] = true)
        push!(groups[tile_j], (Int(p), batch, tile_i))
    end

    return groups, order
end

function _collect_corange_groups(source::OperatorARASource, batch_size::Int)
    layout = source.layout
    nkeys = size(layout)[1]
    seen = falses(nkeys)
    order = Int[]
    groups = [Tuple{Int,Int}[] for _ in 1:nkeys]

    for batch in 1:batch_size
        tile_i, tile_j = _batch_tile_coords(layout, batch, source.compress_diag)
        seen[tile_i] || (push!(order, tile_i); seen[tile_i] = true)
        push!(groups[tile_i], (batch, tile_j))
    end

    return groups, order
end

function _sample_range!(Y::AbstractArray{T,3},
                        source::OperatorARASource{T},
                        ws::ARAWorkspace,
                        active_idx,
                        n_active::Int,
                        sample_size::Int,
                        dense::Bool,
                        seed::Integer,
                        backend) where {T}
    fill!(Y, zero(T))

    active_host = Array(@view(active_idx[1:n_active]))
    groups, order = _collect_range_groups(source, active_host)
    max_group = isempty(order) ? 0 : maximum(length(groups[key]) for key in order)
    max_group == 0 && return Y

    layout = source.layout
    width_max = sample_size * max_group
    X = allocate(backend, T, layout.n, width_max)
    Y_global = allocate(backend, T, layout.m, width_max)
    X_host = zeros(T, layout.n, width_max)

    for tile_j in order
        group = groups[tile_j]
        width = sample_size * length(group)
        fill!(@view(X_host[:, 1:width]), zero(T))

        q0 = (tile_j - 1) * layout.tile_n + 1
        tile_n = min(layout.tile_n, layout.n - q0 + 1)

        for slot in eachindex(group)
            _, batch, _ = group[slot]
            seg_start = (slot - 1) * sample_size + 1
            Omega = randn(MersenneTwister(_operator_batch_seed(seed, batch)), T, tile_n, sample_size)
            copyto!(@view(X_host[q0:(q0 + tile_n - 1), seg_start:(seg_start + sample_size - 1)]), Omega)
        end

        copyto!(@view(X[:, 1:width]), @view(X_host[:, 1:width]))
        mul!(@view(Y_global[:, 1:width]), source.op, @view(X[:, 1:width]))

        for slot in eachindex(group)
            p, _, tile_i = group[slot]
            seg = ((slot - 1) * sample_size + 1):(slot * sample_size)
            p0 = (tile_i - 1) * layout.tile_m + 1
            tile_m = min(layout.tile_m, layout.m - p0 + 1)
            copyto!(@view(Y[1:tile_m, 1:sample_size, p]), @view(Y_global[p0:(p0 + tile_m - 1), seg]))
        end
    end

    return Y
end

function _sample_corange!(V::AbstractArray{T,3},
                          source::OperatorARASource{T},
                          U::AbstractArray{T,3},
                          backend) where {T}
    fill!(V, zero(T))

    groups, order = _collect_corange_groups(source, size(U, 3))
    max_group = isempty(order) ? 0 : maximum(length(groups[key]) for key in order)
    max_group == 0 && return V

    layout = source.layout
    sample_size = size(U, 2)
    width_max = sample_size * max_group
    X = allocate(backend, T, layout.m, width_max)
    Z_global = allocate(backend, T, layout.n, width_max)
    X_host = zeros(T, layout.m, width_max)

    for tile_i in order
        group = groups[tile_i]
        width = sample_size * length(group)
        fill!(@view(X_host[:, 1:width]), zero(T))

        p0 = (tile_i - 1) * layout.tile_m + 1
        tile_m = min(layout.tile_m, layout.m - p0 + 1)

        for slot in eachindex(group)
            batch, _ = group[slot]
            seg = ((slot - 1) * sample_size + 1):(slot * sample_size)
            copyto!(@view(X_host[p0:(p0 + tile_m - 1), seg]), Array(@view(U[1:tile_m, 1:sample_size, batch])))
        end

        copyto!(@view(X[:, 1:width]), @view(X_host[:, 1:width]))
        mul!(@view(Z_global[:, 1:width]), adjoint(source.op), @view(X[:, 1:width]))

        for slot in eachindex(group)
            batch, tile_j = group[slot]
            seg = ((slot - 1) * sample_size + 1):(slot * sample_size)
            q0 = (tile_j - 1) * layout.tile_n + 1
            tile_n = min(layout.tile_n, layout.n - q0 + 1)
            copyto!(@view(V[1:tile_n, 1:sample_size, batch]), @view(Z_global[q0:(q0 + tile_n - 1), seg]))
        end
    end

    return V
end

function ara_batched_operator!(
    U::AbstractArray{T,3},
    V::AbstractArray{T,3},
    ranks::AbstractVector{RankT},
    op,
    layout::TileMap,
    max_rank::Int,
    block_size::Int,
    eps;
    seed::Integer=0x123456789abcdef0,
    compress_diag::Bool=false,
    backend=get_backend(U),
) where {T,RankT<:Integer}
    applicable(adjoint, op) || throw(ArgumentError("operator must support adjoint(op)"))

    source = OperatorARASource{T,typeof(op),typeof(layout)}(op, layout, compress_diag)
    batch_size = compress_diag ? prod(size(layout)) : noffdiag_tiles(layout)

    return _ara_batched_impl!(
        U,
        V,
        ranks,
        source,
        layout.tile_m,
        layout.tile_n,
        batch_size,
        max_rank,
        block_size,
        eps,
        nothing;
        seed,
        backend,
    )
end
