"""
    TLRMatrix

Tile Low-Rank (TLR) matrix representation.

A `TLRMatrix` stores a dense matrix in a tile low-rank format. The matrix is
partitioned into square tiles of size `b × b` and each tile is represented
either as a low-rank factorization or as a dense block.

Given a tiled matrix

```math
A =
\\begin{bmatrix}
A_{11} & A_{12} & \\cdots & A_{1n_t} \\\\
A_{21} & A_{22} & \\cdots & A_{2n_t} \\\\
\\vdots & \\vdots & \\ddots & \\vdots \\\\
A_{m_t1} & A_{m_t2} & \\cdots & A_{m_tn_t}
\\end{bmatrix},
```

each off-diagonal tile is approximated by a rank-`r` factorization

```math
A_{ij} \\approx U_{ij} V_{ij}^{\\mathsf T},
\\qquad i \\neq j,
```

where `U_{ij} ∈ ℝ^{b×r}` and `V_{ij} ∈ ℝ^{b×r}`. Diagonal tiles may be stored
densely or compressed depending on the value of `compress_diag`.

# Storage layout

The low-rank factors are stored in contiguous three-dimensional arrays

```julia
U :: AbstractArray{T,3}  # (b, maxrank, ntiles)
V :: AbstractArray{T,3}  # (b, maxrank, ntiles)
```

where the third dimension indexes the compact sequence of stored low-rank
tiles. When `compress_diag=false`, only off-diagonal tiles are stored in
`U`, `V`, and `ranks`; the diagonal tiles live in `diag`.

Dense diagonal tiles are stored in

```julia
diag :: AbstractArray{T,3}
```

with shape `(b, b, ndiag)`.

# Fields

- `backend`: execution backend.
- `U`, `V`: storage for low-rank tile factors.
- `diag`: storage for dense diagonal tiles.
- `layout`: tile metadata and indexing map.
- `ranks`: numerical rank of each stored tile.
- `m`, `n`: global matrix dimensions.
- `b`: tile size.
- `mt`, `nt`: number of tile rows and columns.
- `ndiag`: number of diagonal tiles.
- `maxrank`: maximum representable tile rank.
- `compress_diag`: whether diagonal tiles are also stored in compressed form.

# Notes

The storage allocated for each tile corresponds to `maxrank`, while the
effective rank is given by the corresponding entry in `ranks`. This allows
tiles with different ranks to be stored in a single contiguous allocation.
"""
struct TLRMatrix{
    BackendT<:Backend,
    T,
    RankT<:Integer,
    UStore<:AbstractArray{T,3},
    VStore<:AbstractArray{T,3},
    DiagStore<:AbstractArray{T,3},
    RanksStore<:AbstractVector{RankT},
    L<:TileMap,
}
    backend::BackendT
    U::UStore
    V::VStore
    diag::DiagStore
    layout::L
    ranks::RanksStore
    m::Int
    n::Int
    ndiag::Int
    b::Int
    mt::Int
    nt::Int
    maxrank::Int
    compress_diag::Bool
end

Base.eltype(::Type{<:TLRMatrix{<:Any,T}}) where {T} = T
Base.eltype(A::TLRMatrix{<:Any,T}) where {T} = T
Base.getproperty(A::TLRMatrix, s::Symbol) = s === :order ? getfield(getfield(A, :layout), :order) : getfield(A, s)
"""Return the dense matrix dimensions `(m, n)` represented by `A`."""
Base.size(A::TLRMatrix) = (A.m, A.n)
Base.size(A::TLRMatrix, d::Int) = size(A)[d]

"""
    tile_linear_index(A, i, j)

Return the logical tile-grid index of tile `(i, j)` using `A.layout.order`.
"""
tile_linear_index(A::TLRMatrix, i::Integer, j::Integer) = tile_linear_index(A.layout.order, i, j)

"""
    tile_rank_index(A, i, j)

Return the index in `A.ranks` associated with tile `(i, j)`.
"""
@inline function tile_rank_index(A::TLRMatrix, i::Integer, j::Integer)
    return A.compress_diag ? tile_linear_index(A, i, j) : offdiag_batch_index(A.layout, Int(i), Int(j))
end

"""
    n_tiles(A)

Return the number of stored low-rank tiles in `A`.
"""
@inline n_tiles(A::TLRMatrix) = length(A.ranks)

"""
    TLRMatrix(backend, T, m, n, b, maxrank; kwargs...)
    TLRMatrix(A, b, maxrank; kwargs...)

Create a Tile Low-Rank matrix of size `m × n` with tile size `b` and maximum
tile rank `maxrank`.

# Keywords

- `compress_diag=false`: compress diagonal tiles.
- `rank_type=Int32`: integer type used to store tile ranks.
- `tile_order=TileColMajor`: tile indexing order.

# Notes

This constructor allocates storage but does not populate it from a dense input
matrix. The factor arrays are left uninitialized, while `ranks` is initialized
to zero.
"""
function TLRMatrix(
    backend::Backend,
    ::Type{T},
    m::Int,
    n::Int,
    b::Int,
    maxrank::Int;
    compress_diag::Bool=false,
    rank_type::Type{<:Integer}=Int32,
    tile_order::Type{<:TileOrder}=TileColMajor,
) where {T}

    m > 0 || throw(ArgumentError("m must be positive"))
    n > 0 || throw(ArgumentError("n must be positive"))
    b > 0 || throw(ArgumentError("b must be positive"))
    maxrank >= 0 || throw(ArgumentError("maxrank must be nonnegative"))

    mt = cld(m, b)
    nt = cld(n, b)
    layout = TileMap(tile_order(mt, nt), b, b, m, n)
    ntiles = compress_diag ? prod(size(layout)) : noffdiag_tiles(layout)

    U = allocate(backend, T, b, maxrank, ntiles)
    V = allocate(backend, T, b, maxrank, ntiles)
    diag = allocate(backend, T, b, b, compress_diag ? 0 : ndiag_tiles(layout))
    ranks = allocate(backend, rank_type, ntiles)

    fill!(ranks, zero(rank_type))

    return TLRMatrix(
        backend,
        U,
        V,
        diag,
        layout,
        ranks,
        m,
        n,
        size(diag, 3),
        b,
        mt,
        nt,
        maxrank,
        compress_diag,
    )
end

"""
    TLRMatrix(A, blocksize, maxrank; kwargs...)

Create an empty `TLRMatrix` on the same backend and with the same element type
as the dense matrix `A`.
"""
function TLRMatrix(
    A::AbstractMatrix{T},
    blocksize::Int,
    maxrank::Int;
    compress_diag::Bool=false,
    rank_type::Type{<:Integer}=Int32,
    tile_order::Type{<:TileOrder}=TileColMajor,
) where {T}
    return TLRMatrix(
        get_backend(A),
        T,
        size(A, 1),
        size(A, 2),
        blocksize,
        maxrank;
        compress_diag=compress_diag,
        rank_type=rank_type,
        tile_order=tile_order,
    )
end
