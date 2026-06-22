using LinearAlgebra

abstract type TileOrderStyle end

struct ColMajor <: TileOrderStyle end
struct RowMajor <: TileOrderStyle end

"""
    TileOrder{S<:TileOrderStyle}(mt, nt)

Defines a mapping between tile coordinates `(i, j)` and linear tile indices for
an `mt × nt` tile grid.

The style parameter `S` determines the tile traversal order:

- `ColMajor`: column-major tile ordering.
- `RowMajor`: row-major tile ordering.

Using a single `TileOrder{S}` family keeps dispatch anchored to one concrete
type identity across modules, which avoids ambiguity from redundant abstract
policy types.
"""
struct TileOrder{S<:TileOrderStyle}
    mt::Int
    nt::Int

    function TileOrder{S}(mt::Integer, nt::Integer) where {S<:TileOrderStyle}
        mt > 0 || throw(ArgumentError("mt must be positive"))
        nt > 0 || throw(ArgumentError("nt must be positive"))
        new{S}(Int(mt), Int(nt))
    end
end

const TileColMajor = TileOrder{ColMajor}
const TileRowMajor = TileOrder{RowMajor}

"""
    size(order)

Return the logical tile-grid dimensions `(mt, nt)` associated with `order`.
"""
Base.size(order::TileOrder) = (order.mt, order.nt)

"""Return the contiguous stride used by the linear tile index mapping."""
@inline tile_stride(order::TileOrder{ColMajor}) = order.mt
@inline tile_stride(order::TileOrder{RowMajor}) = order.nt

"""
    tile_coords(order, i, j)

Map logical tile coordinates `(i, j)` into the reordered coordinate pair used
internally by `order` before flattening to a linear index.
"""
@inline tile_coords(::TileOrder{ColMajor}, i::Int, j::Int) = (i, j)
@inline tile_coords(::TileOrder{RowMajor}, i::Int, j::Int) = (j, i)

"""
    inverse_tile_coords(order, a, b)

Undo [`tile_coords`](@ref), mapping reordered coordinates back to logical tile
coordinates `(i, j)`.
"""
@inline inverse_tile_coords(::TileOrder{ColMajor}, a::Int, b::Int) = (a, b)
@inline inverse_tile_coords(::TileOrder{RowMajor}, a::Int, b::Int) = (b, a)

"""
    checkbounds_tile(order, i, j)

Throw a `BoundsError` unless `(i, j)` is a valid tile coordinate in `order`.
"""
@inline function checkbounds_tile(order::TileOrder, i::Integer, j::Integer)
    1 <= i <= order.mt || throw(BoundsError(order, (i, :)))
    1 <= j <= order.nt || throw(BoundsError(order, (:, j)))
    return nothing
end

"""
    tile_linear_index(order, i, j)

Return the linear index corresponding to tile coordinate `(i, j)`.
"""
@inline function tile_linear_index(order::TileOrder, i::Integer, j::Integer)
    checkbounds_tile(order, i, j)
    a, b = tile_coords(order, Int(i), Int(j))
    return a + (b - 1) * tile_stride(order)
end

@inline (order::TileOrder)(i::Integer, j::Integer) = tile_linear_index(order, i, j)

"""
    inverse_tile_index(order, linear)

Return the tile coordinate `(i, j)` corresponding to a linear tile index.
"""
@inline function inverse_tile_index(order::TileOrder, linear::Integer)
    ntiles = order.mt * order.nt
    1 <= linear <= ntiles || throw(BoundsError(order, linear))

    stride = tile_stride(order)
    b = ((Int(linear) - 1) ÷ stride) + 1
    a = Int(linear) - (b - 1) * stride
    return inverse_tile_coords(order, a, b)
end

function Base.show(io::IO, order::TileOrder{ColMajor})
    print(io, "TileColMajor(", order.mt, ", ", order.nt, ")")
end

function Base.show(io::IO, order::TileOrder{RowMajor})
    print(io, "TileRowMajor(", order.mt, ", ", order.nt, ")")
end
