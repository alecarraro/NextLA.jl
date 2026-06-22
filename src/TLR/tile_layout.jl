"""
    TileMap(order, tile_m, tile_n, m, n)

Lightweight metadata object describing how a dense `m × n` matrix is viewed as
a uniformly padded tile grid with logical tile size `tile_m × tile_n`.

This type does not allocate storage. Its role is to centralize tile indexing,
boundary extents, and compact off-diagonal numbering on top of an existing
[`TileOrder`](@ref).
"""
struct TileMap{O<:TileOrder}
    order::O
    tile_m::Int
    tile_n::Int
    m::Int
    n::Int

    function TileMap(order::O,
                     tile_m::Integer,
                     tile_n::Integer,
                     m::Integer,
                     n::Integer) where {O<:TileOrder}
        tile_m > 0 || throw(ArgumentError("tile_m must be positive"))
        tile_n > 0 || throw(ArgumentError("tile_n must be positive"))
        m > 0 || throw(ArgumentError("m must be positive"))
        n > 0 || throw(ArgumentError("n must be positive"))

        layout = new{O}(order, Int(tile_m), Int(tile_n), Int(m), Int(n))
        size(order) == expected_tile_grid(layout) ||
            throw(DimensionMismatch("tile_order size must match the dense matrix tile grid"))
        return layout
    end
end

"""
    PackedTileStorage(offdiag, diag, layout)

Bundle the arrays that hold a packed tiled representation of a dense matrix
together with the [`TileMap`](@ref) that defines their indexing.

- `offdiag` stores the compact sequence of off-diagonal tiles.
- `diag` stores the diagonal tiles.
- `layout` defines how dense tiles map into those arrays.
"""
struct PackedTileStorage{Offdiag<:AbstractArray,Diag<:AbstractArray,L<:TileMap}
    offdiag::Offdiag
    diag::Diag
    layout::L

    function PackedTileStorage(offdiag::Offdiag,
                               diag::Diag,
                               layout::L) where {Offdiag<:AbstractArray,Diag<:AbstractArray,L<:TileMap}
        size(offdiag, 1) == layout.tile_m ||
            throw(DimensionMismatch("packed off-diagonal tiles must have first dimension $(layout.tile_m)"))
        size(offdiag, 2) == layout.tile_n ||
            throw(DimensionMismatch("packed off-diagonal tiles must have second dimension $(layout.tile_n)"))
        size(diag, 1) == layout.tile_m ||
            throw(DimensionMismatch("packed diagonal tiles must have first dimension $(layout.tile_m)"))
        size(diag, 2) == layout.tile_n ||
            throw(DimensionMismatch("packed diagonal tiles must have second dimension $(layout.tile_n)"))
        size(offdiag, 3) == noffdiag_tiles(layout) ||
            throw(DimensionMismatch("packed off-diagonal batch size must be $(noffdiag_tiles(layout))"))
        size(diag, 3) == ndiag_tiles(layout) ||
            throw(DimensionMismatch("packed diagonal batch size must be $(ndiag_tiles(layout))"))
        return new{Offdiag,Diag,L}(offdiag, diag, layout)
    end
end

Base.size(layout::TileMap) = size(layout.order)

"""
    expected_tile_grid(layout)

Return the tile-grid dimensions implied by the dense matrix shape and uniform
tile size stored in `layout`.
"""
@inline expected_tile_grid(layout::TileMap) = (cld(layout.m, layout.tile_m), cld(layout.n, layout.tile_n))

"""Return the number of diagonal tiles in `layout`."""
@inline ndiag_tiles(layout::TileMap) = min(size(layout)...)

"""Return the number of off-diagonal tiles stored in compact packed form."""
@inline noffdiag_tiles(layout::TileMap) = prod(size(layout)) - ndiag_tiles(layout)

"""
    tile_geometry(layout, linear)

Return `(tile_i, tile_j, p0, q0, tile_m, tile_n)` for the tile with linear
index `linear`, where `(tile_i, tile_j)` are logical tile coordinates, `(p0,
q0)` is the dense starting position, and `tile_m × tile_n` is the valid
unpadded extent of that boundary tile.
"""
@inline function tile_geometry(layout::TileMap, linear::Int)
    tile_i, tile_j = inverse_tile_index(layout.order, linear)
    p0 = (tile_i - 1) * layout.tile_m + 1
    q0 = (tile_j - 1) * layout.tile_n + 1
    tile_m = min(layout.tile_m, layout.m - p0 + 1)
    tile_n = min(layout.tile_n, layout.n - q0 + 1)
    return tile_i, tile_j, p0, q0, tile_m, tile_n
end

"""
    offdiag_batch_index(layout, tile_i, tile_j)

Return the compact off-diagonal batch index corresponding to logical tile
coordinate `(tile_i, tile_j)`.
"""
@inline function offdiag_batch_index(layout::TileMap, tile_i::Int, tile_j::Int)
    tile_i == tile_j && throw(ArgumentError("off-diagonal index is undefined for diagonal tiles"))
    linear = tile_linear_index(layout.order, tile_i, tile_j)
    ndiag = ndiag_tiles(layout)
    diag_prefix = if layout.order isa TileOrder{ColMajor}
        min(ndiag, (linear + layout.order.mt) ÷ (layout.order.mt + 1))
    else
        min(ndiag, (linear + layout.order.nt) ÷ (layout.order.nt + 1))
    end
    return linear - diag_prefix
end

"""
    offdiag_linear_index(layout, batch)

Map a compact off-diagonal batch index back to the corresponding full linear
tile index in the traversal induced by `layout.order`.
"""
@inline function offdiag_linear_index(layout::TileMap{TileOrder{ColMajor}}, batch::Int)
    return batch + min(ndiag_tiles(layout), cld(batch, layout.order.mt))
end

@inline function offdiag_linear_index(layout::TileMap{TileOrder{RowMajor}}, batch::Int)
    return batch + min(ndiag_tiles(layout), cld(batch, layout.order.nt))
end
