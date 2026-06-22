@kernel function materialize_tiles_kernel!(offdiag::AbstractArray{T,3},
                                           diag::AbstractArray{T,3},
                                           A::AbstractMatrix{T},
                                           layout::Layout,
                                           ::Val{ROWS_PER_WORKGROUP},
                                           nwork::Int,
                                           launch_workgroups::Int) where {T,Layout<:TileMap,ROWS_PER_WORKGROUP}
    thread_row = @index(Local, Linear)
    workgroup_id = @index(Group, Linear)

    for work in workgroup_id:launch_workgroups:nwork
        tile_col = ((work - 1) % layout.tile_n) + 1
        linear = ((work - 1) ÷ layout.tile_n) + 1

        # recover the tile's matrix coordinates and physical dimensions
        tile_i, tile_j, p0, q0, tile_m, tile_n = tile_geometry(layout, linear)

        # copy diagonal tiles
        if tile_i == tile_j
            for row_in_tile in thread_row:ROWS_PER_WORKGROUP:layout.tile_m
                value = zero(T)
                if row_in_tile <= tile_m && tile_col <= tile_n
                    @inbounds value = A[p0 + row_in_tile - 1, q0 + tile_col - 1]
                end
                @inbounds diag[row_in_tile, tile_col, tile_i] = value
            end
        else
            batch = offdiag_batch_index(layout, tile_i, tile_j)
            for row_in_tile in thread_row:ROWS_PER_WORKGROUP:layout.tile_m
                value = zero(T)
                if row_in_tile <= tile_m && tile_col <= tile_n
                    @inbounds value = A[p0 + row_in_tile - 1, q0 + tile_col - 1]
                end
                @inbounds offdiag[row_in_tile, tile_col, batch] = value
            end
        end
    end
end

@kernel function materialize_all_tiles_kernel!(tiles::AbstractArray{T,3},
                                               A::AbstractMatrix{T},
                                               layout::Layout,
                                               ::Val{ROWS_PER_WORKGROUP},
                                               nwork::Int,
                                               launch_workgroups::Int) where {T,Layout<:TileMap,ROWS_PER_WORKGROUP}
    thread_row = @index(Local, Linear)
    workgroup_id = @index(Group, Linear)

    for work in workgroup_id:launch_workgroups:nwork
        tile_col = ((work - 1) % layout.tile_n) + 1
        linear = ((work - 1) ÷ layout.tile_n) + 1
        _, _, p0, q0, tile_m, tile_n = tile_geometry(layout, linear)

        for row_in_tile in thread_row:ROWS_PER_WORKGROUP:layout.tile_m
            value = zero(T)
            if row_in_tile <= tile_m && tile_col <= tile_n
                @inbounds value = A[p0 + row_in_tile - 1, q0 + tile_col - 1]
            end
            @inbounds tiles[row_in_tile, tile_col, linear] = value
        end
    end
end

"""
    pack!(storage, A; ROWS_PER_WORKGROUP=256, NWORKGROUPS=256, backend=get_backend(A))

Transform the dense matrix `A` into the packed tile storage described by
`storage.layout`, writing off-diagonal tiles into `storage.offdiag` and
diagonal tiles into `storage.diag`. Boundary tiles are zero padded out to the
uniform tile shape.

The GPU path launches a fixed number of workgroups and strides them over the
flat work queue of `(tile, in-tile column)` tasks, so we keep enough work in
flight to hide memory latency without launching one workgroup per tile column.
"""
function pack!(storage::PackedTileStorage{Offdiag,Diag,<:TileMap},
               A::AbstractMatrix{T};
               ROWS_PER_WORKGROUP::Int=256,
               NWORKGROUPS::Int=256,
               backend=KernelAbstractions.get_backend(A)) where {T,Offdiag<:AbstractArray{T,3},Diag<:AbstractArray{T,3}}
    ROWS_PER_WORKGROUP > 0 || throw(ArgumentError("ROWS_PER_WORKGROUP must be positive"))
    NWORKGROUPS > 0 || throw(ArgumentError("NWORKGROUPS must be positive"))
    size(A) == (storage.layout.m, storage.layout.n) ||
        throw(DimensionMismatch("dense matrix size must match the layout"))

    nwork = storage.layout.tile_n * prod(size(storage.layout))
    launch_workgroups = min(NWORKGROUPS, max(1, nwork))
    kernel! = materialize_tiles_kernel!(backend, ROWS_PER_WORKGROUP)
    kernel!(
        storage.offdiag,
        storage.diag,
        A,
        storage.layout,
        Val(ROWS_PER_WORKGROUP),
        nwork,
        launch_workgroups;
        ndrange=ROWS_PER_WORKGROUP * launch_workgroups,
    )
    return storage
end

function pack!(tiles::AbstractArray{T,3},
               A::AbstractMatrix{T},
               layout::TileMap;
               ROWS_PER_WORKGROUP::Int=256,
               NWORKGROUPS::Int=256,
               backend=KernelAbstractions.get_backend(A)) where {T}
    ROWS_PER_WORKGROUP > 0 || throw(ArgumentError("ROWS_PER_WORKGROUP must be positive"))
    NWORKGROUPS > 0 || throw(ArgumentError("NWORKGROUPS must be positive"))
    size(A) == (layout.m, layout.n) ||
        throw(DimensionMismatch("dense matrix size must match the layout"))
    size(tiles) == (layout.tile_m, layout.tile_n, prod(size(layout))) ||
        throw(DimensionMismatch("tile batch must have size ($(layout.tile_m), $(layout.tile_n), $(prod(size(layout))))"))

    nwork = layout.tile_n * prod(size(layout))
    launch_workgroups = min(NWORKGROUPS, max(1, nwork))
    kernel! = materialize_all_tiles_kernel!(backend, ROWS_PER_WORKGROUP)
    kernel!(
        tiles,
        A,
        layout,
        Val(ROWS_PER_WORKGROUP),
        nwork,
        launch_workgroups;
        ndrange=ROWS_PER_WORKGROUP * launch_workgroups,
    )
    return tiles
end
