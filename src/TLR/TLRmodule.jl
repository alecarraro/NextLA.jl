module TLRmodule

using LinearAlgebra
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

using ..NextLA: gemm_batched!, gemm_batched_ptrs!, syrk_batched!, trsm_batched!, potrf_batched!, supports_pointer_batched

export TileOrderStyle, TileOrder, ColMajor, RowMajor, TileColMajor, TileRowMajor
export tile_stride, tile_coords, inverse_tile_coords
export AbstractTLROperator, TLRLinearOperator
export TLRMatrix, tile_linear_index, tile_rank_index, inverse_tile_index
export TileMap, PackedTileStorage
export ndiag_tiles, noffdiag_tiles, tile_geometry, pack!
export ara_batched!, compress!

include("tile_order.jl")
include("tile_layout.jl")
include("operator.jl")
include("tlr_matrix.jl")
include("pack_tiles.jl")
include("ara_core.jl")
include("ara_batched.jl")
include("ara_operator.jl")
include("experimental/rademacher_sampling.jl")
include("compress.jl")
include("experimental/ara_batched.jl")
include("experimental/compress.jl")
end
