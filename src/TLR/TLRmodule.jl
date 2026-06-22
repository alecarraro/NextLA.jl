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

# 1. types/
# operator.jl defines AbstractTLROperator and TLRLinearOperator
include("algorithms/operator.jl")
# order.jl defines TileOrder which is used in TileMap and TLRMatrix
include("layout/order.jl")
# layout.jl defines TileMap which is used in TLRMatrix and PackedTileStorage
include("layout/layout.jl")
# matrix.jl defines TLRMatrix
include("types/matrix.jl")

# 2. layout/ (continued)
# packing.jl defines pack! and depends on TileMap
include("layout/packing.jl")

# 3. algorithms/
include("algorithms/ara/core.jl")
include("algorithms/ara/batched.jl")
include("algorithms/ara/operator.jl")
include("algorithms/compress.jl")

# experimental/
include("experimental/rademacher_sampling.jl")
include("experimental/ara_batched.jl")
include("experimental/compress.jl")

end
