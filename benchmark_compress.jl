# NextLA TLR compression benchmark — CPU and GPU
#
# Run (CPU only):
#   julia --project=. benchmark_compress.jl
#
# Run (CPU + GPU):
#   julia --project=../gpuenv benchmark_compress.jl
#
# The script detects CUDA availability automatically.

using LinearAlgebra, Printf, Random, Statistics

# ── load CUDA if available ────────────────────────────────────────────────────
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

using NextLA
using NextLA: TLRMatrix, compress!, ara_compress!, fb_ara_compress!, tile_u, tile_v, dense_diag, dense_diag_corner, ranks,
              ndiag_tiles, noffdiag_tiles, tile_origin_coords, tile_size, alloc_workspace

# Skewed per-tile rank: most tiles are very low rank, a few approach the b/2
# ceiling (a tile of size b can carry at most b/2 rank before dense is cheaper).
# rand^4 concentrates mass near 1 with a heavy tail — high variance across tiles.
function _skewed_rank(rng, b::Int)
    return clamp(1 + floor(Int, (b ÷ 2) * rand(rng)^4), 1, b ÷ 2)
end

# ── generate a matrix whose off-diagonal tiles have low rank ─────────────────
# `dist=:uniform` draws rank in `[r_lo, r_hi]`; `dist=:skewed` draws the
# high-variance distribution above (ranks in `[1, b/2]`, mostly small).
function generate_tiled_lowrank(n::Int, b::Int;
                                 r_lo::Int=10, r_hi::Int=20, dist::Symbol=:uniform,
                                 T::Type=Float32, seed::Int=42)
    rng        = MersenneTwister(seed)
    A          = zeros(T, n, n)
    mt         = cld(n, b)
    true_ranks = zeros(Int, mt, mt)

    for j in 1:mt, i in 1:mt
        p0 = (i-1)*b + 1;  q0 = (j-1)*b + 1
        tm = min(b, n-p0+1); tn = min(b, n-q0+1)
        if i == j
            A[p0:p0+tm-1, q0:q0+tn-1] .= randn(rng, T, tm, tn)
        else
            k = dist === :skewed ? _skewed_rank(rng, b) : rand(rng, r_lo:r_hi)
            mul!(view(A, p0:p0+tm-1, q0:q0+tn-1),
                 randn(rng, T, tm, k),
                 randn(rng, T, tn, k)')
            true_ranks[i, j] = k
        end
    end
    A, true_ranks
end

# ── tile-by-tile relative Frobenius error ─────────────────────────────────────
function rel_error(A_cpu::AbstractMatrix, A_tlr::TLRMatrix)
    D  = Array(dense_diag(A_tlr))
    D_corner = Array(dense_diag_corner(A_tlr))
    err_sq = norm_sq = 0.0
    for ob in 1:noffdiag_tiles(A_tlr)
        lin = NextLA.TLRmodule._linear_from_offdiag(A_tlr, ob)
        i, j = NextLA.TLRmodule._inverse_tile_index(A_tlr, lin)
        p0, q0 = tile_origin_coords(A_tlr, i, j)
        tm, tn = tile_size(A_tlr, i, j)
        tile  = A_cpu[p0:p0+tm-1, q0:q0+tn-1]
        U_ob  = Array(tile_u(A_tlr, ob))
        V_ob  = Array(tile_v(A_tlr, ob))
        recon = U_ob * V_ob'
        err_sq  += sum(abs2, tile - recon)
        norm_sq += sum(abs2, tile)
    end
    for k in 1:ndiag_tiles(A_tlr)
        p0, q0 = tile_origin_coords(A_tlr, k, k)
        tm, tn = tile_size(A_tlr, k, k)
        tile = A_cpu[p0:p0+tm-1, q0:q0+tn-1]
        diag_tile = k <= size(D, 3) ? @view(D[1:tm, 1:tn, k]) : @view(D_corner[1:tm, 1:tn, 1])
        err_sq  += sum(abs2, tile - diag_tile)
        norm_sq += sum(abs2, tile)
    end
    sqrt(err_sq / norm_sq)
end

# ── synchronise helper (no-op on CPU, CUDA.synchronize() on GPU) ─────────────
gpu_sync() = HAS_CUDA ? CUDA.synchronize() : nothing

# Reclaim device memory between algorithms (the four variants + warmups otherwise
# fragment/exhaust small GPUs).
function gpu_reclaim()
    GC.gc(true)
    HAS_CUDA && CUDA.reclaim()
    nothing
end

# ── run one benchmark case on a given device ──────────────────────────────────
function run_case(A_cpu, true_rk, device_label, B, MAXRANK, TOL;
                  to_device = identity)
    n = size(A_cpu, 1)
    @printf("  [%s]  n=%d  b=%d  maxrank=%d  tol=%.1f\n",
            device_label, n, B, MAXRANK, TOL)

    A = to_device(A_cpu)
    gpu_sync()

    # Geometry map (host-side) so recovery stats don't need a container kept alive.
    Tg   = TLRMatrix(A, B, MAXRANK)
    noff = noffdiag_tiles(Tg)
    ob2ij = map(1:noff) do ob
        lin = NextLA.TLRmodule._linear_from_offdiag(Tg, ob)
        NextLA.TLRmodule._inverse_tile_index(Tg, lin)
    end
    budget = noff * MAXRANK
    Tg = nothing; gpu_reclaim()

    recovery(rk) = count(ob -> rk[ob] == true_rk[ob2ij[ob]...], 1:noff)

    # Each algorithm runs in its own scope, extracts host-side results, prints,
    # and drops its device buffers before the next — small GPUs can't hold all
    # four containers + workspaces at once.

    # ── Algorithm 1: cholqr2, fixed rank ─────────────────────────────────────
    t1 = let
        T1 = TLRMatrix(A, B, MAXRANK); ws1 = alloc_workspace(T1); gpu_sync()
        compress!(T1, A, ws1); gpu_sync()                                 # warmup
        t = @elapsed begin; compress!(T1, A, ws1); gpu_sync(); end
        e = rel_error(A_cpu, T1); ok = !any(isnan, Array(T1.int_U)); ue = ortho_errors(T1)
        @printf("    Alg1 cholqr2   %8.4f s  rel_err=%.2e  ok=%-5s  rank=%d (fixed)\n", t, e, ok, MAXRANK)
        print_ortho_vec("Alg1", ue)
        T1 = nothing; ws1 = nothing; gpu_reclaim(); t
    end

    # ── Algorithm 2: cholqr + NS + adaptive truncation (old compress!) ───────
    t2 = let
        T2 = TLRMatrix(A, B, MAXRANK); ws2 = alloc_workspace(T2); gpu_sync()
        compress!(T2, A, ws2; tol=TOL); gpu_sync()                        # warmup
        t = @elapsed begin; compress!(T2, A, ws2; tol=TOL); gpu_sync(); end
        e = rel_error(A_cpu, T2); rk = Array(ranks(T2)); ok = !any(isnan, Array(T2.int_U)); ue = ortho_errors(T2)
        @printf("    Alg2 cholqr+NS %8.4f s  rel_err=%.2e  ok=%-5s  ranks∈[%d,%d]  recovery=%d/%d\n",
                t, e, ok, minimum(rk), maximum(rk), recovery(rk), noff)
        print_ortho_vec("Alg2", ue)
        T2 = nothing; ws2 = nothing; gpu_reclaim(); t
    end

    # ── Algorithm 3: ARA (adaptive, dynamic-compaction baseline) ─────────────
    t3 = let
        T3 = TLRMatrix(A, B, MAXRANK); gpu_sync()
        ara_compress!(T3, A; tol=ARA_TOL, bs=ARA_BS); gpu_sync()          # warmup
        t = @elapsed begin; ara_compress!(T3, A; tol=ARA_TOL, bs=ARA_BS); gpu_sync(); end
        e = rel_error(A_cpu, T3); rk = Array(ranks(T3)); ok = !any(isnan, Array(T3.int_U)); ue = ortho_errors(T3)
        @printf("    Alg3 ARA       %8.4f s  rel_err=%.2e  ok=%-5s  ranks∈[%d,%d]  recovery=%d/%d\n",
                t, e, ok, minimum(rk), maximum(rk), recovery(rk), noff)
        print_ortho_vec("Alg3", ue)
        @printf("    ARA basis columns: %d used vs %d fixed budget (%.0f%%)\n",
                sum(rk), budget, 100 * sum(rk) / budget)
        T3 = nothing; gpu_reclaim(); t
    end

    # ── Algorithm 4: FB-ARA (fixed-budget, GPU-efficient sibling) ────────────
    # Fixed sampling steps, NO per-iteration host-synced compaction; ranks
    # certified at the end by the exact Frobenius residual (throws if a tile
    # cannot be certified within the budget).
    t4 = try
        T4 = TLRMatrix(A, B, MAXRANK); gpu_sync()
        fb_ara_compress!(T4, A; tol=FB_TOL, bs=ARA_BS, T=FB_STEPS); gpu_sync()  # warmup
        t = @elapsed begin; fb_ara_compress!(T4, A; tol=FB_TOL, bs=ARA_BS, T=FB_STEPS); gpu_sync(); end
        e = rel_error(A_cpu, T4); rk = Array(ranks(T4)); ok = !any(isnan, Array(T4.int_U)); ue = ortho_errors(T4)
        @printf("    Alg4 FB-ARA    %8.4f s  rel_err=%.2e  ok=%-5s  ranks∈[%d,%d]  cols=%d (%.0f%% of budget)\n",
                t, e, ok, minimum(rk), maximum(rk), sum(rk), 100 * sum(rk) / budget)
        print_ortho_vec("Alg4", ue)
        T4 = nothing; gpu_reclaim(); t
    catch err
        @printf("    Alg4 FB-ARA    could not certify all tiles: %s\n",
                err isa ArgumentError ? sprint(showerror, err) : string(typeof(err)))
        gpu_reclaim(); NaN
    end

    if isnan(t4)
        @printf("    speedup vs Alg1:  Alg2 %.2fx   Alg3(ARA) %.2fx\n", t1/t2, t1/t3)
    else
        @printf("    speedup vs Alg1:  Alg2 %.2fx   Alg3(ARA) %.2fx   Alg4(FB-ARA) %.2fx\n",
                t1/t2, t1/t3, t1/t4)
    end
end

# ── per-tile orthogonality errors for U and V ────────────────────────────────
function ortho_errors(A_tlr::TLRMatrix)
    rk      = Array(ranks(A_tlr))
    noff    = noffdiag_tiles(A_tlr)
    u_errs  = Float64[]
    for ob in 1:noff
        kr = Int(rk[ob])
        kr == 0 && continue
        Uob = Array(tile_u(A_tlr, ob))
        Ikr = Matrix{eltype(Uob)}(I, kr, kr)
        push!(u_errs, norm(Uob' * Uob - Ikr))
    end
    return u_errs
end

function print_ortho_summary(label::AbstractString, A_tlr::TLRMatrix)
    print_ortho_vec(label, ortho_errors(A_tlr))
end

function print_ortho_vec(label::AbstractString, u_errs::AbstractVector)
    isempty(u_errs) && return
    @printf("      %-4s U ortho ||UᵀU-I||_F: min=%.2e  median=%.2e  max=%.2e\n",
            label, minimum(u_errs), median(u_errs), maximum(u_errs))
end

# ── main ──────────────────────────────────────────────────────────────────────
const B       = 128
const MAXRANK = B ÷ 2    # a b×b tile carries at most b/2 rank before dense is cheaper
const TOL     = 1.0f0    # Frobenius budget for removed columns; null columns
                         # have V-norms ≈ 0 so any TOL > 0 removes them
const RANK_DIST = :skewed  # high variance: most tiles tiny rank, a few near b/2
const ARA_TOL   = 1.0f-3   # ARA relative singular-value tolerance
const ARA_BS    = 16       # ARA / FB-ARA sampling block size
const FB_STEPS  = cld(MAXRANK, ARA_BS)  # FB-ARA fixed sampling steps (T*bs ≤ maxrank)
const FB_TOL    = 1.0f-3   # FB-ARA relative Frobenius tolerance for rank certification

println("NextLA TLR compress! benchmark")
println("  tile size b=$B  maxrank=$MAXRANK  tol=$TOL")
println("  off-diagonal tile ranks: ", RANK_DIST === :skewed ?
        "1..$(B ÷ 2) (skewed, high variance)" : "10..20 (uniform)")
if HAS_CUDA
    println("  GPU: ", CUDA.name(CUDA.device()))
else
    println("  GPU: not available (run with ../gpuenv to enable CUDA)")
end
println()

println("view alloc check: ",
        @allocated(view(zeros(Float32,256,256), 1:128, 1:128)),
        " B (vs ", @allocated(copy(zeros(Float32,128,128))), " B for data copy)")
println()

for n in [2046]
    @printf("═══ n = %d ════════════════════════════════════════════\n", n)

    t_gen = @elapsed A_cpu, true_rk = generate_tiled_lowrank(n, B; dist=RANK_DIST)
    @printf("  matrix generated in %.3f s  (%.0f MB)\n\n", t_gen, n^2*4/1e6)

    if n<=8192
        run_case(A_cpu, true_rk, "CPU", B, MAXRANK, TOL)
    end

    if HAS_CUDA
        println()
        run_case(A_cpu, true_rk, "GPU", B, MAXRANK, TOL; to_device=CuArray)
    end

    println()
end
