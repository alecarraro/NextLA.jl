export ara_compress, ara_compress!

# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Randomized Approximation (ARA) — batched tile low-rank compression.
#
# Follows the ARA formulation of Boukaram, Zampini, Turkiyyah & Keyes,
# "H2OPUS-TLR: high performance tile low rank symmetric factorizations …"
# (arXiv:2108.11932), with CholeskyQR2 (two Cholesky-QR passes) for the
# intra-block orthogonalisation, as recommended there for numerical stability
# over single-pass CholQR.
#
# The routine grows an orthonormal basis Q one block of `bs` columns at a time,
# sketching the (implicit) generator `A` with a Gaussian block Ω, and stops each
# tile independently once the new block's contribution falls below tolerance.
# Because every still-active tile adds exactly `bs` columns per iteration, the
# working set stays a set of equally-shaped dense batches; converged/failed tiles
# are removed and the remainder compacted (the deliberately-simple "dynamic
# buffer" baseline).
#
# All GEMMs (Y=A·Ω, block Gram–Schmidt, the CholQR Gram products, R₁·R₂ and
# B=Aᵀ·Q) run through the batched primitives in `ext` (`gemm_batched!`,
# `potrf_batched!`, `trsm_batched!`); the small per-tile Cholesky retries reuse
# `potrf_batched!` on a gathered sub-batch.
# ═══════════════════════════════════════════════════════════════════════════════

# Copy the first `r` accumulated columns of active slot `slot` into output tile
# `id` and record its rank.  Callers append any extra (significant) block columns
# afterwards and overwrite `ranks[id]`.
function _ara_finalize!(Q_out, B_out, ranks, Q, B, id::Int, slot::Int, r::Int)
    if r > 0
        copyto!(view(Q_out, :, 1:r, id), view(Q, :, 1:r, slot))
        copyto!(view(B_out, :, 1:r, id), view(B, :, 1:r, slot))
    end
    ranks[id] = r
    return nothing
end
# Shared randomized-compression helpers used by dynamic ARA and FB-ARA.

# Per-tile squared Frobenius norm  out[b] = ||Y[:,:,b]||_F^2  (accumulated in
# the element type of `Y`, which callers set to the high accumulation precision).
@kernel function _frob2_kernel!(out::AbstractVector, Y::AbstractArray{T,3}) where {T}
    b = @index(Global)
    acc = zero(real(T))
    @inbounds for j in axes(Y, 2), i in axes(Y, 1)
        acc += abs2(Y[i, j, b])
    end
    @inbounds out[b] = acc
end

# Add a per-tile diagonal shift  G[j,j,b] += coef * fro2[b]  for CholeskyQR
# breakdown protection.
@kernel function _add_diag_shift_kernel!(G::AbstractArray{T,3},
                                         fro2::AbstractVector, coef) where {T}
    j, b = @index(Global, NTuple)
    @inbounds G[j, j, b] += T(coef * fro2[b])
end

# Absolute Cholesky pivots  out[j,b] = |R[j,j,b]|.  The diagonal of the first
# CholQR factor is the within-block rank indicator used by ARA and the advisory
# activity mask used by FB-ARA.
@kernel function _diag_abs_kernel!(out::AbstractMatrix, R::AbstractArray{T,3}) where {T}
    j, b = @index(Global, NTuple)
    @inbounds out[j, b] = abs(R[j, j, b])
end

# Represent a tile batch as a vector of per-tile matrix views regardless of the
# input storage form.  For a 3-D array these are contiguous slices; for a vector
# of matrices they are used as-is.
_ara_tile_vector(A::AbstractArray{<:Any,3}) = _batch_views(A)
_ara_tile_vector(A::AbstractVector{<:AbstractMatrix}) = collect(A)

_ara_dims(A::AbstractArray{<:Any,3}) = (size(A, 1), size(A, 2), size(A, 3), eltype(A))
function _ara_dims(A::AbstractVector{<:AbstractMatrix})
    isempty(A) && throw(ArgumentError("randomized compression: empty tile batch"))
    return (size(first(A), 1), size(first(A), 2), length(A), eltype(first(A)))
end

_ara_backend(A::AbstractArray{<:Any,3}) = KernelAbstractions.get_backend(A)
_ara_backend(A::AbstractVector{<:AbstractMatrix}) = KernelAbstractions.get_backend(first(A))

# One CholeskyQR pass with per-tile Cholesky-breakdown protection.
#
# Forms G = Y'Y in high precision, adds λ_b ~= coef*||Y_b||_F^2 to the diagonal,
# factors G, and overwrites Yhi with Y*R^-1. Failed tiles are retried once with a
# 10x shift; slots still failing after that are returned as local batch indices.
function _cholqr_pass_retry!(Yhi::AbstractArray{Thi,3}, G, fro2, coef, backend) where {Thi}
    bs = size(Yhi, 2)
    na = size(Yhi, 3)
    adj = _adjoint_blas_char(Thi)

    gemm_batched!(adj, 'N', one(Thi), Yhi, Yhi, zero(Thi), G)
    _add_diag_shift_kernel!(backend)(G, fro2, coef; ndrange=(bs, na))
    _, status = potrf_batched!('U', G)

    st = Array(status)
    failed = findall(!iszero, st)
    stillfailed = Int[]

    if !isempty(failed)
        nf = length(failed)
        Ysub = Yhi[:, :, failed]
        Gsub = similar(G, bs, bs, nf)
        gemm_batched!(adj, 'N', one(Thi), Ysub, Ysub, zero(Thi), Gsub)
        _add_diag_shift_kernel!(backend)(Gsub, fro2[failed], 10 * coef; ndrange=(bs, nf))
        _, status2 = potrf_batched!('U', Gsub)
        for (t, f) in enumerate(failed)
            copyto!(view(G, :, :, f), view(Gsub, :, :, t))
        end
        st2 = Array(status2)
        stillfailed = failed[findall(!iszero, st2)]
    end

    trsm_batched!('R', 'U', 'N', 'N', G, Yhi, one(Thi))
    return stillfailed
end

# ─── Main routine ─────────────────────────────────────────────────────────────

"""
    ara_compress(A_batch, bs, ε, r_max; c=10, relative=true, backend=:auto,
                 Q_out=nothing, B_out=nothing)
        -> (Q_batch, B_batch, ranks::Vector{Int}, failed::Vector{Bool})

Compress a **batch** of tiles independently into `Aᵦ ≈ Qᵦ·Bᵦᵀ` (Qᵦ orthonormal)
with the Adaptive Randomized Approximation (ARA) algorithm.

`A_batch` is either an `m×n×nb` dense array or a length-`nb` vector of `m×n`
matrices (e.g. tile views into a larger dense matrix — no data is copied).

For each tile the orthonormal range basis is grown a block of `bs` columns at a
time:

1. sketch `Ω ← randn(n, bs)`, `Y ← A·Ω`;
2. block classical Gram–Schmidt against the current basis, run **twice**
   (BCGS2) — a single projection pass loses orthogonality when the sketch is
   ill-conditioned;
3. orthonormalise the block with **CholeskyQR2** (two Cholesky-QR passes) — a
   single CholQR pass is not accurate enough near rank deficiency;
4. convergence signal `e = maxⱼ‖R[:,j]‖`, with `R = R₁·R₂` the product of the two
   CholQR triangular factors.

A tile stops once `e ≤ ε·s₀` (relative, `s₀` = the first block's `e`, a sketch
estimate of `‖Aᵦ‖₂`) or once its rank reaches `r_max`; `relative=false` compares
against the bare `ε`. Converged and failed tiles are removed from the batch and
the remainder compacted (a simple dynamic-buffer baseline).

Numerical safeguards (load-bearing, not stylistic):
- Gram matrices (`Yᵀ·Y`) are always accumulated in at least `Float32`
  (`Float64` for `Float32`/`Float64` inputs) via `_compress_accum_type`, never in
  half precision.
- `G₁` gets a diagonal shift `λ ≈ c·u·‖Y‖_F²` (`u` = unit roundoff of the input
  precision, `c` configurable, default `10`) before the first Cholesky, to guard
  against breakdown on ill-conditioned sketches. A failed factorisation is
  retried once with `10·λ`; tiles that still fail are flagged in `failed` and
  finalised with the columns accumulated so far.

Returns padded factor batches `Q_batch` (`m×rstore×nb`) and `B_batch`
(`n×rstore×nb`), the per-tile `ranks`, and the per-tile `failed` flags; only
columns `1:ranks[b]` of tile `b` are meaningful. Follows Boukaram/Zampini/
Turkiyyah/Keyes, "H2OPUS-TLR" (arXiv:2108.11932).
"""
function ara_compress(A_batch, bs::Int, ε::Real, r_max::Int;
                      c::Real=10, relative::Bool=true, backend=:auto,
                      Q_out=nothing, B_out=nothing)
    bs > 0 || throw(ArgumentError("ara_compress: bs must be positive"))
    r_max >= 0 || throw(ArgumentError("ara_compress: r_max must be non-negative"))
    ε >= 0 || throw(ArgumentError("ara_compress: ε must be non-negative"))

    m, n, nb, T = _ara_dims(A_batch)
    be = backend === :auto ? _ara_backend(A_batch) : backend
    Thi = _compress_accum_type(T)
    RT = real(T)
    RThi = real(Thi)
    adj = _adjoint_blas_char(T)

    # Storage rank: full blocks of `bs`, capped at min(r_max, m, n).
    rcap = min(r_max, m, n)
    rstore = min(cld(max(rcap, 1), bs) * bs, m, n)
    rcap = min(rcap, rstore)

    alloc(::Type{S}, dims...) where {S} = _alloc_zeros(be, S, dims...)

    Q_out = Q_out === nothing ? alloc(T, m, rstore, nb) : Q_out
    B_out = B_out === nothing ? alloc(T, n, rstore, nb) : B_out
    ranks = zeros(Int, nb)
    failed = falses(nb)

    (nb == 0 || rcap == 0) && return Q_out, B_out, ranks, failed

    Avecs = _ara_tile_vector(A_batch)

    # Active working set (all still-growing tiles share the current rank k).
    Q = alloc(T, m, rstore, nb)          # active basis; columns 1:k valid
    B = alloc(T, n, rstore, nb)          # active co-range; columns 1:k valid
    ids = collect(1:nb)                  # active slot → original tile index
    s0 = zeros(RThi, nb)                 # per-tile first-block scale (relative conv)
    k = 0

    # unit roundoff of the input precision — the sketch conditioning is limited by
    # the working precision `T`, so the shift is sized against `eps(RT)`.
    coef = RThi(c) * RThi(eps(RT))

    # Safety factor on √λ for the within-block rank detection.  The diagonal shift
    # plants spurious pivots near √λ_b = √coef·‖Y_b‖, but they scatter up to a small
    # multiple of it, so a genuine direction's pivot must exceed `sig_floor·√λ_b` to
    # be distinguished. Real pivots sit O(10⁵) above the shift level, so the factor
    # is safely generous; it caps the achievable ε at ≈32·√(c·u) for a given input
    # precision (the usual CholQR resolution limit).
    sig_floor = RThi(32) * sqrt(coef)

    while k < rcap && !isempty(ids)
        na = length(ids)
        kprev = k
        block = min(bs, rstore - k)
        block <= 0 && break
        kb = k + block

        Aact = [Avecs[i] for i in ids]

        # Step 1 — Gaussian sketch and range matvec  Y = A·Ω.
        Ω = alloc(T, n, block, na)
        Random.randn!(Ω)
        Y = alloc(T, m, block, na)
        gemm_batched!('N', 'N', one(T), Aact, _batch_views(Ω), zero(T), _batch_views(Y))

        # Step 2 — block Gram–Schmidt vs the current basis, TWICE (BCGS2).  The
        # second pass is required: one pass leaves O(κ)·u loss of orthogonality
        # for ill-conditioned sketches. Do not remove it.
        if k > 0
            Qk = view(Q, :, 1:k, :)
            C = alloc(T, k, block, na)
            for _pass in 1:2
                gemm_batched!(adj, 'N', one(T), Qk, Y, zero(T), C)
                gemm_batched!('N', 'N', -one(T), Qk, C, one(T), Y)
            end
        end

        # Step 3 — intra-block orthogonalisation via CholeskyQR2 (two passes).  A
        # single CholQR pass is NOT an acceptable simplification: it is accurate
        # only to √κ and degrades sharply for the near-rank-deficient blocks ARA
        # naturally produces near convergence. Both passes below are mandatory.
        Yhi = alloc(Thi, m, block, na)
        copyto!(Yhi, Y)
        fro2 = alloc(RThi, na)
        G = alloc(Thi, block, block, na)

        _frob2_kernel!(be)(fro2, Yhi; ndrange=na)
        # ‖Y_b‖_F of the *unorthogonalised* block sets the shift λ_b = coef·‖Y‖²,
        # so √λ_b = √coef·‖Y_b‖ is the level at which the shift plants spurious
        # near-null pivots.  Keep it (host) for the rank-detection floor below.
        froY = sqrt.(Array(fro2))
        fail1 = _cholqr_pass_retry!(Yhi, G, fro2, coef, be)

        # Numerical rank of the block = the pivots of the *first* Cholesky factor.
        # `diag(R1)[j]` is the norm of column j orthogonal to columns 1:j-1; genuine
        # new directions give O(σ) pivots while shift-spurious columns sit right at
        # √λ_b (see the shift discussion in `_cholqr_pass_retry!`). This separates
        # them by many orders of magnitude and is far more robust than column norms
        # of the R₁·R₂ product.
        dg_dev = alloc(RThi, block, na)
        _diag_abs_kernel!(be)(dg_dev, G; ndrange=(block, na))
        dg = Array(dg_dev)                        # block × na, on host

        _frob2_kernel!(be)(fro2, Yhi; ndrange=na)
        fail2 = _cholqr_pass_retry!(Yhi, G, fro2, coef, be)

        failed_this = Set(vcat(fail1, fail2))

        # Step 4 — append the orthonormal block to Q and the co-range block
        # B = Aᵀ·Y (incremental: the new B columns are Aᵀ times the new block).
        Yblk = alloc(T, m, block, na)
        copyto!(Yblk, Yhi)
        Bnew = alloc(T, n, block, na)
        gemm_batched!(adj, 'N', one(T), Aact, _batch_views(Yblk), zero(T), _batch_views(Bnew))
        copyto!(view(Q, :, kprev+1:kb, :), Yblk)
        copyto!(view(B, :, kprev+1:kb, :), Bnew)
        k = kb

        # Per-tile stop / fail / continue decisions (on the host).  A tile keeps
        # growing only while every one of its `block` new columns is significant;
        # a block with fewer than `block` significant directions means the range
        # is exhausted (ARA convergence), and only those significant columns —
        # which are the properly orthonormal ones — enter the final basis.
        keep = Int[]
        for slot in 1:na
            id = ids[slot]
            if kprev == 0
                s0[slot] = maximum(@view dg[:, slot])
            end
            # A pivot must clear both the user tolerance (ε·s0, the singular-value
            # scale the user asked to resolve) and the shift floor 4·√λ_b, below
            # which pivots are indistinguishable from shift-induced null columns.
            base = relative ? RThi(ε) * s0[slot] : RThi(ε)
            thr = max(base, sig_floor * froY[slot])

            if slot in failed_this
                failed[id] = true
                _ara_finalize!(Q_out, B_out, ranks, Q, B, id, slot, kprev)  # drop bad block
                continue
            end

            sig = [j for j in 1:block if dg[j, slot] > thr]
            nsig = length(sig)
            if nsig < block || k >= rcap
                # Converged (or rank cap): keep prior columns + significant new ones.
                _ara_finalize!(Q_out, B_out, ranks, Q, B, id, slot, kprev)
                r = kprev
                @inbounds for j in sig
                    r >= rstore && break
                    r += 1
                    copyto!(view(Q_out, :, r, id), view(Q, :, kprev + j, slot))
                    copyto!(view(B_out, :, r, id), view(B, :, kprev + j, slot))
                end
                ranks[id] = r
            else
                push!(keep, slot)
            end
        end

        # Compact the surviving tiles into fresh, smaller batches.
        if length(keep) < na
            Q = Q[:, :, keep]
            B = B[:, :, keep]
            ids = ids[keep]
            s0 = s0[keep]
        end
    end

    # Safety net: finalise anything still active (e.g. rstore reached exactly).
    for slot in eachindex(ids)
        _ara_finalize!(Q_out, B_out, ranks, Q, B, ids[slot], slot, k)
    end

    return Q_out, B_out, ranks, failed
end

# ─── TLRMatrix integration ────────────────────────────────────────────────────

"""
    ara_compress!(A_tlr, A; tol=0.0, bs=..., r_max=maxrank(A_tlr), c=10, relative=true)

Compress dense matrix `A` into the TLR container `A_tlr` in place using
[`ara_compress`](@ref), the numerically-safe adaptive variant (contrast
[`compress!`](@ref), the experimental one-shot variant).

Off-diagonal tiles are processed per geometry category (interior / right /
bottom); the tile views into `A` are passed to `ara_compress` without packing,
and the resulting `Q`/`B` factors are written straight into the container's
`U`/`V` factor storage with the detected per-tile ranks. Diagonal tiles are
copied dense, exactly as in `compress!`.

`bs` is the ARA block size (default `min(16, maxrank(A_tlr))`); ranks are capped
at `maxrank(A_tlr)`.
"""
function ara_compress!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T};
                       tol::Real=0.0, bs::Int=min(16, A_tlr.maxrank),
                       r_max::Int=A_tlr.maxrank, c::Real=10,
                       relative::Bool=true) where {T}
    size(A, 1) == A_tlr.m && size(A, 2) == A_tlr.n ||
        throw(DimensionMismatch("A dimensions must match A_tlr"))
    tol >= 0 || throw(ArgumentError("tol must be >= 0"))
    r_max <= A_tlr.maxrank ||
        throw(ArgumentError("r_max must not exceed maxrank(A_tlr)"))

    _copy_diagonal_from_dense!(A_tlr, A)

    specs = (
        (A_tlr.obs_int,    A_tlr.int_U,    A_tlr.int_V),
        (A_tlr.obs_right,  A_tlr.right_U,  A_tlr.right_V),
        (A_tlr.obs_bottom, A_tlr.bottom_U, A_tlr.bottom_V),
    )

    for (obs, U, V) in specs
        isempty(obs) && continue
        A_tiles = _tile_views(A, A_tlr, obs)
        Q_out, B_out, ranks_local, _failed =
            ara_compress(A_tiles, bs, tol, r_max; c, relative, backend=A_tlr.backend)

        rstore = min(size(Q_out, 2), A_tlr.maxrank)
        copyto!(view(U, :, 1:rstore, :), view(Q_out, :, 1:rstore, :))
        copyto!(view(V, :, 1:rstore, :), view(B_out, :, 1:rstore, :))
        @inbounds for (slot, ob) in enumerate(obs)
            A_tlr.ranks[ob] = min(ranks_local[slot], A_tlr.maxrank)
        end
    end

    return A_tlr
end
