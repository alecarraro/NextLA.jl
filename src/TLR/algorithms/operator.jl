"""
    AbstractTLROperator{T}

Marker abstract type for matrix-free operators that can participate in TLR
compression. Users are not required to subtype this type: any object that
supports `size`, `eltype`, `mul!(Y, op, X)`, and `mul!(Y, adjoint(op), X)` can
be passed to [`compress!`](@ref).
"""
abstract type AbstractTLROperator{T} end

Base.eltype(::Type{<:AbstractTLROperator{T}}) where {T} = T
Base.eltype(::AbstractTLROperator{T}) where {T} = T

"""
    TLRLinearOperator(::Type{T}, m, n, apply!, apply_adjoint!)

Convenience wrapper for matrix-free TLR compression. The callbacks must
implement in-place block application:

- `apply!(Y, X)` computes `Y := A * X`
- `apply_adjoint!(Y, X)` computes `Y := A' * X` (or `Aᴴ * X` for complex `T`)
"""
struct TLRLinearOperator{T,F,G} <: AbstractTLROperator{T}
    m::Int
    n::Int
    apply!::F
    apply_adjoint!::G

    function TLRLinearOperator{T}(m::Integer, n::Integer, apply!::F, apply_adjoint!::G) where {T,F,G}
        m > 0 || throw(ArgumentError("m must be positive"))
        n > 0 || throw(ArgumentError("n must be positive"))
        return new{T,F,G}(Int(m), Int(n), apply!, apply_adjoint!)
    end
end

function TLRLinearOperator(::Type{T},
                           m::Integer,
                           n::Integer,
                           apply!::F,
                           apply_adjoint!::G) where {T,F,G}
    return TLRLinearOperator{T}(m, n, apply!, apply_adjoint!)
end

Base.size(op::TLRLinearOperator) = (op.m, op.n)
Base.size(op::TLRLinearOperator, d::Int) = size(op)[d]

function LinearAlgebra.mul!(Y::AbstractVecOrMat, op::TLRLinearOperator{T}, X::AbstractVecOrMat) where {T}
    op.apply!(Y, X)
    return Y
end

struct TLRAdjointOperator{T,Op<:AbstractTLROperator{T}} <: AbstractTLROperator{T}
    parent::Op
end

Base.size(op::TLRAdjointOperator) = reverse(size(op.parent))
Base.size(op::TLRAdjointOperator, d::Int) = size(op)[d]
Base.adjoint(op::AbstractTLROperator{T}) where {T} = TLRAdjointOperator{T,typeof(op)}(op)

function LinearAlgebra.mul!(Y::AbstractVecOrMat, op::TLRAdjointOperator{T}, X::AbstractVecOrMat) where {T}
    op.parent.apply_adjoint!(Y, X)
    return Y
end
