struct Node{R<:AbstractFloat,N}
    ind::CartesianIndex{N}
    val::R
end

Base.isless(a::Node, b::Node) = isless(a.val, b.val)

# Tags for nodes in heap set when marching
const _TAGS_UNKNOWN::UInt8 = 0x1
const _TAGS_FRONT::UInt8 = 0x2
const _TAGS_KNOWN::UInt8 = 0x3

"""
    FEFMMGrid{N,R,A<:AbstractArray{<:R,N}}

Struct holding the information and working grids needed to compute
the eikonal solution via fast marching.

`N` is the number of dimensions of the problem, and `R` is the real
number type used.  `A` is the type of the grid holding the
squared slowness value of the field.
"""
struct FEFMMGrid{N,R,A,H}
    κ2::A
    dx::NTuple{N,R}
    cs::NTuple{N,CartesianIndex{N}}
    inds::CartesianIndices{N, NTuple{N, Base.OneTo{Int}}}
    I1::CartesianIndex{N}
    Iend::CartesianIndex{N}
    τ0::A
    ∇τ0::Vector{A}
    τ1::A
    tags::Array{UInt8,N}
    front::H
    α::Vector{R}
    β::Vector{R}
    LI::LinearIndices{N, NTuple{N, Base.OneTo{Int}}}
    xn::Vector{CartesianIndex{N}}
end

"""
    FEFMMGrid(κ2, dx)

Construct a grid of squared slowness values (`κ2`) upon which the
Eikonal solution will be found via the factored eikonal fast-marching
method using [`fefmm!`](@ref).

`dx` is a tuple of grid spacings for each dimension (length of `ndims(κ2)`).
"""
function FEFMMGrid(κ2::AbstractArray{<:R,<:N}, dx) where {R,N}
    length(dx) == N ||
        throw(ArgumentError(
            "dimensions of grid ($(size(κ2))) and grid spacing ($(dx)) do not match"
        ))

    # Ensure spacing is of correct type
    dx_R = convert(NTuple{N,R}, dx)

    # Create arrays
    cs = cartstrides(κ2)
    inds = CartesianIndices(κ2)
    I1 = first(inds)
    Iend = last(inds)
    τ0 = similar(κ2)
    ∇τ0 = [similar(κ2) for _ in 1:N]
    τ1 = similar(κ2)
    tags = similar(κ2, UInt8)

    front = BinaryMinHeap{Node{R,N}}()

    α = Vector{R}(undef, N)
    β = similar(α)
    LI = LinearIndices(τ1)
    xn = Array{typeof(I1)}(undef, N*2)

    FEFMMGrid(
        κ2, dx_R, cs, inds, I1, Iend, τ0, ∇τ0, τ1, tags, front, α, β, LI, xn
    )
end

function Base.show(io::IO, ::MIME"text/plain", grid::FEFMMGrid{N,R,A,H}) where {N,R,A,H}
    println(io, "FEFMMGrid{$N,$R,$A,$H}:")
    println(io, "    κ2 range: $(extrema(grid.κ2))")
    println(io, "  grid nodes: $(size(grid.κ2))")
    print(io,   "          dx: $(grid.dx)")
end
