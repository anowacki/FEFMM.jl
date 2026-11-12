function solve_node(
    τ1::Array{R, N}, 
    α::AbstractVector{<:R},
    β::AbstractVector{<:R},
    τ0::Array{R, N},
    ∇τ0::Array{T},
    tags::Array{UInt8, N},
    κ2::Array{R, N},
    x::S,
    dx,
    cs::NTuple{N,CartesianIndex{N}},
    I1::S, 
    Iend::S
) where {N, R<:AbstractFloat, T<:Array{R, N}, S<:CartesianIndex{N}}
    # Loop over dimensions to set α and β
    for (i, s) in enumerate(cs)
        fwdok = check1fwd(τ1, τ0, tags, x, s, Iend)
        bwdok = check1bwd(τ1, τ0, tags, x, s, I1)
        if fwdok && bwdok 
            if @inbounds τ1[x+s]*τ0[x+s] < τ1[x-s]*τ0[x-s]
                bwkok = false
            else 
                fwdok = false
            end
        end
        # Note: working out the maths gives an additional 1/α for the β terms
        # to reduce total operations we redefine β = β'/α where β' is the β in the paper
        if fwdok 
            if check2fwd(τ1, τ0, tags, x, s, Iend)
                @inbounds α[i] = 1.5*τ0[x]/dx[i] - ∇τ0[i][x]
                @inbounds β[i] = (2*τ1[x+s] - 0.5*τ1[x+2*s])*τ0[x]/dx[i]
            else
                @inbounds α[i] = τ0[x]/dx[i] - ∇τ0[i][x]
                @inbounds β[i] = τ0[x]*τ1[x+s]/dx[i]
            end
        elseif bwdok
            if check2bwd(τ1, τ0, tags, x, s, I1)
                @inbounds α[i] = 1.5*τ0[x]/dx[i]+∇τ0[i][x]
                @inbounds β[i] = (2*τ1[x-s] - 0.5*τ1[x-2*s])*τ0[x]/dx[i]
            else
                @inbounds α[i] = τ0[x]/dx[i] + ∇τ0[i][x]
                @inbounds β[i] = τ0[x]*τ1[x-s]/dx[i]
            end
        else
            @inbounds α[i] = zero(R)
            @inbounds β[i] = zero(R)
        end
    end
    # All of the α and β should be set, so now we will proceed with solving the solve_quadratic
    # @inbounds solve_piecewise_quadratic(α..., β..., κ2[x])
    if N == 1
        @inbounds τ1y = solve_piecewise_quadratic(α[1], β[1], κ2[x])
    elseif N == 2
        @inbounds τ1y = solve_piecewise_quadratic(α[1], α[2], β[1], β[2], κ2[x])
    elseif N == 3
        @inbounds τ1y = solve_piecewise_quadratic(α[1], α[2], α[3], β[1], β[2], β[3], κ2[x])        
    end
    τ1y
end

function fefmm_loop!(
    τ1::Array{R, N}, 
    ocount::Integer,
    α::Vector{R},
    β::Vector{R},
    τ0::Array{R, N}, 
    ∇τ0::Array{T},
    tags::Array{UInt8, N}, 
    front::DataStructures.AbstractHeap{<:Node},
    κ2::Array{R,N},
    dx::Tuple,
    cs::NTuple{N,CartesianIndex{N}},
    xn::Vector{S},
    I1::S, 
    Iend::S, 
    LI::LinearIndices{N}
) where {N, R<:AbstractFloat, T<:Array{R, N}, S<:CartesianIndex{N}}
    while isempty(front) == false
        x = pop!(front).ind
        # As we are using a non-mutable minheap, we might have already set
        # this node to known on another pass (i.e. this could be an old
        # worthless version of the node). This check makes sure we don't
        # accidentally override something we already have computed.
        @inbounds if tags[x] < _TAGS_KNOWN
            ocount += 1
            @inbounds tags[x] = _TAGS_KNOWN
            set_neighbours!(xn, x, tags, cs, I1, Iend)
            for y in xn
                # To avoid allocating neighbours repeatedly we reuse `xn`
                # but have `y == x` as the "neighbour not acceptable" flag value
                if y != x
                    # Move neighbours to front
                    @inbounds tags[y] = _TAGS_FRONT
                    τ1y = solve_node(τ1, α, β, τ0, ∇τ0, tags, κ2, y, dx, cs, I1, Iend)
                    @inbounds if τ1y < τ1[y]
                        # Only update and add new node to heap if the new solution is smaller 
                        @inbounds τ1[y] = τ1y
                        @inbounds push!(front, Node(y, τ1[y]*τ0[y]))
                    end
                end
            end
        end
    end
end
function fefmm_loop!(grid::FEFMMGrid, ocount)
    fefmm_loop!(
        grid.τ1,
        ocount,
        grid.α,
        grid.β,
        grid.τ0,
        grid.∇τ0,
        grid.tags,
        grid.front,
        grid.κ2,
        grid.dx,
        grid.cs,
        grid.xn,
        grid.I1,
        grid.Iend,
        grid.LI,
    )
end


"""
    fefmm(κ2::Array{R, N}, dx::Tuple, xs::CartesianIndex{N}; return_factor=false) where {R<:AbstractFloat, N} -> τ1τ0, ∇τ0

Compute the distance from `xs` to each point in `κ2`, where `κ2` is a regular
grid of points of dimension `N`.  The distances are placed in a new
grid returned `τ1τ0` with the same dimensions as `κ2`.

If `κ2` is a grid of squared wave slownesses (i.e., squared inverse of
wave velocities), then `τ1τ0` is the travel time between each point and
the source location `xs`.

# Arguments
- `κ2`: array of slowness squared.
- `dx`: tuple of grid spacings of `κ2`.  I.e., `dx[1]` is the grid spacing in the
  first dimension of `κ2`, `dx[2]` the second, and so on.
- `xs`: index of source within grid of points.  This must be a `CartesianIndex`.

# Keyword arguments
- `return_factor=false`: If `true`, return only the factor `τ1` rather than
  `τ1τ0` (the default).

# Outputs
One of the following
- `τ1τ0`: travel time array from source (default; `return_factors=false`)
- `τ1`: factors (when `return_factors=true`)

!!! note
    If calculating multiple sets of travel times on the same grid of
    slowness-squared points, consider using [`fefmm!`](@ref) on a
    preallocated [`FEFMMGrid`](@ref) instead, as this
"""
function fefmm(
    κ2::Array{R,N},
    dx::Tuple,
    xs::CartesianIndex{N};
    return_factor=false
) where {R<:AbstractFloat, N}
    length(dx) == N ||
        throw(ArgumentError("`dx` must contain $N items (one grid spacing per dimension)"))

    # Initialization
    cs = cartstrides(κ2)
    inds = CartesianIndices(κ2)
    I1 = first(inds)
    Iend = last(inds)
    τ0 = ones(R, size(κ2))
    mul_analytic!(τ0, dx..., size(κ2)..., Tuple(xs)...)
    ∇τ0 = grad_analytic(τ0, dx..., size(κ2)..., Tuple(xs)...)
    τ1 = R(Inf).*ones(R, size(κ2))
    τ1[xs] = sqrt(κ2[xs])

    # Tags: 
    # 1 = unknown
    # 2 = front
    # 3 = known
    tags = ones(UInt8, size(κ2))
    tags[xs] = _TAGS_FRONT
    front = DataStructures.BinaryMinHeap{Node{R,N}}()
    push!(front, Node(inds[xs], τ1[xs]*τ0[xs])) 
    α = Array{R}(undef, length(size(τ1)))
    β = similar(α)
    LI = LinearIndices(τ1)
    xn = Array{typeof(I1)}(undef, N*2)

    # Main loop
    fefmm_loop!(τ1, 1, α, β, τ0, ∇τ0, tags, front, κ2, dx, cs, xn, I1, Iend, LI)

    if return_factor
        return τ1 # for some ML workflows you might only want the factor rather than the full solution
    else
        # Compute travel times via τ0 .* τ1
        # Avoid allocating a new array by updating and returning τ1
        τ1 .*= τ0
        return τ1
    end
end

function initialize!(grid::FEFMMGrid{N,R}, xs::CartesianIndex{N}) where {N,R}
    xs in CartesianIndices(grid.κ2) ||
        throw(ArgumentError("source index is not within grid"))

    grid.τ0 .= 1
    mul_analytic!(grid.τ0, grid.dx..., size(grid.κ2)..., Tuple(xs)...)
    grad_analytic!(grid.∇τ0, grid.τ0, grid.dx, xs)
    grid.τ1 .= R(Inf)
    grid.τ1[xs] = sqrt(grid.κ2[xs])

    grid.tags .= _TAGS_UNKNOWN
    grid.tags[xs] = _TAGS_FRONT
    empty!(grid.front)
    push!(grid.front, Node(grid.inds[xs], grid.τ1[xs]*grid.τ0[xs]))

    nothing
end
initialize!(grid::FEFMMGrid{N}, xs) where N = initialize!(grid, CartesianIndex{N}(xs))

"""
    fefmm!(grid, xs; return_factor=false) -> τ1τ0

Compute the distance from `xs` to each slowness-squared point within `grid.κ2`,
where `grid` is a [`FEFMMGrid`](@ref), returning `grid.τ1` by default.
If `return_factor` is `true`, then instead return only the factors `τ1`,
which is useful in some applications.  `xs` must be a tuple or `CartesianIndex`
giving the location of the source within the grid in terms of the array index.

If `grid.κ2` is a grid of squared wave slownesses (i.e., squared inverse of
wave velocities), then `τ1τ0` is the travel time between each point and
the source location `xs`.

!!! note
    The returned traveltime array is a member of `grid`, so if you wish to
    re-use `grid` again, make sure to copy or otherwise extract the values
    needed from `τ1τ0` (or `τ1`)

# Arguments
- `grid`: Pre-prepared FEFMM computation grid and slowness-squared values.
  (See [`FEFMMGrid`](@ref).)
- `xs`: index of source within grid of points.  This must be a `CartesianIndex`
  or a tuple of integers.

# Keyword arguments
- `return_factor=false`: If `true`, return only the factor `τ1` rather than
  `τ1τ0` (the default).

# Outputs
One of the following
- `τ1τ0`: travel time array from source (default; `return_factors=false`)
- `τ1`: factors (when `return_factors=true`)
"""
function fefmm!(grid::FEFMMGrid, xs; return_factor::Bool=false)
    initialize!(grid, xs)

    fefmm_loop!(grid, 1)

    if !return_factor
        # Compute travel times via τ0 .* τ1
        # Avoid allocating a new array by updating and returning τ1 even
        # if we are after the 
        grid.τ1 .*= grid.τ0
    end

    grid.τ1
end
