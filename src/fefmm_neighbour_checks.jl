# function schecklt(a::Int, b1::Int, b2::Int, s::Int)
#     a < b && cld(b1, s) == cld(b2, s)
# end

cilt(a::S, b::S) where {S <: CartesianIndex} = all(a.I .<= b.I)

function cartstrides(A::AbstractArray{R,N}) where {R,N}
    ntuple(N) do i
        z = ntuple(ii -> ii == i ? 1 : 0, N)
        CartesianIndex(z...)
    end
end

function set_neighbours!(xn::Vector{S}, x::S, tags::Array{UInt8, N}, cs::NTuple{N,S}, I1::S, Iend::S) where {N, S <: CartesianIndex{N}}
    for (i,s) in enumerate(cs)
        if cilt(I1, x-s) && @inbounds tags[x-s] < _TAGS_KNOWN 
            @inbounds xn[2*i-1] = x-s
        else
            @inbounds xn[2*i-1] = x
        end
        if cilt(x+s, Iend) && @inbounds tags[x+s] < _TAGS_KNOWN
            @inbounds xn[2*i] = x+s
        else
            @inbounds xn[2*i] = x
        end
    end
end


@noinline function check1bwd(τ1::Array{R, N}, τ0::Array{R, N}, tags::Array{UInt8, N}, x::S, s::S, I1::S) where {N, R <: AbstractFloat, S <: CartesianIndex{N}}
    if cilt(I1, x-s) && @inbounds tags[x-s] == _TAGS_KNOWN
        return true
    else
        return false
    end
end

@noinline function check1fwd(τ1::Array{R, N}, τ0::Array{R, N}, tags::Array{UInt8, N}, x::S, s::S, Iend::S) where {N, R <: AbstractFloat, S <: CartesianIndex{N}}
    if cilt(x+s, Iend) && @inbounds tags[x+s] == _TAGS_KNOWN
        return true
    else
        return false
    end
end

@noinline function check2bwd(τ1::Array{R, N}, τ0::Array{R, N}, tags::Array{UInt8, N}, x::S, s::S, I1::S) where {N, R <: AbstractFloat, S <:CartesianIndex{N}}
    if cilt(I1, x-2*s) && @inbounds tags[x-2*s] == _TAGS_KNOWN && @inbounds τ1[x-s]*τ0[x-s] >= τ1[x-2*s]*τ0[x-2*s]
        return true
    else
        return false
    end 
end

@noinline function check2fwd(τ1::Array{R, N}, τ0::Array{R, N}, tags::Array{UInt8, N}, x::S, s::S, Iend::S) where {N, R <: AbstractFloat, S <:CartesianIndex{N}}
    if cilt(x+2*s, Iend) && @inbounds tags[x+2*s] == _TAGS_KNOWN && @inbounds τ1[x+s]*τ0[x+s] > τ1[x+2*s]*τ0[x+2*s]
        return true
    else
        return false
    end 
end

