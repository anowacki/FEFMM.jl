module FEFMM

    using DataStructures: DataStructures, BinaryMinHeap
    
    export
        FEFMMGrid,
        fefmm!,
        fefmm

    include("types.jl")
    # Small helper functions
    include("fefmm_neighbour_checks.jl")
    include("fefmm_piecewise_quadratic.jl")
    include("fefmm_utilities.jl")
    # Core functions (initialization, main loop, node solve)
    include("factoredeikonalfmm.jl")

end 
