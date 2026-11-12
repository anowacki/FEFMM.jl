using Test
using FEFMM
using LinearAlgebra: norm
import UnicodePlots

include("testmedia.jl")

function convergence_test(hlist, test_fun, test_name)
    @testset "$test_name" begin
        linorm = zeros(length(hlist))
        l2norm = zeros(length(hlist))
        for i = 1:length(hlist)
            (x0, k2, t_exact) = test_fun(hlist[i]) 
            t_pred = FEFMM.fefmm(k2, ntuple(_ -> hlist[i], ndims(k2)), x0)
            err = t_exact.-t_pred
            linorm[i] = norm(err, Inf)
            l2norm[i] = norm(err, 2)/sqrt(length(err))
        end

        # Don't create plots running on a CI platform, which normally
        # has the CI environment variable set
        if !haskey(ENV, "CI")
            plt = UnicodePlots.lineplot(
                log10.(hlist), 
                log10.(linorm);
                title=test_name,
                name="L_inf norm",
                xlabel="log_10(h)",
                ylabel="log_10(Error)", 
                xlim=[floor(minimum(log10.(hlist))), ceil(maximum(log10.(hlist)))],
                ylim=[floor(minimum(log10.([linorm; l2norm]))), ceil(maximum(log10.([linorm; l2norm])))],
                color=:blue
            )
            
            UnicodePlots.lineplot!(
                plt, log10.(hlist), log10.(l2norm), name="mean L2 norm", color=:red
            )
            println(plt)
        end

        @test reverse(linorm) == sort(linorm) #test for decreasing error with decreasing h
        @test reverse(l2norm) == sort(l2norm)
    end
end

@testset "FEFMM tests" begin
    @testset "Solve piecewise quadratic" begin
        @test isapprox(FEFMM.solve_piecewise_quadratic(1.0,1.0,1.0), 2.0)
        @test isapprox(FEFMM.solve_piecewise_quadratic(1.0,1.0,1.0,1.0,1.0), 1.0+1/sqrt(2))
        @test isapprox(FEFMM.solve_piecewise_quadratic(1.0,1.0,100.0,1.0,1.0), 2.0)
        @test isapprox(FEFMM.solve_piecewise_quadratic(2.0,1.0,1.0,100.0,1.0), 1.0)
        @test isapprox(FEFMM.solve_piecewise_quadratic(1.0,1.0,1.0,1.0,1.0,1.0,1.0), 1.0+1/sqrt(3))
        @test isapprox(FEFMM.solve_piecewise_quadratic(1.0,2.0,3.0,100.0,10.0,1.0,1.0), 2.0/3.0)
        @test isapprox(FEFMM.solve_piecewise_quadratic(1.0,2.0,3.0,10.0,100.0,1.0,1.0), 2.0/3.0)
        @test isapprox(FEFMM.solve_piecewise_quadratic(1.0,2.0,3.0,10.0,1.0,100.0,1.0), 1.0)
        @test isapprox(FEFMM.solve_piecewise_quadratic(1.0,2.0,3.0,1.0,10.0,100.0,1.0), 2.0)
    end

    @testset "Basic FEFMM checks" begin
        κ21d = ones(101)
        dx1d = (0.1,)
        xs1d = CartesianIndex(1)
        τa1d = 0:0.1:10
        @test all(isapprox.(τa1d, FEFMM.fefmm(κ21d, dx1d, xs1d)))
        κ22d = ones(101,101)
        dx2d = (0.1,0.1)
        xs2d = CartesianIndex(1,1)
        τaxis = 0:0.1:10
        τa2d = sqrt.(τaxis'.^2 .+ τaxis.^2)
        @test all(isapprox.(τa2d, FEFMM.fefmm(κ22d, dx2d, xs2d)))
        @test all(isapprox.(Float32.(τa2d), Float32.(FEFMM.fefmm(κ22d, Float32.(dx2d), xs2d))))
        κ23d = ones(101, 101, 101)
        dx3d = (0.1, 0.1, 0.1)
        xs3d = CartesianIndex(1, 1, 1)
        τa3d = zeros(101, 101, 101)
        for k in 1:101
            for j in 1:101
                for i in 1:101
                    τa3d[i,j,k] = sqrt(τaxis[i]^2 + τaxis[j]^2 + τaxis[k]^2)
                end
            end
        end
        @test all(isapprox.(τa3d, FEFMM.fefmm(κ23d, dx3d, xs3d)))
        @test all(isapprox.(Float32.(τa3d), Float32.(FEFMM.fefmm(κ23d, Float32.(dx3d), xs3d))))
    end

    @testset "Congruity of fefmm and fefmm!" begin
        κ2 = rand(20, 20)
        dx = (0.1, 0.2)
        xs = CartesianIndex(10, 10)
        grid = FEFMMGrid(κ2, dx)
        @test fefmm(κ2, dx, xs) == fefmm!(grid, xs)

        @testset "Idempotency of fefmm!" begin
            xs2 = CartesianIndex(1, 1)
            @test fefmm(κ2, dx, xs2) == fefmm!(grid, xs2)
        end
    end

    @testset "No reallocation when solving with same FEFMMGrid" begin
        grid = FEFMMGrid(ones(5, 5, 5), (2, 2, 2))
        xs = CartesianIndex(2, 2, 2)
        # First solve will allocate for the binary heap
        fefmm!(grid, xs)
        # Subsequent solves should not allocate
        @test @allocated(fefmm!(grid, xs)) == 0
    end

    @testset "Argument checking" begin
        @test_throws ArgumentError FEFMM.fefmm(ones(3, 4), (1., 1., 1.), CartesianIndex(1, 1))
        @test_throws MethodError FEFMM.fefmm(ones(3, 4), [1., 1.], CartesianIndex(1, 1))
        @test_throws ArgumentError FEFMMGrid(ones(2, 3), (1, 2, 3))
        @test_throws ArgumentError fefmm!(FEFMMGrid(ones(2, 3), (1, 1)), CartesianIndex(0, 0))
    end

    @testset "Convergence tests" begin
        convergence_test([1/20, 1/40, 1/80, 1/160, 1/320], const_k2_2D, "Constant Gradient of Squared Slowness 2D")
        convergence_test([1/20, 1/40, 1/80, 1/160, 1/320], const_v2_2D, "Constant Gradient of Squared Velocity 2D")
        convergence_test([1/20, 1/40, 1/80, 1/160, 1/320], gaussian_factor_2D, "Gaussian Factor 2D")
        convergence_test([1/20, 1/40, 1/80], const_k2_3D, "Constant Gradient of Squared Slowness 3D")
        convergence_test([1/20, 1/40, 1/80], const_v2_3D, "Constant Gradient of Squared Velocity 3D")
        convergence_test([1/20, 1/40, 1/80], gaussian_factor_3D, "Gaussian Factor 3D")
    end
end
