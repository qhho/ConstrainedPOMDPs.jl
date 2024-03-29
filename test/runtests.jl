using Test
using POMDPs
using ConstrainedPOMDPs
const CPOMDPs = ConstrainedPOMDPs
using POMDPTools
using POMDPModels

include(joinpath(@__DIR__, "forward.jl"))

include(joinpath(@__DIR__, "sparse_tabular.jl"))

@testset "gen" begin
    pomdp = BabyPOMDP()
    cpomdp = ConstrainedPOMDPs.constrain(pomdp,[1.0]) do s,a
        [1.0]
    end

    # POMDP
    s = false
    a = false
    sp, o, r, c = @gen(:sp,:o,:r,:c)(cpomdp, s, a)
    @test sp isa Bool
    @test o isa Bool
    @test r isa Float64
    @test c isa Vector{Float64}

    # check that original gen isn't overwritten
    sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
    @test sp isa Bool
    @test o isa Bool
    @test r isa Float64

    ## MDP
    mdp = SimpleGridWorld()
    cmdp = ConstrainedPOMDPs.constrain(mdp,[1.0]) do s,a
        [1.0]
    end

    s = GWPos(2,2)
    a = :up
    sp, r, c = @gen(:sp,:r,:c)(cmdp, s, a)
    @test sp isa GWPos
    @test r isa Float64
    @test c isa Vector{Float64}

    # check that original gen isn't overwritten
    sp, r = @gen(:sp,:r)(mdp, s, a)
    @test sp isa GWPos
    @test r isa Float64
end

@testset "rollout" begin
    cpomdp = ConstrainedPOMDPs.constrain(BabyPOMDP(),[1.0]) do s,a
        [1.0]
    end

    sol = POMDPTools.RandomSolver()
    pol = solve(sol, cpomdp)
    sim = RolloutSimulator(max_steps=10)
    R, C = simulate(sim, cpomdp, pol)
    @test R isa Float64
    @test C isa Vector{Float64}

    cmdp = ConstrainedPOMDPs.constrain(SimpleGridWorld(),[1.0]) do s,a
        [1.0]
    end
    sol = POMDPTools.RandomSolver()
    pol = solve(sol, cmdp)
    sim = RolloutSimulator(max_steps=10)
    R, C = simulate(sim, cmdp, pol)
    @test R isa Float64
    @test C isa Vector{Float64}
end

@testset "ineq" begin
    v1 = [1.,2.,3.]
    v2 = [2.,3.,4.]

    @test v1 ⪯ v2
    @test v1 ≺ v2
    @test v2 ⪰ v1
    @test v2 ≻ v1

    v1 = [2.,2.,3.]
    v2 = [2.,3.,4.]

    @test v1 ⪯ v2
    @test !(v1 ≺ v2)
    @test v2 ⪰ v1
    @test !(v2 ≻ v1)

    v1 = [1,2,3]
    v2 = [3,2,1]
    @test !(v1 ⪯ v2)
    @test !(v1 ≺ v2)
    @test !(v2 ⪰ v1)
    @test !(v2 ≻ v1)
end
