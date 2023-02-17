using Test
using POMDPs
using ConstrainedPOMDPs
using POMDPTools
import POMDPModels: SimpleGridWorld, BabyPOMDP
# using CCPOMCP
# using CGCP

@testset "gen" begin
    pomdp = BabyPOMDP()
    cpomdp = ConstrainedPOMDPs.Constrain(pomdp,[1.0])

    ConstrainedPOMDPs.cost(m::typeof(cpomdp),s,a) = [1.0]

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
end

@testset "rollout" begin
    cpomdp = ConstrainedPOMDPs.Constrain(BabyPOMDP(),[1.0])
    sol = POMDPTools.RandomSolver()
    pol = solve(sol, cpomdp)
    sim = RolloutSimulator(max_steps=10)
    R, C = simulate(sim, c_bb, pol)
    @test R isa Float64
    @test C isa Vector{Float64}
end
