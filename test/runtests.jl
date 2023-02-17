using Test
using POMDPs
using ConstrainedPOMDPs
using POMDPTools
using POMDPModels

@testset "gen" begin
    pomdp = BabyPOMDP()
    cpomdp = ConstrainedPOMDPs.Constrain(pomdp,[1.0])
    ConstrainedPOMDPs.cost(m::typeof(cpomdp),s,a) = [1.0]

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
    cmdp = ConstrainedPOMDPs.Constrain(mdp,[1.0])
    ConstrainedPOMDPs.cost(m::typeof(cmdp),s,a) = [1.0]

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
    cpomdp = ConstrainedPOMDPs.Constrain(BabyPOMDP(),[1.0])
    ConstrainedPOMDPs.cost(m::typeof(cpomdp),s,a) = [1.0]

    sol = POMDPTools.RandomSolver()
    pol = solve(sol, cpomdp)
    sim = RolloutSimulator(max_steps=10)
    R, C = simulate(sim, cpomdp, pol)
    @test R isa Float64
    @test C isa Vector{Float64}

    cmdp = ConstrainedPOMDPs.Constrain(SimpleGridWorld(),[1.0])
    ConstrainedPOMDPs.cost(m::typeof(cmdp),s,a) = [1.0]
    sol = POMDPTools.RandomSolver()
    pol = solve(sol, cmdp)
    sim = RolloutSimulator(max_steps=10)
    R, C = simulate(sim, cmdp, pol)
    @test R isa Float64
    @test C isa Vector{Float64}
end
