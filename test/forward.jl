function _test_mdp_forward(wrapper, mdp, s, a, sp)
    @test reward(wrapper, s, a, sp) == reward(mdp, s, a, sp)
    @test reward(wrapper, s, a) == reward(mdp, s, a)
    @test states(wrapper) == states(mdp)
    @test actions(wrapper) == actions(mdp)
    @test stateindex(wrapper, s) == stateindex(mdp, s)
    @test actionindex(wrapper, a) == actionindex(mdp, a)
    @test discount(wrapper) == discount(mdp)
    @test initialstate(wrapper) == initialstate(mdp)
    @test transition(wrapper, s, a) == transition(mdp, s, a)
    @test isterminal(wrapper, s) == isterminal(mdp, s)
    @test ordered_states(wrapper) == ordered_states(mdp)
    @test ordered_actions(wrapper) == ordered_actions(mdp)
end

function _test_pomdp_forward(wrapper, pomdp, s, a, sp, o)
    @test reward(wrapper, s, a, sp, o) == reward(pomdp, s, a, sp, o)
    @test reward(wrapper, s, a, sp) == reward(pomdp, s, a, sp)
    @test reward(wrapper, s, a) == reward(pomdp, s, a)
    @test states(wrapper) == states(pomdp)
    @test actions(wrapper) == actions(pomdp)
    @test observations(wrapper) == observations(pomdp)
    @test stateindex(wrapper, s) == stateindex(pomdp, s)
    @test actionindex(wrapper, a) == actionindex(pomdp, a)
    @test obsindex(wrapper, o) == obsindex(pomdp, o)
    @test discount(wrapper) == discount(pomdp)
    @test initialstate(wrapper) == initialstate(pomdp)
    @test transition(wrapper, s, a) == transition(pomdp, s, a)
    @test observation(wrapper, a, sp) == observation(pomdp, a, sp)
    @test isterminal(wrapper, s) == isterminal(pomdp, s)
    @test ordered_states(wrapper) == ordered_states(pomdp)
    @test ordered_actions(wrapper) == ordered_actions(pomdp)
    @test ordered_observations(wrapper) == ordered_observations(pomdp)
end

@testset "forward" begin
    Base.@kwdef struct ConstrainedGW <: CMDP{GWPos, Symbol}
        gw::SimpleGridWorld = SimpleGridWorld()
    end

    @MDP_forward ConstrainedGW.gw

    wrapper = ConstrainedGW()
    s = GWPos(1,1)
    a = :up
    sp = GWPos(1,2)
    _test_mdp_forward(wrapper, wrapper.gw, s, a, sp)


    Base.@kwdef struct ConstrainedTiger <: CPOMDP{Bool, Int, Bool}
        tiger::TigerPOMDP = TigerPOMDP()
    end

    @POMDP_forward ConstrainedTiger.tiger

    wrapper = ConstrainedTiger()
    s = true; a = 0; sp = true; o = true
    
    _test_pomdp_forward(wrapper, wrapper.tiger, s, a, sp, o)
end
