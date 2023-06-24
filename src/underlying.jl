struct UnderlyingPOMDP{P<:CPOMDP,S,A,O} <: POMDP{S,A,O}
    m::P
end

UnderlyingPOMDP(p::POMDP) = p

POMDPs.POMDP(m::CPOMDP) = UnderlyingPOMDP(m)

function UnderlyingPOMDP(c_pomdp::CPOMDP{S, A, O}) where {S,A,O}
    return UnderlyingPOMDP{typeof(c_pomdp),S,A,O}(c_pomdp)
end

UnderlyingPOMDP(c_pomdp::CPOMDPWrapper) = c_pomdp.m

POMDPs.reward(m::UnderlyingPOMDP, s, a, sp, o)      = reward(m.m, s, a, sp, o)
POMDPs.reward(m::UnderlyingPOMDP, s, a, sp)         = reward(m.m, s, a, sp)
POMDPs.reward(m::UnderlyingPOMDP, s, a)             = reward(m.m, s, a)
POMDPs.states(m::UnderlyingPOMDP)                   = states(m.m)
POMDPs.actions(m::UnderlyingPOMDP)                  = actions(m.m)
POMDPs.observations(m::UnderlyingPOMDP)             = observations(m.m)
POMDPs.stateindex(m::UnderlyingPOMDP, s)            = stateindex(m.m, s)
POMDPs.actionindex(m::UnderlyingPOMDP, a)           = actionindex(m.m, a)
POMDPs.obsindex(m::UnderlyingPOMDP, o)              = obsindex(m.m, o)
POMDPs.discount(m::UnderlyingPOMDP)                 = discount(m.m)
POMDPs.initialstate(m::UnderlyingPOMDP)             = initialstate(m.m)
POMDPs.transition(m::UnderlyingPOMDP, s, a)         = transition(m.m, s, a)
POMDPs.observation(m::UnderlyingPOMDP, a, s)        = observation(m.m, a, s)
POMDPs.isterminal(m::UnderlyingPOMDP, s)            = isterminal(m.m, s)
POMDPTools.ordered_states(m::UnderlyingPOMDP)       = ordered_states(m.m)
POMDPTools.ordered_actions(m::UnderlyingPOMDP)      = ordered_actions(m.m)
POMDPTools.ordered_observations(m::UnderlyingPOMDP) = ordered_observations(m.m)
