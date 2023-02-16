"""
    Constrain(m::Union{MDP. POMDP}, cost_constraints::Vector{Float64})

"""
Constrain(m::MDP, constraints::Vector{Float64}) = ConstrainedMDPWrapper(m, constraints)
Constrain(m::POMDP, constraints::Vector{Float64}) = ConstrainedPOMDPWrapper(m, constraints)


############

# Wrapper types

############

struct ConstrainedMDPWrapper{S,A, M<:MDP} <: MDP{Tuple{S, Int}, A}
    m::M
    constraints::Vector{Float64}
end

function ConstrainedMDPWrapper(m::MDP{S,A}, c::Vector{Float64}) where {S,A}
    return ConstrainedMDPWrapper{S,A,typeof(m)}(m, c)
end

struct ConstrainedPOMDPWrapper{S,A,O,M<:POMDP} <: POMDP{Tuple{S, Int}, A, Tuple{O,Int}}
    m::M
    constraints::Vector{Float64}
end

function ConstrainedPOMDPWrapper(m::POMDP{S,A,O}, c::Vector{Float64}) where {S,A,O}
    return ConstrainedPOMDPWrapper{S,A,O,typeof(m)}(m, c)
end

const CMDP = ConstrainedMDPWrapper
const CPOMDP = ConstrainedPOMDPWrapper
const ConstrainWrapper = Union{CMDP, CPOMDP}

##################

# ConstrainedPOMDP interface

##################
"""
    Return the constraints
"""
function constraints end

# constraints(::Type{<:ConstrainWrapper}) = FiniteHorizon()
constraints(w::ConstrainWrapper) = w.constraints

"""

    Return the immediate cost (vector) for the s-a pair

"""
function cost end

cost(m::ConstrainWrapper, s, a, sp) = cost(m, s, a)
cost(m::ConstrainWrapper, s, a, sp, o) = cost(m , s, a, sp)

####################

# Forward parts of POMDPs interface

####################
constraint_size(w::ConstrainWrapper) = length(w.constraints)
POMDPs.reward(w::ConstrainWrapper, ss::Tuple{<:Any,Int}, a, ssp::Tuple{<:Any,Int}) = reward(w.m, first(ss), a, first(ssp))
POMDPs.reward(w::ConstrainedPOMDPWrapper, ss, a, ssp, so) = reward(w.m, first(ss), a, first(ssp), first(so))
POMDPs.reward(w::ConstrainWrapper, ss::Tuple{<:Any,Int}, a) = reward(w.m, first(ss), a)
POMDPs.reward(w::ConstrainWrapper, ss, a) = reward(w.m, ss, a)
POMDPs.states(m::ConstrainWrapper) = states(m.m)
POMDPs.actions(w::ConstrainWrapper) = actions(w.m)
POMDPs.observations(w::ConstrainedPOMDPWrapper) = observations(w.m)
POMDPs.actionindex(w::ConstrainWrapper, a) = actionindex(w.m, a)
POMDPs.discount(w::ConstrainWrapper) = discount(w.m)
POMDPs.stateindex(w::ConstrainWrapper, s) = stateindex(w.m, s)
POMDPs.statetype(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = statetype(m.m)
POMDPs.actiontype(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = actiontype(m.m)
POMDPs.obstype(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = obstype(m.m)
POMDPs.initialstate(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = initialstate(m.m)
POMDPs.obsindex(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper, o) = obsindex(m.m, o)
POMDPs.transition(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper, s, a) = transition(m.m, s, a)
POMDPs.observation(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper, a, s) = observation(m.m, a, s)
POMDPs.isterminal(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper, s) = isterminal(m.m, s)
POMDPTools.ordered_states(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = ordered_states(m.m)
POMDPTools.ordered_actions(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = ordered_actions(m.m)
POMDPTools.ordered_observations(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = ordered_observations(m.m)
