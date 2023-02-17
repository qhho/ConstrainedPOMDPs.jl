"""
    Constrain(m::Union{MDP, POMDP}, cost_constraints::Vector{Float64})

"""
Constrain(m::MDP, constraints::Vector{Float64}) = ConstrainedMDPWrapper(m, constraints)
Constrain(m::POMDP, constraints::Vector{Float64}) = ConstrainedPOMDPWrapper(m, constraints)


############

# Wrapper types

############

struct ConstrainedMDPWrapper{S,A, M<:MDP} <: MDP{S, A}
    m::M
    constraints::Vector{Float64}
end

function ConstrainedMDPWrapper(m::MDP{S,A}, c::Vector{Float64}) where {S,A}
    return ConstrainedMDPWrapper{S,A,typeof(m)}(m, c)
end

struct ConstrainedPOMDPWrapper{S,A,O,M<:POMDP} <: POMDP{S, A, O}
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

constraints(w::ConstrainWrapper) = w.constraints

"""

    Return the immediate cost (vector) for the s-a pair

"""
function cost end

cost(m::ConstrainWrapper, s, a, sp) = cost(m, s, a)
cost(m::ConstrainWrapper, s, a, sp, o) = cost(m, s, a, sp)

####################

# Forward parts of POMDPs interface

####################
constraint_size(w::ConstrainWrapper)            = length(w.constraints)
POMDPs.reward(w::CPOMDP, s, a, sp, o)           = reward(w.m, s, a, sp, o)
POMDPs.reward(w::ConstrainWrapper, s, a, sp)    = reward(w.m, s, a, sp)
POMDPs.reward(w::ConstrainWrapper, s, a)        = reward(w.m, s, a)
POMDPs.states(m::ConstrainWrapper)              = states(m.m)
POMDPs.actions(w::ConstrainWrapper)             = actions(w.m)
POMDPs.observations(w::CPOMDP)                  = observations(w.m)
POMDPs.stateindex(w::ConstrainWrapper, s)       = stateindex(w.m, s)
POMDPs.actionindex(w::ConstrainWrapper, a)      = actionindex(w.m, a)
POMDPs.obsindex(m::CPOMDP, o)                   = obsindex(m.m, o)
POMDPs.discount(w::ConstrainWrapper)            = discount(w.m)
POMDPs.initialstate(m::ConstrainWrapper)        = initialstate(m.m)
POMDPs.transition(m::ConstrainWrapper, s, a)    = transition(m.m, s, a)
POMDPs.observation(m::CPOMDP, a, s)             = observation(m.m, a, s)
POMDPs.isterminal(m::ConstrainWrapper, s)       = isterminal(m.m, s)
POMDPTools.ordered_states(m::ConstrainWrapper)  = ordered_states(m.m)
POMDPTools.ordered_actions(m::ConstrainWrapper) = ordered_actions(m.m)
POMDPTools.ordered_observations(m::CPOMDP)      = ordered_observations(m.m)
