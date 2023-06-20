abstract type CPOMDP{S,A,O} <: POMDP{S,A,O} end
abstract type CMDP{S,A} <: MDP{S,A} end

const ConstrainedProblem = Union{CMDP, CPOMDP}

"""
    Constrain(m::Union{MDP, POMDP}, cost_constraints::Vector{Float64})

"""
Constrain(m::MDP, constraints::Vector{Float64}) = ConstrainedMDPWrapper(m, constraints)
Constrain(m::POMDP, constraints::Vector{Float64}) = ConstrainedPOMDPWrapper(m, constraints)

############

# Wrapper types

############

struct ConstrainedMDPWrapper{S,A, M<:MDP, F} <: CMDP{S, A}
    cost::F
    m::M
    constraints::Vector{Float64}
end

function ConstrainedMDPWrapper(m::MDP{S,A}, c::Vector{Float64}) where {S,A}
    return ConstrainedMDPWrapper{S,A,typeof(m), typeof(cost)}(cost, m, c)
end

struct ConstrainedPOMDPWrapper{S,A,O,M<:POMDP, F} <: CPOMDP{S, A, O}
    cost::F
    m::M
    constraints::Vector{Float64}
end

function ConstrainedPOMDPWrapper(m::POMDP{S,A,O}, c::Vector{Float64}, cost) where {S,A,O}
    return ConstrainedPOMDPWrapper{S,A,O,typeof(m), typeof(cost)}(cost, m, c)
end

const CMDPW = ConstrainedMDPWrapper
const CPOMDPW = ConstrainedPOMDPWrapper
const ConstrainWrapper = Union{CMDPW, CPOMDPW}

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

# cost(m::ConstrainWrapper, s, a, sp) = cost(m, s, a)
# cost(m::ConstrainWrapper, s, a, sp, o) = cost(m, s, a, sp)

function cost(m::ConstrainWrapper, args...)
    r = m.cost
    if static_hasmethod(r, typeof(args)) # static_hasmethod could cause issues, but I think it is worth doing in this single spot
        return r(args...)
    elseif m isa POMDP && length(args) == 4
        if static_hasmethod(r, typeof(args[1:3])) # (s, a, sp, o) -> (s, a, sp)
            return r(args[1:3]...)
        elseif static_hasmethod(r, typeof(args[1:2])) # (s, a, sp, o) -> (s, a)
            return r(args[1:2]...)
        end
    elseif length(args) == 3 && static_hasmethod(r, typeof(args[1:2])) # (s, a, sp) -> (s, a)
        return r(args[1:2]...)
    else
        return r(args...)
    end
end

####################

# Forward parts of POMDPs interface

####################
constraint_size(w::ConstrainWrapper)            = length(w.constraints)
POMDPs.reward(w::CPOMDPW, s, a, sp, o)           = reward(w.m, s, a, sp, o)
POMDPs.reward(w::ConstrainWrapper, s, a, sp)    = reward(w.m, s, a, sp)
POMDPs.reward(w::ConstrainWrapper, s, a)        = reward(w.m, s, a)
POMDPs.states(m::ConstrainWrapper)              = states(m.m)
POMDPs.actions(w::ConstrainWrapper)             = actions(w.m)
POMDPs.observations(w::CPOMDPW)                  = observations(w.m)
POMDPs.stateindex(w::ConstrainWrapper, s)       = stateindex(w.m, s)
POMDPs.actionindex(w::ConstrainWrapper, a)      = actionindex(w.m, a)
POMDPs.obsindex(m::CPOMDPW, o)                   = obsindex(m.m, o)
POMDPs.discount(w::ConstrainWrapper)            = discount(w.m)
POMDPs.initialstate(m::ConstrainWrapper)        = initialstate(m.m)
POMDPs.transition(m::ConstrainWrapper, s, a)    = transition(m.m, s, a)
POMDPs.observation(m::CPOMDPW, a, s)             = observation(m.m, a, s)
POMDPs.isterminal(m::ConstrainWrapper, s)       = isterminal(m.m, s)
POMDPTools.ordered_states(m::ConstrainWrapper)  = ordered_states(m.m)
POMDPTools.ordered_actions(m::ConstrainWrapper) = ordered_actions(m.m)
POMDPTools.ordered_observations(m::CPOMDPW)      = ordered_observations(m.m)
