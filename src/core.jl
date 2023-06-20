abstract type CPOMDP{S,A,O} <: POMDP{S,A,O} end
abstract type CMDP{S,A} <: MDP{S,A} end

const ConstrainedProblem = Union{CMDP, CPOMDP}

"""
    Return the immediate cost (vector) for the s-a pair
"""
function cost end

"""
    Return the constraints
"""
function constraints end

cost(m::ConstrainedProblem, s, a, sp) = cost(m, s, a)
cost(m::ConstrainedProblem, s, a, sp, o) = cost(m, s, a, sp)

"""
    Constrain(cost::Function, m::Union{MDP, POMDP}, cost_constraints::Vector{Float64})
"""
constrain(cost::Function, m::MDP, constraints::Vector{Float64}) = CMDPWrapper(cost, m, constraints)
constrain(cost::Function, m::POMDP, constraints::Vector{Float64}) = CPOMDPWrapper(cost, m, constraints)

struct CMDPWrapper{S,A,M<:MDP,F} <: CMDP{S, A}
    cost::F
    m::M
    constraints::Vector{Float64}
end

function CMDPWrapper(cost::F, m::MDP{S,A}, c::Vector{Float64}) where {S,A,F}
    return CMDPWrapper{S,A,typeof(m),F}(cost, m, c)
end

struct CPOMDPWrapper{S,A,O,M<:POMDP,F} <: CPOMDP{S, A, O}
    cost::F
    m::M
    constraints::Vector{Float64}
end

function CPOMDPWrapper(cost::F, m::POMDP{S,A,O}, c::Vector{Float64}) where {S,A,O,F}
    return CPOMDPWrapper{S,A,O,typeof(m), F}(cost, m, c)
end

const CMDPW = CMDPWrapper
const CPOMDPW = CPOMDPWrapper
const ConstrainWrapper = Union{CMDPW, CPOMDPW}

constraints(w::ConstrainWrapper) = w.constraints

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
