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
    λ::Float64
    initialized::Bool
end


ConstrainedMDPWrapper(m::MDP, constraints::Vector{Float64}) = ConstrainedMDPWrapper{statetype(m), actiontype(m), typeof(m)}(m, constraints)

struct ConstrainedPOMDPWrapper{S,A,O,M<:POMDP} <: POMDP{Tuple{S, Int}, A, Tuple{O,Int}}
    m::M
    constraints::Vector{Float64}
    λ::Float64
    initialized::Bool
end

ConstrainedPOMDPWrapper(m::POMDP, constraints::Vector{Float64}) = ConstrainedPOMDPWrapper{statetype(m), actiontype(m), obstype(m), typeof(m)}(m, constraints::Vector{Float64}, 0.0, false)

const ConstrainWrapper = Union{ConstrainedMDPWrapper, ConstrainedPOMDPWrapper}

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

cost(m::Union{POMDP,MDP}, s, a, sp) = cost(m, s, a)
cost(m::Union{POMDP,MDP}, s, a, sp, o) = cost(m , s, a, sp)

"""
    POMDPs.gen(w::ConstrainWrapper, ss::Tuple{<:Any,Int}, a, rng::AbstractRNG)
Implement the entire MDP/POMDP generative model by returning a NamedTuple.
"""

function POMDPs.gen(w::ConstrainWrapper, ss::Tuple{<:Any,Int}, a, rng::AbstractRNG)
    out = gen(w.m, first(ss), a, rng)
    if haskey(out, :sp)
        return merge(out, (sp=(out.sp, stage(w, ss)+1),))
    else
        return out
    end
end

"""
    c_gen(cpomdp, s, a, rng)
Implement the entire constrained MDP/POMDP generative model by returning a tuple of sp, o, r, c
"""

function gen(cpomdp::ConstrainWrapper, s, a, rng)
    sp, o, r = @gen(:sp, :o, :r)(cpomdp.m, s, a, rng)
    c = cost(cpomdp.m, s, a, sp, o)
    return (sp,o,r,c)
end

####################

# Forward parts of POMDPs interface

####################

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
POMDPs.states(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = states(m.m)
POMDPs.initialstate(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = initialstate(m.m)
POMDPs.obsindex(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper, o) = obsindex(m.m, o)
POMDPs.transition(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper, s, a) = transition(m.m, s, a)
POMDPs.observation(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper, a, s) = observation(m.m, a, s)
POMDPModelTools.ordered_states(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = ordered_states(m.m)
POMDPModelTools.ordered_actions(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = ordered_actions(m.m)
POMDPModelTools.ordered_observations(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper) = ordered_observations(m.m)