"""
Provides a basic interface for defining and solving constrained MDPs and POMDPs
"""
module ConstrainedPOMDPs

using POMDPs
using POMDPModelTools
using POMDPSimulators
using Random: Random, AbstractRNG

export
    # Abtract Type
    ConstrainedMDPWrapper,
    ConstrainedPOMDPWrapper,
    ConstrainWrapper,

    #Model Functions from POMDPs.jl
    reward,
    states,
    actions,
    observations,
    actionindex,
    discount,
    stateindex,
    statetype,
    actiontype,
    states,
    initialstate,
    obsindex,
    transition,
    observation,
    ordered_states,
    ordered_actions,
    ordered_observations,

    # Additional Model Functions
    cost,
    constraints,
    Constrain,
    simulate,
    RolloutSimulator


include("core.jl")
include("rollout.jl")
# include("ConstrainedRockSample.jl")

end