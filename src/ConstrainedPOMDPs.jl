"""
Provides a basic interface for defining and solving constrained MDPs and POMDPs
"""
module ConstrainedPOMDPs

using POMDPs
using POMDPModelTools
using POMDPSimulators
using Random: Random, AbstractRNG

export
    # Main Type
    ConstrainedMDPWrapper,
    ConstrainedPOMDPWrapper,
    ConstrainedWrapper,

    # Additional Model Functions
    cost,
    constraints,
    Constrain,
    simulate,
    RolloutSimulator

include("core.jl")
include("rollout.jl")

end