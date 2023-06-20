"""
Provides a basic interface for defining and solving constrained MDPs and POMDPs
"""
module ConstrainedPOMDPs

using POMDPs
using POMDPTools
using Tricks
using Random: Random, AbstractRNG

export
    # Main Type
    CMDPWrapper,
    CPOMDPWrapper,
    ConstrainedWrapper,

    # Additional Model Functions
    cost,
    constraints,
    constrain,
    constraint_size

include("core.jl")
include("gen_impl.jl")
include("rollout.jl")

end
