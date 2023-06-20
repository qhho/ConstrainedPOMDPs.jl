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
    ConstrainedMDPWrapper,
    ConstrainedPOMDPWrapper,
    ConstrainedWrapper,

    # Additional Model Functions
    cost,
    constraints,
    Constrain,
    constraint_size

include("core.jl")
include("gen_impl.jl")
include("rollout.jl")

end
