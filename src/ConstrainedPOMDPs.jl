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
    CMDP,
    CPOMDP,
    CMDPWrapper,
    CPOMDPWrapper,
    ConstrainedWrapper,

    # Additional Model Functions
    costs,
    constraints,
    constrain,
    constraint_size

include("core.jl")
include("gen_impl.jl")
include("rollout.jl")
include("underlying.jl")

end
