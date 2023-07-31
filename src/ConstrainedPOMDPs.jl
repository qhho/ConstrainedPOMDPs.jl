"""
Provides a basic interface for defining and solving constrained MDPs and POMDPs
"""
module ConstrainedPOMDPs

using POMDPs
using POMDPTools
using Tricks
using Lazy
using Random: Random, AbstractRNG

export
    @MDP_forward,
    @POMDP_forward,
    CMDP,
    CPOMDP,
    CMDPWrapper,
    CPOMDPWrapper,
    ConstrainedWrapper,

    # Additional Model Functions
    costs,
    constraints,
    constrain,
    constraint_size,
    ⪯, ≺, ⪰, ≻

include("forward.jl")
include("core.jl")
include("gen_impl.jl")
include("rollout.jl")
include("underlying.jl")

end
