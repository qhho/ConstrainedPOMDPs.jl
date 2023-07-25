# ConstrainedPOMDPs.jl
POMDPs.jl-compatible interface for defining Constrained MDPs and POMDPs

This package aims to provide a standard interface for defining constrained MDP and POMDP problems.

The goals are to

1. Express problems as constrained MDPs and POMDPs.
2. Interface easily with constrained POMDP solvers.

## Usage
```julia
using ConstrainedPOMDPs
using POMDPModels
using POMDPs

pomdp = TigerPOMDP()
cpomdp = constrain(pomdp, [1.0]) do s,a
    iszero(a) ? [0.5] : 0.0
end

s = false
a = 0
@show costs(cpomdp, s, a)
sp, o, r, c = @gen(:sp, :o, :r, :c)(cpomdp, s, a)
@show sp, o, r, c
```

```julia
costs(cpomdp, s, a) = [0.5]

(sp, o, r, c) = (false, false, -1.0, [0.5])
```
