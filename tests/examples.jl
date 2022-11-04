using Test
using POMDPs
using ConstrainedPOMDPs
import POMDPModels: SimpleGridWorld, BabyPOMDP
using CCPOMCP
using CGCP

bb = BabyPOMDP()
constraints = [1.0]
c_bb = ConstrainedPOMDPs.Constrain(bb,constraints)

function ConstrainedPOMDPs.cost(c_bb,s,a)
    return [1.0]
end

@test c_bb.constraints == constraints
@test ConstrainedPOMDPs.constraints(c_bb) == constraints
@test ConstrainedPOMDPs.cost(c_bb, true, true) == [1]

solver = CCPOMCPSolver()
planner = solve(solver,c_bb)

solver2 = CGCPSolver()
solution = solve(solver2,c_bb)
