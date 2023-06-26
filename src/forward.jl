macro MDP_forward(ex)
    ret_expr = quote
        ConstrainedPOMDPs.Lazy.@forward($ex, (
            POMDPs.transition, 
            POMDPs.reward,
            POMDPs.discount,
            POMDPs.actionindex,
            POMDPs.stateindex,
            POMDPs.states,
            POMDPs.actions,
            POMDPs.initialstate,
            POMDPTools.ordered_states,
            POMDPTools.ordered_actions,
            POMDPs.isterminal))
    end
    return esc(ret_expr)
end

macro POMDP_forward(ex)
    ret_expr = quote
        ConstrainedPOMDPs.Lazy.@forward($ex, (
            POMDPs.transition,
            POMDPs.observation, 
            POMDPs.reward,
            POMDPs.discount,
            POMDPs.stateindex,
            POMDPs.actionindex,
            POMDPs.obsindex,
            POMDPs.states,
            POMDPs.actions,
            POMDPs.observations,
            POMDPs.initialstate,
            POMDPTools.ordered_states,
            POMDPTools.ordered_actions,
            POMDPTools.ordered_observations,
            POMDPs.isterminal))
    end
    return esc(ret_expr)
end
