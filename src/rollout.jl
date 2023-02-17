POMDPs.action(policy, b, d) = action(policy, b)

function POMDPs.simulate(sim::RolloutSimulator, pomdp::CPOMDP{S}, policy::Policy, updater::Updater, b0, s::S) where S
    ϵ = isnothing(sim.eps) ? 0.0 : sim.eps
    max_steps = isnothing(sim.max_steps) ? typemax(Int) : sim.max_steps

    γ = discount(pomdp)
    γt = 1.0
    r_total = 0.0
    c_total = zeros(constraint_size(pomdp))
    d = copy(constraints(pomdp))
    b = initialize_belief(updater, b0)
    step = 1

    while γt > ϵ && !isterminal(pomdp, s) && step ≤ max_steps
        a = action(policy, b, d)
        sp, o, r, c = @gen(:sp,:o,:r,:c)(pomdp, s, a, sim.rng)
        r_total += γt*r
        @. c_total += γt*c
        @. d = (1/γ)*(d - c)
        s = sp
        bp = update(updater, b, a, o)
        b = bp
        γt *= γ
        step += 1
    end

    return (r_total, c_total)
end

function POMDPs.simulate(sim::RolloutSimulator, mdp::CMDP{S}, policy::Policy, s::S) where S
    ϵ = isnothing(sim.eps) ? 0.0 : sim.eps
    max_steps = isnothing(sim.max_steps) ? typemax(Int) : sim.max_steps

    γ = discount(mdp)
    γt = 1.0
    r_total = 0.0
    c_total = zeros(constraint_size(mdp))
    d = copy(constraints(mdp))
    step = 1

    while γt > ϵ && !isterminal(mdp, s) && step ≤ max_steps
        a = action(policy, s, d)
        sp, r, c = @gen(:sp,:r,:c)(mdp, s, a, sim.rng)
        r_total += γt*r
        @. c_total += γt*c
        @. d = (1/γ)*(d - c)
        s = sp
        γt *= γ
        step += 1
    end

    return (r_total, c_total)
end
