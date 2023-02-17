function POMDPs.simulate(sim::RolloutSimulator, pomdp::CPOMDP, policy::Policy, updater::Updater, initial_belief, s)
    eps = isnothing(sim.eps) ? 0.0 : sim.eps
    max_steps = isnothing(sim.max_steps) ? typemax(Int) : sim.max_steps

    disc = 1.0
    r_total = 0.0
    c_total = zeros(constraint_size(pomdp))
    b = initialize_belief(updater, initial_belief)
    step = 1

    while disc > eps && !isterminal(pomdp, s) && step ≤ max_steps
        a = action(policy, b)
        sp, o, r, c = @gen(:sp,:o,:r,:c)(pomdp, s, a, sim.rng)
        r_total += disc*r
        c_total += disc*c
        s = sp
        bp = update(updater, b, a, o)
        b = bp
        disc *= discount(pomdp)
        step += 1
    end

    return (r_total, c_total)
end
