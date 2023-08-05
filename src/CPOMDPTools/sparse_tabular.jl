struct TabularCPOMDP <: CPOMDP{Int,Int,Int}
    T::Vector{SparseMatrixCSC{Float64, Int64}} # T[a][sp, s]
    R::Array{Float64, 2} # R[s,a]
    O::Vector{SparseMatrixCSC{Float64, Int64}} # O[a][sp, o]
    C::Array{Float64, 3} # C[s,a,c_i]
    isterminal::BitVector
    initialstate::SparseVector{Float64, Int}
    constraints::Vector{Float64}
    discount::Float64
end

function TabularCPOMDP(pomdp::CPOMDP)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)

    T = POMDPTools.ModelTools.transition_matrix_a_s_sp(pomdp)
    R = _tabular_rewards(pomdp, S, A)
    O = POMDPTools.ModelTools.observation_matrix_a_sp_o(pomdp)
    C = _tabular_costs(pomdp, S, A)
    term = _vectorized_terminal(pomdp, S)
    b0 = _vectorized_initialstate(pomdp, S)
    return TabularCPOMDP(T,R,O,C,term,b0,pomdp.constraints,discount(pomdp))
end

function _tabular_rewards(pomdp, S, A)
    R = Matrix{Float64}(undef, length(S), length(A))
    for (s_idx, s) ∈ enumerate(S)
        if isterminal(pomdp, s)
            R[s_idx,:] .= 0.0
        else
            for (a_idx, a) ∈ enumerate(A)
                R[s_idx, a_idx] = reward(pomdp, s, a)
            end
        end
    end
    R
end

function _tabular_costs(pomdp, S, A)
    n_c = ConstrainedPOMDPs.constraint_size(pomdp)
    C = Array{Float64, 3}(undef, length(S), length(A), n_c)
    for (s_idx,s) ∈ enumerate(S)
        if isterminal(pomdp, s)
            C[s_idx,:,:] .= 0.0
        else
            for (a_idx,a) ∈ enumerate(A)
                C[s_idx, a_idx, :] .= costs(pomdp, s, a)
            end
        end
    end
    C
end

function _vectorized_terminal(pomdp, S)
    term = BitVector(undef, length(S))
    @inbounds for i ∈ eachindex(term,S)
        term[i] = isterminal(pomdp, S[i])
    end
    return term
end

function _vectorized_initialstate(pomdp, S)
    b0 = initialstate(pomdp)
    b0_vec = Vector{Float64}(undef, length(S))
    @inbounds for i ∈ eachindex(S, b0_vec)
        b0_vec[i] = pdf(b0, S[i])
    end
    return sparse(b0_vec)
end

POMDPTools.ordered_states(pomdp::TabularCPOMDP) = axes(pomdp.R, 1)
POMDPs.states(pomdp::TabularCPOMDP) = ordered_states(pomdp)
POMDPTools.ordered_actions(pomdp::TabularCPOMDP) = eachindex(pomdp.T)
POMDPs.actions(pomdp::TabularCPOMDP) = ordered_actions(pomdp)
POMDPTools.ordered_observations(pomdp::TabularCPOMDP) = axes(first(pomdp.O), 2)
POMDPs.observations(pomdp::TabularCPOMDP) = ordered_observations(pomdp)

POMDPs.stateindex(::TabularCPOMDP, s::Int) = s
POMDPs.actionindex(::TabularCPOMDP, a::Int) = a
POMDPs.obsindex(::TabularCPOMDP, o::Int) = o

POMDPs.discount(pomdp::TabularCPOMDP) = pomdp.discount

ConstrainedPOMDPs.constraint_size(pomdp::TabularCPOMDP) = size(pomdp.C, 3)

n_states(pomdp::TabularCPOMDP) = length(states(pomdp))
n_actions(pomdp::TabularCPOMDP) = length(actions(pomdp))
n_observations(pomdp::TabularCPOMDP) = length(observations(pomdp))
n_constraints(pomdp::TabularCPOMDP) = size(pomdp.C, 3)
const n_cost = n_constraints

POMDPs.initialstate(p::TabularCPOMDP) = SparseCat(ordered_states(p), p.initialstate)
POMDPs.transition(p::TabularCPOMDP, s, a) = SparseCat(ordered_states(p), p.T[a][:,s])
POMDPs.observation(p::TabularCPOMDP, a, sp) = SparseCat(ordered_observations(p), p.O[a][sp,:])
POMDPs.reward(p::TabularCPOMDP, s, a) = p.R[s,a]
ConstrainedPOMDPs.costs(p::TabularCPOMDP, s, a) = p.C[s,a,:]
ConstrainedPOMDPs.constraints(p::TabularCPOMDP) = p.constraints
