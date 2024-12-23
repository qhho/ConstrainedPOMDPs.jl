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

    T = transition_matrix_a_sp_s(pomdp)
    R = _tabular_rewards(pomdp, S, A)
    O = POMDPTools.ModelTools.observation_matrix_a_sp_o(pomdp)
    C = _tabular_costs(pomdp, S, A)
    term = _vectorized_terminal(pomdp, S)
    b0 = _vectorized_initialstate(pomdp, S)
    return TabularCPOMDP(T,R,O,C,term,b0,constraints(pomdp),discount(pomdp))
end

struct TabularCMDP <: CMDP{Int,Int}
    T::Vector{SparseMatrixCSC{Float64, Int64}} # T[a][sp, s]
    R::Array{Float64, 2} # R[s,a]
    C::Array{Float64, 3} # C[s,a,c_i]
    isterminal::BitVector
    initialstate::SparseVector{Float64, Int}
    constraints::Vector{Float64}
    discount::Float64
end

function TabularCMDP(m::CMDP)
    S = ordered_states(m)
    A = ordered_actions(m)

    T = transition_matrix_a_sp_s(m)
    R = _tabular_rewards(m, S, A)
    C = _tabular_costs(m, S, A)
    term = _vectorized_terminal(m, S)
    b0 = _vectorized_initialstate(m, S)
    return TabularCMDP(T, R, C, term, b0, constraints(m), discount(m))
end

function transition_matrix_a_sp_s(mdp::Union{MDP, POMDP})
    S = ordered_states(mdp)
    A = ordered_actions(mdp)

    ns = length(S)
    na = length(A)
    
    transmat_row_A = [Int64[] for _ in 1:na]
    transmat_col_A = [Int64[] for _ in 1:na]
    transmat_data_A = [Float64[] for _ in 1:na]

    for (si,s) in enumerate(S)
        for (ai,a) in enumerate(A)
            if isterminal(mdp, s) # if terminal, there is a probability of 1 of staying in that state
                push!(transmat_row_A[ai], si)
                push!(transmat_col_A[ai], si)
                push!(transmat_data_A[ai], 1.0)
            else
                td = transition(mdp, s, a)
                for (sp, p) in weighted_iterator(td)
                    if p > 0.0
                        spi = stateindex(mdp, sp)
                        push!(transmat_row_A[ai], spi)
                        push!(transmat_col_A[ai], si)
                        push!(transmat_data_A[ai], p)
                    end
                end
            end
        end
    end
    transmats_A_SP_S = [sparse(transmat_row_A[a], transmat_col_A[a], transmat_data_A[a], ns, ns) for a in 1:na]
    return transmats_A_SP_S
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

const TabularConstrainedProblem = Union{TabularCMDP, TabularCPOMDP}

POMDPTools.ordered_states(pomdp::TabularConstrainedProblem) = axes(pomdp.R, 1)
POMDPs.states(pomdp::TabularConstrainedProblem) = ordered_states(pomdp)
POMDPTools.ordered_actions(pomdp::TabularConstrainedProblem) = eachindex(pomdp.T)
POMDPs.actions(pomdp::TabularConstrainedProblem) = ordered_actions(pomdp)
POMDPTools.ordered_observations(pomdp::TabularCPOMDP) = axes(first(pomdp.O), 2)
POMDPs.observations(pomdp::TabularCPOMDP) = ordered_observations(pomdp)

POMDPs.stateindex(::TabularConstrainedProblem, s::Int) = s
POMDPs.actionindex(::TabularConstrainedProblem, a::Int) = a
POMDPs.obsindex(::TabularCPOMDP, o::Int) = o

POMDPs.discount(pomdp::TabularConstrainedProblem) = pomdp.discount

ConstrainedPOMDPs.constraint_size(pomdp::TabularConstrainedProblem) = size(pomdp.C, 3)

n_states(pomdp::TabularConstrainedProblem) = length(states(pomdp))
n_actions(pomdp::TabularConstrainedProblem) = length(actions(pomdp))
n_observations(pomdp::TabularCPOMDP) = length(observations(pomdp))
n_constraints(pomdp::TabularConstrainedProblem) = size(pomdp.C, 3)
const n_cost = n_constraints

POMDPs.initialstate(p::TabularConstrainedProblem) = SparseCat(ordered_states(p), p.initialstate)
POMDPs.transition(p::TabularConstrainedProblem, s, a) = SparseCat(ordered_states(p), p.T[a][:,s])
POMDPs.observation(p::TabularCPOMDP, a, sp) = SparseCat(ordered_observations(p), p.O[a][sp,:])
POMDPs.reward(p::TabularConstrainedProblem, s, a) = p.R[s,a]
ConstrainedPOMDPs.costs(p::TabularConstrainedProblem, s, a) = p.C[s,a,:]
ConstrainedPOMDPs.constraints(p::TabularConstrainedProblem) = p.constraints
