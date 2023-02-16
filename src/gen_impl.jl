# Do we really need to rewrite this?
# Can't we just dispatch on internals below?
@generated function POMDPs.genout(v::POMDPs.DDNOut{symbols}, m::ConstrainWrapper, s, a, rng) where symbols

    # use anything available from gen(m, s, a, rng)
    expr = quote
        x = gen(m, s, a, rng)
        @assert x isa NamedTuple "gen(m::ConstrainWrapper, ...) must return a NamedTuple; got a $(typeof(x))"
    end

    # add gen for any other variables
    for (var, depargs) in POMDPs.sorted_deppairs(m, symbols)
        if var in (:s, :a) # input nodes
            continue
        end

        sym = Meta.quot(var)

        varblock = quote
            if haskey(x, $sym) # should be constant at compile time
                $var = x[$sym]
            else
                $var = $(POMDPs.node_expr(Val(var), depargs))
            end
        end
        append!(expr.args, varblock.args)
    end

    # add return expression
    return_expr = if symbols isa Tuple
        :(return $(Expr(:tuple, symbols...)))
    else
        :(return $symbols)
    end
    append!(expr.args, return_expr.args)

    return expr
end

function POMDPs.sorted_deppairs(m::Type{<:CMDP}, symbols)
    deps = Dict(:s => Symbol[],
                :a => Symbol[],
                :sp => [:s, :a],
                :r => [:s, :a, :sp],
                :c => [:s, :a, :sp],
                :info => Symbol[]
               )
    return POMDPs.sorted_deppairs(deps, symbols)
end

function POMDPs.sorted_deppairs(m::Type{<:CPOMDP}, symbols)
    deps = Dict(:s => Symbol[],
                :a => Symbol[],
                :sp => [:s, :a],
                :o => [:s, :a, :sp],
                :r => [:s, :a, :sp, :o],
                :c => [:s, :a, :sp, :o],
                :info => Symbol[]
               )
    return POMDPs.sorted_deppairs(deps, symbols)
end

POMDPs.node_expr(::Val{:c}, depargs) = :(cost(m, $(depargs...)))
