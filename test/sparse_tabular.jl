@testset begin "sparse_tabular"
    cpomdp = ConstrainedPOMDPs.constrain(TigerPOMDP(),[1.0]) do s,a
        iszero(a) ? [1.0] : [0.0]
    end 
    tcpomdp = TabularCPOMDP(cpomdp)

    @test has_consistent_distributions(tcpomdp)
end
