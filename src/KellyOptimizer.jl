module KellyOptimizer

#functions to perform kelly portfolio optimization according to Practical Implementation of the Kelly Criterion: Optimal Growth Rate, Number of Trades, and Rebalancing Frequency for Equity Portfolios found at: https://www.frontiersin.org/articles/10.3389/fams.2020.577050/full

#also provides helper functions to generate and merge price series for calculating statistics

using JuMP, HiGHS, LinearAlgebra, Statistics, StatsBase, DataFrames

#helper function to add a constraint to the optimization based on the absolute value sum of a vector
function sumAbs(model, expr::Vector)
    slack = @variable(model, [1:length(expr)], lower_bound = 0.0)
    @constraint(model, slack .>= expr)
    @constraint(model, slack .>= -expr)
    return @expression(model, sum(slack))
end

"""
    optimize_portfolio(M, Σ; allowshort = false, lmax = Inf)

Given a vector of means returns (M) and covariances (Σ) optimize a model of expected rebalance returns with the optimal exposure fractions for each instrument.
Additional contraints that can be provided are allowing shorting and setting a maximum limit for total leverage.  
"""
function optimize_portfolio(M, Σ, rint; allowshort = false, lmax = Inf, modval = 252)
    l = length(M)
    (m, n) = size(Σ)
    #since Σ is the covariance matrix is must be square
    @assert m == n
    #the covariance matrix must match the size of the mean returns
    @assert l == m

    portfoliomodel = Model(HiGHS.Optimizer)
    if allowshort
        @variable(portfoliomodel, f[1:length(M)])
    else
        @variable(portfoliomodel, f[1:length(M)] >= 0)
    end
    z = sumAbs(portfoliomodel, f)
    if !isinf(lmax)
        @constraint(portfoliomodel, z <= lmax)
    end
    @objective(portfoliomodel, Max, (modval .*rint) .+ (f' * (modval .*(M .- rint))) - ((f' * (modval .*Σ) * f)/2))
    optimize!(portfoliomodel)
    return portfoliomodel
end

"""
    getexposures(portfolio)

Given an optimized portfolio model, extract the optimal exposures to each instrument
"""
getexposures(portfolio) = value.(portfolio.obj_dict[:f])

export optimize_portfolio, getexposures

end # module KellyOptimizer
