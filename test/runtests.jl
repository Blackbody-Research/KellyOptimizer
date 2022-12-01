using KellyOptimizer, Statistics, StatsBase, LinearAlgebra

n = 10

#create mean daily returns for n instruments
M = randn(n)

#create std for daily returns with a range from 1 to 2 divided by 10
σs = (1 .+ rand(n)) ./ 10
#create simple covariance matrix assuming zero correlation
Σ = diagm(σs .^2)
#set daily mean interest rate at 4%
rint = 0.04/252

portfolio = optimize_portfolio(M, Σ, rint)
fopt = getexposures(portfolio)

println("Optimized exposures are as follows: $fopt")

