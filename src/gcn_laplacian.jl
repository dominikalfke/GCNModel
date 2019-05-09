
using LinearAlgebra
import Arpack

export
	applySmoother,
    getFullLaplacian,
    getLaplacianEigenvalues

"""
	applySmoother(smoother, adjacency :: AbstractMatrix{Float64}, graph :: AbstractGraph)

Apply a smoother object to a given adjacency matrix. Returns the smoothed
adjacency matrix. Most smoothers work by adding terms to the diagonal of the
adjacency matrix. The underlying graph is passed as the third argument, in case
the smoothing depends on its properties. Custom smoother types must implement
this method. Default smoothers can be of type `Nothing`, `Float64`, or
`Vector{Float64}`.
"""
applySmoother(:: Nothing, adjacency :: AbstractMatrix{Float64}, graph :: AbstractGraph) =
	adjacency
applySmoother(α :: Float64, adjacency :: AbstractMatrix{Float64}, graph :: AbstractGraph) =
	α == 0.0 ? adjacency : adjacency + UniformScaling(α)
applySmoother(s :: Vector{Float64}, adjacency :: AbstractMatrix{Float64}, graph :: AbstractGraph) =
	adjacency + Diagonal(s)

"""
	getFullLaplacian(graph :: AbstractGraph)
	getFullLaplacian(graph :: AbstractGraph, smoother)

Compute the full graph Laplacian matrix of a graph. For most graph types, this
is the matrix `I - D^{-1} A D^{-1}`, where `A` is the graph's (possibly
smoothed) adjacency matrix and `D` is the diagonal matrix holding `A`'s row
sums. `AbstractGraph` graph types with other Laplacian definitions or more
efficient computation schemes must override this method.
"""
function getFullLaplacian(graph :: AbstractGraph, smoother = nothing)

    smoothedAdj = applySmoother(smoother, getAdjacency(graph), graph)

    degrees = sum(smoothedAdj, dims=1)[:]
    Dinvsqrt = Diagonal(1 ./ sqrt.(degrees))

    return UniformScaling(1.0) - Symmetric(Dinvsqrt * smoothedAdj * Dinvsqrt)
end



### getLaplacianEigenvalues
"""
	getLaplacianEigenvalues(graph :: AbstractGraph, numEV :: Int64, whichEV :: Symbol)
	getLaplacianEigenvalues(graph :: AbstractGraph, numEV :: Int64, whichEV :: Symbol, smoother)

Compute a small number of eigenvalues of the graph Laplacian matrix. Calls the
`eigs` function from `Arpack` on the result of `getFullLaplacian` with an
optional `smoother`. The type of eigenvalues is determined by the `whichEV`
argument, which can be one of the following symbols:
`:small` - compute the smallest eigenvalues.
`:large` - compute the largest eigenvalues.
`:smallnonzero` - compute the smallest eigenvalues, skipping the first
	eigenvalue, which is assumed to be zero.
"""
getLaplacianEigenvalues(graph :: AbstractGraph, numEV :: Int64, whichEV :: Symbol, smoother = nothing) =
	_getLaplacianEigenvalues(graph, numEV, whichEV, smoother)

function _getLaplacianEigenvalues(graph :: AbstractGraph, numEV :: Int64, whichEV :: Symbol, smoother)
    L = getFullLaplacian(graph, smoother)
    if whichEV == :small
        λ, U = Arpack.eigs(2*UniformScaling(1.0) - L, nev=numEV, which=:LM)
        λ = 2 .- λ
    elseif whichEV == :smallnonzero
        λ, U = Arpack.eigs(2*UniformScaling(1.0) - L, nev=numEV+1, which=:LM)
        λ = 2 .- λ[2:end]
        U = U[:, 2:end]
    elseif whichEV == :large
        λ, U = Arpack.eigs(L, nev=numEV, which=:LM)
    else
        throw(ArgumentError("Unknown eigenvalue specifier: $whichEV"))
    end
    return λ, U
end
