

using SparseArrays


export
	AbstractGraph,
	getNumNodes,
	getNumEdges,
	getDenseAdjacency,
	getSparseAdjacency,
	getAdjacency,
	isSparse,
	getSubGraph,
    AdjacencyGraph,
	SparseAdjacencyGraph,
	EdgeListGraph,
	FullGaussianGraph


"""
	AbstractGraph

Abstract supertype for all graph objects.
"""
abstract type AbstractGraph end

"""
	getNumNodes(graph :: AbstractGraph)

Returns the number of nodes in a graph. Each `AbstractGraph` subtype must
override this method.
"""
getNumNodes( :: AbstractGraph) = 0

"""
	getNumEdges(graph :: AbstractGraph)

Returns the number of edges (of some kind) in a graph.
"""
getNumEdges( :: AbstractGraph) = 0

"""
	getDenseAdjacency(g :: AbstractGraph)

Returns the dense adjacency matrix of a graph. By default, the result of
`getSparseAdjacency` is converted to non-sparse format. Each `ÁbstractGraph`
subtype must override either this method or `getSparseAdjacency`.
"""
getDenseAdjacency(g :: AbstractGraph) = full(getSparseAdjacency(g))

"""
	getSparseAdjacency(g :: AbstractGraph)

Returns the adjacency matrix of a graph in sparse storage (whether this makes
sense or not). By default, the result of `getDenseAdjacency` is converted to
sparse format. Each `AbstractGraph` subtype must override either this method
or `getSparseAdjacency`.
"""
getSparseAdjacency(g :: AbstractGraph) = sparse(getDenseAdjacency(g))

"""
	isSparse(g :: AbstractGraph)

Returns true if the graph supports a sparse format for its adjacency matrix.
This determines whether `getAdjacency` will call `getDenseAdjacency` or
`getSparseAdjacency`. Default is true. Sparse subtypes of `AbstractGraph` must
override this method to return true.
"""
isSparse(:: AbstractGraph) = false

"""
	getAdjacency(g :: AbstractGraph)

Calls either `getDenseAdjacency(g)` or `getSparseAdjacency(g)`, based on the
result of `isSparse(g)`. `AbstractGraph` subtypes must not override this
method.
"""
getAdjacency(g :: AbstractGraph) =
	isSparse(g) ? getSparseAdjacency(g) : getDenseAdjacency(g)

"""
	getSubGraph(g :: AbstractGraph, ind :: Vector{Int64})

Returns a new graph object with a subset of nodes, given by `ind`, and
the same adjacency information. `AbstractGraph` subtypes must override this
method if they are to support `largestConnectedSubDataset` behaviour.
"""
getSubGraph(g :: AbstractGraph, ind :: Vector{Int64}) = g

Base.show(io :: IO, g :: AbstractGraph) =
	print(io, "$(typeof(g))($(getNumNodes(g)) nodes, $(getNumEdges(g)) edges)")


### Type for graphs given directly by their adjacency matrix

"""
	AdjacencyGraph

Subtype of `AbstractGraph` for graphs given by their dense adjacency matrix
"""
mutable struct AdjacencyGraph <: AbstractGraph

    adjacency :: Matrix{Float64}

end

getNumNodes(g :: AdjacencyGraph) = size(g.adjacency, 1)
getNumEdges(g :: AdjacencyGraph) = div(count(!iszero, g.adjacency), 2)

getDenseAdjacency(g :: AdjacencyGraph) = g.adjacency

getSubGraph(g :: AdjacencyGraph, ind :: Vector{Int64}) = AdjacencyGraph(g.adjacency[ind, ind])


### Sparse equivalent of AdjacencyGraph

"""
	SparseAdjacencyGraph

Subtype of `AbstractGraph` for graphs given by their sparse adjacency matrix.
"""
mutable struct SparseAdjacencyGraph <: AbstractGraph

	adjacency :: AbstractSparseMatrix{Float64, Int64}

end

getNumNodes(g :: SparseAdjacencyGraph) = size(g.adjacency, 1)
getNumEdges(g :: SparseAdjacencyGraph) = div(nnz(g.adjacency), 2)

getSparseAdjacency(g :: SparseAdjacencyGraph) = g.adjacency
isSparse(:: SparseAdjacencyGraph) = true

getSubGraph(g :: SparseAdjacencyGraph, ind :: Vector{Int64}) = AdjacencyGraph(g.adjacency[ind, ind])

### Type for classical graphs given by a list of edge endpoints

"""
	EdgeListGraph

Subtype of `AbstractGraph` for classical graphs given by a list of edge
endpoints.
"""
mutable struct EdgeListGraph <: AbstractGraph

    numNodes :: Int64
    edges :: Vector{Tuple{Int64,Int64}}
    weights :: Vector{Float64}

end

getNumNodes(g :: EdgeListGraph) = g.numNodes
getNumEdges(g :: EdgeListGraph) = length(g.edges)

function getSparseAdjacency(g :: EdgeListGraph)
	numEdges = getNumEdges(g)
	I = zeros(2*numEdges)
	J = zeros(2*numEdges)
	V = zeros(2*numEdges)
	for i = 1:numEdges
		I[2*i-1], J[2*i-1] = J[2*i], I[2*i] = g.edges[i]
		V[2*i-1] = V[2*i] = g.weights[i]
	end
	return sparse(I, J, V, g.numNodes, g.numNodes)
end
isSparse(:: EdgeListGraph) = true

function getSubGraph(g :: EdgeListGraph, ind :: Vector{Int64})
    newEdges = Vector{Tuple{Int64,Int64}}(0)
    newWeights = Vector{Float64}
    for k = 1:length(g.edges)
        i,j = g.edges[k]
        indi = findfirst(ind .== i)
        indj = findfirst(ind .== j)
        if indi > 0 && indj > 0
            push!(newEdges, (indi, indj))
            push!(newWeights, g.weights[k])
        end
    end
    return EdgeListGraph(size(ind), newEdges, newWeights)
end


### FULL GAUSSIAN GRAPH

"""
	FullGaussianGraph

Subtype of `AbstractGraph` for fully connected graphs with Gaussian weights
based on a node feature matrix.
"""
mutable struct FullGaussianGraph <: AbstractGraph
	features :: Matrix{Float64} # n×d, one row per node
	sigma :: Float64
end

getNumNodes(g :: FullGaussianGraph) = size(g.features, 1)
getNumEdges(g :: FullGaussianGraph) = (n = size(g.features, 1); div(n*(n-1), 2))

function getDenseAdjacency(g :: FullGaussianGraph)
	n = size(g.features, 1)
	dist2 = sum(g.features.^2, dims=2) .-
		2*g.features*g.features' .+
		sum(g.features.^2, dims=2)'
	return exp.(-dist2 ./ g.sigma^2) - UniformScaling(1)
end
isSparse(:: FullGaussianGraph) = false
getSubGraph(g :: FullGaussianGraph, ind :: Vector{Int64}) =
	FullGaussianGraph(g.features[ind, :], g.sigma)
