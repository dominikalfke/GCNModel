

export
	Hypergraph,
	removeEmptyHyperEdges!,
	transformIntoGraph,
	HypergraphSmoother,
	PolySmoothedHypergraphLaplacianKernel,
	PolyHypergraphLaplacianKernel,
	LowRankPolyHypergraphLaplacianKernel,
	LowRankInvHypergraphLaplacianKernel,
	InvHypergraphLaplacianKernel

# Type for Hypergraphs given by their incidence matrix

"""
	Hypergraph

Subtype of `AbstractGraph` for hypergraphs, i.e. graphs where edges can connect
more than two nodes. Each hypergraph is determined by an incidence matrix and
the hyperedge weight vector. For convenience, this type also stores the edge
degrees, node degrees, and loop weights that occur by transforming the graph
into a classical graph with self loops.
"""
mutable struct Hypergraph <: AbstractGraph
    incidence :: Matrix{Float64}
    weights :: Vector{Float64}
	edgeDegrees :: Vector{Float64}
	nodeDegrees :: Vector{Float64}
	loopWeights :: Vector{Float64}

	function Hypergraph(inc :: Matrix{Float64}, weights :: Vector{Float64})
		n, m = size(inc)
		length(weights) == m || throw(ArgumentError("Incorrect number of entries in Hypergraph \"weights\" argument"))
		h = new(inc, weights, zeros(m), zeros(n), zeros(n))

		for k = 1:m
			h.edgeDegrees[k] = sum(inc[:,k])
			h.nodeDegrees += weights[k] * inc[:,k]
			h.loopWeights += weights[k] / h.edgeDegrees[k] * inc[:, k]
		end
		return h
	end
end



getNumNodes(h :: Hypergraph) = size(h.incidence, 1)
getNumEdges(h :: Hypergraph) = size(h.incidence, 2)

getDenseAdjacency(h :: Hypergraph) =
	h.incidence * Diagonal(h.weights ./ h.edgeDegrees) * h.incidence' - Diagonal(h.loopWeights)

isSparse(:: Hypergraph) = false

function getSubGraph(h :: Hypergraph, ind :: Vector{Int64})
    return removeEmptyHyperEdges!(Hypergraph(h.incidence[ind,:], h.weights))
end

"""
	removeEmptyHyperEdges!(h :: Hypergraph)

Removes columns from the hypergraph's incidence matrix that have one or less
entries, as well as the corresponding edge weights and degrees.
"""
function removeEmptyHyperEdges!(h :: Hypergraph)
    doKeep = h.edgeDegrees .>= 2.0
    h.incidence = h.incidence[:, doKeep]
    h.weights = h.weights[doKeep]
	h.edgeDegrees = h.edgeDegrees[doKeep]
    return h
end

"""
	transformIntoGraph(h :: Hypergraph)

Transform a Hypergraph into a classical graph with self-loops. Returns an
`AdjacencyGraph` object.
"""
transformIntoGraph(h :: Hypergraph) =
	AdjacencyGraph(getDenseAdjacency(h))


"""
	HypergraphSmoother

Simple type for hypergraph smoothing, which implements the smoother
	`S = α I + β diag(h.loopWeights)`.
Only the coefficients `α` and `β` are stored in this type. Comes with a special
`applySmoother` method, where the `graph` argument must be a `Hypergraph`
object.
"""
struct HypergraphSmoother
	α :: Float64
	β :: Float64
end
applySmoother(hs :: HypergraphSmoother, adjacency :: AbstractMatrix{Float64}, hypergraph :: Hypergraph) =
	adjacency + Diagonal(hs.β * hypergraph.loopWeights .+ hs.α)


function getFullLaplacian(h :: Hypergraph, smoother = nothing)

	SminusSH = applySmoother(smoother, Diagonal(-h.loopWeights), h)
	invDplusS = inv(Diagonal(h.nodeDegrees) + SminusSH)
	normalizedInc = sqrt(invDplusS) * h.incidence
	return Symmetric(LinearAlgebra.I -
			invDplusS * SminusSH -
			normalizedInc * Diagonal(h.weights ./ h.edgeDegrees) * transpose(normalizedInc))
end


"""
	PolySmoothedHypergraphLaplacianKernel

`GCNKernel` subtype for efficient evaluation of a full-rank polynomial filter
space using a smoothed Laplacian of a hypergraph dataset. The fields `α` and
`β` determine the diagonal smoother matrix `S = α I + β S_H`, where `I` is the
identity and `S_H` is the loop matrix that turns the graph Laplacian into the
hypergraph Laplacian. The case `α=0,β=0` represents turning the hypergraph into
a graph, removing its loops, and using the graph Laplacian. The case `α=0,β=1`
represents the classical hypergraph Laplacian, for which the type
`PolyHypergraphLaplacianKernel` supplies an even more efficient implementation.
"""
mutable struct PolySmoothedHypergraphLaplacianKernel <: GCNKernel
    coeffs :: Vector{Vector{Float64}}
    α :: Float64
    β :: Float64
end
PolySmoothedHypergraphLaplacianKernel(coeffs :: Vector{Vector{Float64}}; α :: Float64 = 0.0, β :: Float64 = 0.0) =
	PolySmoothedHypergraphLaplacianKernel(coeffs, α, β)
PolySmoothedHypergraphLaplacianKernel(singleCoeffs :: Vector{Float64}; α :: Float64 = 0.0, β:: Float64 = 0.0) =
	PolySmoothedHypergraphLaplacianKernel([singleCoeffs], α, β)

numParts(k :: PolySmoothedHypergraphLaplacianKernel) = length(k.coeffs)


"""
	PolyHypergraphLaplacianKernel

`GCNKernel` subtype for efficient evaluation of a full-rank polynomial filter
space using the hypergraph Laplacian.
"""
mutable struct PolyHypergraphLaplacianKernel <: GCNKernel
	coeffs :: Vector{Vector{Float64}}
end
numParts(k :: PolyHypergraphLaplacianKernel) = length(k.coeffs)


"""
	LowRankPolyHypergraphLaplacianKernel

`GCNKernel` subtype for efficient evaluation of a low-rank polynomial filter
space using the hypergraph Laplacian.
"""
mutable struct LowRankPolyHypergraphLaplacianKernel <: GCNKernel
	coeffs :: Vector{Vector{Float64}}
	rank :: Int64
	whichEV :: Symbol
	isReduced :: Bool
end
LowRankPolyHypergraphLaplacianKernel(coeffs :: Vector{Vector{Float64}}, rank :: Int64;
		whichEV :: Symbol = :small, isReduced :: Bool = false) =
	LowRankPolyHypergraphLaplacianKernel(coeffs, rank, whichEV, isReduced)
numParts(kernel :: LowRankPolyHypergraphLaplacianKernel) = length(kernel.coeffs)




"""
	LowRankInvHypergraphLaplacianKernel

`GCNKernel` subtype for efficient evaluation of the one-dimensional filter space
spanned by the low-rank pseudoinverse filter function with the hypergraph
Laplacian.
"""
mutable struct LowRankInvHypergraphLaplacianKernel <: GCNKernel
	rank :: Int64
	isReduced :: Bool
end


"""
	InvHypergraphLaplacianKernel

`GCNKernel` subtype for efficient evaluation of the one-dimensional filter space
spanned by the full-rank pseudoinverse filter function with the hypergraph
Laplacian.
"""
struct InvHypergraphLaplacianKernel <: GCNKernel end
