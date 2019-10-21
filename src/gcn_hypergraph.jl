

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

function setupMatrices(kernel :: PolySmoothedHypergraphLaplacianKernel, dataset :: Dataset)
	h = dataset.graph :: Hypergraph
	s = kernel.α .+ (kernel.β-1)*h.loopWeights
	return (1.0 .- (s ./ (s ./ (h.nodeDegrees .+ s))),
		Diagonal(1 ./ sqrt.(h.nodeDegrees .+ s)) * h.incidence * Diagonal(sqrt.(h.weights./h.edgeDegrees)))
end

"""
	PolyHypergraphLaplacianKernel

`GCNKernel` subtype for efficient evaluation of a full-rank polynomial filter
space using the hypergraph Laplacian.
"""
mutable struct PolyHypergraphLaplacianKernel <: GCNKernel
	coeffs :: Vector{Vector{Float64}}
end
numParts(k :: PolyHypergraphLaplacianKernel) = length(k.coeffs)

function setupMatrices(kernel :: PolyHypergraphLaplacianKernel, dataset :: Dataset)
	hg = dataset.graph :: Hypergraph
	H = Diagonal(1 ./ sqrt.(hg.nodeDegrees)) * hg.incidence * Diagonal(sqrt.(hg.weights ./ hg.edgeDegrees))

	K = numParts(kernel)
	degrees = zeros(Int, K)
	b = [Float64[] for i = 1:K]
	for i = 1:K
		a = kernel.coeffs[i]
		d = length(a) - 1
		degrees[i] = d
		b[i] = [dot(binomial.(k:d, k), a[k+1:end]) for k=0:d]
	end

	M = Any[UniformScaling(degrees[i] == 0 ? 0.0 : -b[i][2]) for i=1:K]
	if any(degrees .> 1)
		HtH = H' * H
		negHtHPower = UniformScaling(-1.0)

		for k in 2:maximum(degrees)
			negHtHPower *= - HtH # in each iteration: negHtHPower == -(-H'*H)^(k-1) == (-1)^k (H'*H)^(k-1)
			for i in 1:K
				if degrees[i] >= k
					M[i] += b[i][k+1] * negHtHPower
				end
			end
		end
	end

	# vector of scaling factors, H-matrix, vector of M-matrices
	return [v[1] for v in b], H, M
end

"""
	LowRankPolyHypergraphLaplacianKernel

`GCNKernel` subtype for efficient evaluation of a low-rank polynomial filter
space using the hypergraph Laplacian.
"""
mutable struct LowRankPolyHypergraphLaplacianKernel <: GCNKernel
	coeffs :: Vector{Vector{Float64}}
	rank :: Int64
	whichEV :: Symbol
end
LowRankPolyHypergraphLaplacianKernel(coeffs :: Vector{Vector{Float64}}, rank :: Int64;
		whichEV :: Symbol = :small) =
	LowRankPolyHypergraphLaplacianKernel(coeffs, rank, whichEV)
numParts(kernel :: LowRankPolyHypergraphLaplacianKernel) = length(kernel.coeffs)

function setupMatrices(kernel :: LowRankPolyHypergraphLaplacianKernel, dataset :: Dataset)
	hg = dataset.graph :: Hypergraph
	H = Diagonal(1 ./ sqrt.(hg.nodeDegrees)) * hg.incidence * Diagonal(sqrt.(hg.weights ./ hg.edgeDegrees))

	if kernel.whichEV == :small
		firstEV = 1
		numEV = kernel.rank
	elseif kernel.whichEV == :smallnonzero
		firstEV = 2
		numEV = kernel.rank+1
	else
		throw(ArgumentError("Unsupported eigenvalue specifier: $(tfk.kernel.whichEV)"))
	end

	if numEV > minimum(size(H))
		throw(ArgumentError(
			"Requested number of eigenvalues $(lastEV) is larger than the incidence matrix allows"))
	elseif 2*numEV > minimum(size(H))
		U, Σ = svd(H)
	else
		(U,Σ), = Arpack.svds(H, nsv=numEV)
	end
	U = U[:, firstEV:numEV]
	λ = 1 .- Σ[firstEV:numEV].^2

	diags = Vector{Float64}[]
    for i = 1:numParts(kernel)
        d = zeros(kernel.rank)
        for j = 1:length(kernel.coeffs[i])
            d += kernel.coeffs[i][j] * λ.^(j-1)
        end
        push!(diags, d)
    end
	return U, d
end


"""
	LowRankInvHypergraphLaplacianKernel

`GCNKernel` subtype for efficient evaluation of the one-dimensional filter space
spanned by the low-rank pseudoinverse filter function with the hypergraph
Laplacian.
"""
mutable struct LowRankInvHypergraphLaplacianKernel <: GCNKernel
	rank :: Int64
end

function setupMatrices(kernel :: LowRankInvHypergraphLaplacianKernel, dataset :: Dataset)

	hg = dataset.graph :: Hypergraph
	H = Diagonal(1 ./ sqrt.(hg.nodeDegrees)) * hg.incidence * Diagonal(sqrt.(hg.weights ./ hg.edgeDegrees))

	numEV = kernel.rank+1
	if numEV > minimum(size(H))
		throw(ArgumentError(
			"Requested number of eigenvalues $(numEV) is larger than the incidence matrix allows"))
	elseif 2*numEV > minimum(size(H))
		U, Σ = svd(H)
	else
		(U,Σ), = Arpack.svds(H, nsv=numEV)
	end
	λ = 1 .- Σ[2:numEV].^2
	return U[:,2:end], λ
end


"""
	InvHypergraphLaplacianKernel

`GCNKernel` subtype for efficient evaluation of the one-dimensional filter space
spanned by the full-rank pseudoinverse filter function with the hypergraph
Laplacian.
"""
struct InvHypergraphLaplacianKernel <: GCNKernel end

function setupMatrices( :: InvHypergraphLaplacianKernel, dataset :: Dataset)

	hg = dataset.graph :: Hypergraph
	H = Diagonal(1 ./ sqrt.(hg.nodeDegrees)) * hg.incidence * Diagonal(sqrt.(hg.weights ./ hg.edgeDegrees))

	U, Σ = svd(H)
	Λ = 1 .- Σ[2:end].^2
	λmin = minimum(Λ)

	# scaling factor, U, diagonal of lowrank part
	return λmin, U, λmin * (vcat(0, 1 ./ Λ) .- 1)
end
