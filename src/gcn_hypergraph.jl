

export
	Hypergraph,
	removeEmptyHyperEdges!,
	transformIntoGraph,
	HypergraphSmoother,
	PolySmoothedHypergraphLaplacianKernel,
	TFPolySmoothedHypergraphLaplacianKernel,
	PolyHypergraphLaplacianKernel,
	TFPolyHypergraphLaplacianKernel,
	InvHypergraphLaplacianKernel,
	TFInvHypergraphLaplacianKernel

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

function getDenseAdjacency(h :: Hypergraph)
    n, m = size(h.incidence)

    adj = zeros(n,n)
    for i = 1:n
        for j = 1:i-1
            for k = 1:m
                if h.incidence[i,k] > 0 && h.incidence[j,k] > 0
                    adj[i,j] += h.weights[k]
                end
            end
            adj[j,i] = adj[i,j]
        end
    end
    return adj
end
isSparse(:: Hypergraph) = false

function getSubGraph(h :: Hypergraph, ind :: Vector{Int64})
    return removeEmptyHyperEdges!(Hypergraph(h.incidence[ind,:], h.weights))
end

"""
	removeEmptyHyperEdges!(h :: Hypergraph)

Removes zero columns from the hypergraph's incidence matrix, as well as the
corresponding edge weights and degrees.
"""
function removeEmptyHyperEdges!(h :: Hypergraph)
    doKeep = h.edgeDegrees .> 0
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

	normalizer = applySmoother(smoother, Diagonal(-h.loopWeights), h)
	invD = inv(Diagonal(h.nodeDegrees) + normalizer)
	normalizedInc = sqrt(invD) * h.incidence
	return Symmetric(LinearAlgebra.I -
			invD * normalizer -
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
PolySmoothedHypergraphLaplacianKernel(coeffs :: Vector{Vector{Float64}}; α = 0.0 :: Float64, β = 0.0 :: Float64) =
	PolySmoothedHypergraphLaplacianKernel(coeffs, α, β)
PolySmoothedHypergraphLaplacianKernel(singleCoeffs :: Vector{Float64}; α = 0.0 :: Float64, β = 0.0 :: Float64) =
	PolySmoothedHypergraphLaplacianKernel([singleCoeffs], α, β)

numParts(k :: PolySmoothedHypergraphLaplacianKernel) = length(k.coeffs)

mutable struct TFPolySmoothedHypergraphLaplacianKernel <: TFKernel
	kernel :: PolySmoothedHypergraphLaplacianKernel
	diagPartTensor :: tf.Tensor
	lowRankFactorTensor :: tf.Tensor
end
function TFKernel(k :: PolySmoothedHypergraphLaplacianKernel)
	return TFPolySmoothedHypergraphLaplacianKernel(k,
		tf.placeholder(Float64, shape=[missing, 1], name="kernelDiagPart"),
		tf.placeholder(Float64, shape=[missing,missing], name="kernelLowRankFactor"))
end
function applyLayer(tfk::TFPolySmoothedHypergraphLaplacianKernel, XParts :: Vector{<: tf.Tensor})
	parts = tf.Tensor[]
	for i in 1:length(XParts)
		x = XParts[i]
		push!(parts, tfk.kernel.coeffs[i][1] * x)
		for c in tfk.kernel.coeffs[i][2:end]
			x = (tfk.diagPartTensor .* x) - tfk.lowRankFactorTensor * (tfk.lowRankFactorTensor' * x)
			push!(parts, c * x)
		end
	end
	return sum(parts)
end
function fillFeedDict(tfk :: TFPolySmoothedHypergraphLaplacianKernel, dataset :: Dataset, dict :: Dict)
	k = tfk.kernel
	h = dataset.graph
	@assert h isa Hypergraph
	s = k.α .+ (k.β-1)*h.loopWeights
	dict[tfk.diagPartTensor] = hcat(1.0 .- (s ./ (h.nodeDegrees .+ s))) # hcat transforms vector into matrix
	dict[tfk.lowRankFactorTensor] =
		Diagonal(1 ./ sqrt.(h.nodeDegrees .+ s)) * h.incidence * Diagonal(sqrt.(h.weights./h.edgeDegrees))
	return dict
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

mutable struct TFPolyHypergraphLaplacianKernel <: TFKernel
	kernel :: PolyHypergraphLaplacianKernel
	bTensors :: Vector{<: tf.Tensor}
	HTensor :: tf.Tensor
	MTensors :: Vector{<: tf.Tensor}
end
function TFKernel(k :: PolyHypergraphLaplacianKernel)
	K = numParts(k)
	return TFPolyHypergraphLaplacianKernel(k,
		[tf.placeholder(Float64, shape=[], name="kernel_b$i") for i=1:K],
		tf.placeholder(Float64, shape=[missing,missing], name="kernel_H"),
		[tf.placeholder(Float64, shape=[missing,missing], name="kernel_M$i") for i=1:K])
end
function applyLayer(tfk :: TFPolyHypergraphLaplacianKernel, XParts :: Vector{<: tf.Tensor})
	K = length(XParts)
	scaledParts = tf.Tensor[tfk.bTensors[i] .* XParts[i] for i=1:K]
	lowrankParts = tf.Tensor[]
	for i=1:K
		degree = length(tfk.kernel.coeffs[i]) - 1
		if degree == 1
			push!(lowrankParts, tfk.MTensors[i] .* (tfk.HTensor'*XParts[i])) # M is scalar
		elseif degree > 1
			push!(lowrankParts, tfk.MTensors[i] * (tfk.HTensor'*XParts[i])) # M is matrix
		end
	end
	return tf.Ops.add_n(scaledParts) + tfk.HTensor * tf.Ops.add_n(lowrankParts)
end
function fillFeedDict(tfk :: TFPolyHypergraphLaplacianKernel, dataset :: Dataset, dict :: Dict)
	hg = dataset.graph :: Hypergraph
	H = Diagonal(1 ./ sqrt.(hg.nodeDegrees)) * hg.incidence * Diagonal(sqrt.(hg.weights ./ hg.edgeDegrees))

	K = numParts(tfk.kernel)
	degrees = zeros(Int, K)
	b = [Float64[] for i = 1:K]
	for i = 1:K
		a = tfk.kernel.coeffs[i]
		d = length(a) - 1
		degrees[i] = d
		b[i] = [sum(binomial.(k:d, k) .* a[k+1:end]) for k=0:d]
	end

	M = Any[UniformScaling(degrees[i] == 0 ? 0.0 : -b[i][2]) for i=1:K]
	if any(degrees .> 0)
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

	dict[tfk.HTensor] = H
	for i in 1:K
		dict[tfk.bTensors[i]] = b[i][1]
		dict[tfk.MTensors[i]] = isa(M[i], UniformScaling) ? hcat(M[i].λ) : M[i]
	end
end


mutable struct LowRankPolyHypergraphLaplacianKernel <: GCNKernel
	coeffs :: Vector{Vector{Float64}}
	whichEV :: Symbol
	isReduced :: Bool
end
numParts(kernel :: LowRankPolyHypergraphLaplacianKernel) = length(kernel.coeffs)


mutable struct TFLowRankPolyHypergraphLaplacianKernel <: TFKernel
	kernel :: LowRankPolyHypergraphLaplacianKernel
    UTensor :: tf.Tensor
    kernelDiagTensors :: Vector{tf.Tensor}
end
function TFKernel(kernel :: LowRankPolyHypergraphLaplacianKernel)
    TFLowRankPolyHypergraphLaplacianKernel(kernel,
        tf.placeholder(Float64, shape=[missing, kernel.rank], name="U"),
        [tf.placeholder(Float64, shape=[kernel.rank, 1], name="KernelDiag$i")
            for i in 1:numParts(kernel)])
end

function applyLayer(tfk :: TFLowRankPolyHypergraphLaplacianKernel, XParts :: Vector{tf.Tensor{Float64}})
    numParts = length(XParts)
    if tfk.kernel.isReduced
        parts = [tfk.kernelDiagTensors[i] .* XParts[i] for i = 1:numParts]
        return numParts == 1 ? parts[1] : tf.Ops.add_n(parts)
    else
        parts = [tfk.kernelDiagTensors[i] .* (tfk.UTensor' * XParts[i])
                    for i = 1:numParts]
        return tfk.UTensor * (numParts == 1 ? parts[1] : tf.Ops.add_n(parts))
    end
end
transformInput(tfk :: TFLowRankPolyHypergraphLaplacianKernel, input :: tf.Tensor) =
    tfk.kernel.isReduced ? tfk.UTensor' * input : input
transformOutput(tfk :: TFLowRankPolyHypergraphLaplacianKernel, output :: tf.Tensor) =
    tfk.kernel.isReduced ? tfk.UTensor * output : output

function fillFeedDict(tfk :: TFLowRankPolyHypergraphLaplacianKernel, dataset :: Dataset, dict :: Dict)

	hg = dataset.graph :: Hypergraph
	H = Diagonal(1 ./ sqrt.(hg.nodeDegrees)) * hg.incidence * Diagonal(sqrt.(hg.weights ./ hg.edgeDegrees))

	if tfk.kernel.whichEV == :small
		(U,Σ), = Arpack.svds(H, nsv=tfk.kernel.rank)
		λ = 1 .- Σ.^2
	elseif tfk.kernel.whichEV == :smallnonzero
		(U,Σ), = Arpack.svds(H, nsv=tfk.kernel.rank+1)
		U = U[:, 2:end]
		λ = 1 .- Σ[2:end].^2
	else
		throw(ArgumentError("Unsupported eigenvalue specifier: $(tfk.kernel.whichEV)"))
	end

    dict[tfk.UTensor] = U
    for i = 1:numParts(tfk.kernel)
        d = zeros(tfk.kernel.rank)
        for j = 1:length(tfk.kernel.coeffs[i])
            d += tfk.kernel.coeffs[i][j] * λ.^(j-1)
        end
        dict[tfk.kernelDiagTensors[i]] = hcat(d)
    end
end




mutable struct LowRankInvHypergraphLaplacianKernel <: GCNKernel
	rank :: Int64
	isReduced :: Bool
end

mutable struct TFLowRankInvHypergraphLaplacianKernel <: TFKernel
	kernel :: LowRankInvHypergraphLaplacianKernel
	UTensor :: tf.Tensor
	kernelDiagTensor :: tf.Tensor
end

function TFKernel(kernel :: LowRankInvHypergraphLaplacianKernel)
    return TFLowRankInvHypergraphLaplacianKernel(kernel,
        tf.placeholder(Float64, shape=[missing, kernel.rank], name="U"),
        tf.placeholder(Float64, shape=[kernel.rank, 1], name="kernelDiag"))
end

function applyLayer(tfk :: TFLowRankInvHypergraphLaplacianKernel, XParts :: Vector{tf.Tensor{Float64}})
    if tfk.kernel.isReduced
        return tfk.kernelDiagTensor .* XParts[1]
    else
        return tfk.UTensor * (tfk.kernelDiagTensor .* (tfk.UTensor' * XParts[1]))
    end
end
transformInput(tfk :: TFLowRankInvHypergraphLaplacianKernel, input :: tf.Tensor) =
    tfk.kernel.isReduced ? tfk.UTensor' * input : input
transformOutput(tfk :: TFLowRankInvHypergraphLaplacianKernel, output :: tf.Tensor) =
    tfk.kernel.isReduced ? tfk.UTensor * output : output

function fillFeedDict(tfk :: TFLowRankInvHypergraphLaplacianKernel, dataset :: Dataset, dict :: Dict)

	hg = dataset.graph :: Hypergraph
	H = Diagonal(1 ./ sqrt.(hg.nodeDegrees)) * hg.incidence * Diagonal(sqrt.(hg.weights ./ hg.edgeDegrees))

	(U,Σ), = Arpack.svds(H, nsv=tfk.kernel.rank+1)
	λ = 1 .- Σ[2:end].^2

    dict[tfk.UTensor] = U[:,2:end]
    dict[tfk.kernelDiagTensor] = hcat(minimum(λ) ./ λ)
end



struct InvHypergraphLaplacianKernel <: GCNKernel end


mutable struct TFInvHypergraphLaplacianKernel <: TFKernel
	UTensor :: tf.Tensor
	kernelDiagTensor :: tf.Tensor
end

TFKernel(:: InvHypergraphLaplacianKernel) = TFInvHypergraphLaplacianKernel(
	tf.placeholder(Float64, shape=[missing, missing], name="U"),
	tf.placeholder(Float64, shape=[missing, 1], name="kernelDiag"))


applyLayer(tfk :: TFInvHypergraphLaplacianKernel, XParts :: Vector{tf.Tensor{Float64}}) =
    XParts[1] - tfk.UTensor * (tfk.kernelDiagTensor .* (tfk.UTensor' * XParts[1]))


function fillFeedDict(tfk :: TFInvHypergraphLaplacianKernel, dataset :: Dataset, dict :: Dict)

	hg = dataset.graph :: Hypergraph
	H = Diagonal(1 ./ sqrt.(hg.nodeDegrees)) * hg.incidence * Diagonal(sqrt.(hg.weights ./ hg.edgeDegrees))

	U, Σ = svd(H)
	Λ = 1 .- Σ[2:end].^2
	λmin = minimum(Λ)

    dict[tfk.UTensor] = U
    dict[tfk.kernelDiagTensor] = hcat(1 .- vcat(0, λmin ./ Λ))
end
