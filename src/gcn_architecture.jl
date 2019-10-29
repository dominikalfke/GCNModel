

export
    GCNKernel,
    numParts,
    computeMatrices,
    IdentityKernel,
    FixedMatrixKernel,
    FixedLowRankKernel,
    PolyLaplacianKernel,
	InverseLaplacianKernel,
    LowRankPolyLaplacianKernel,
    LowRankInvLaplacianKernel,
    GCNArchitecture,
	checkCompatibility

"""
    GCNKernel

Abstract supertype for kernel architectures.
"""
abstract type GCNKernel end

"""
    K = numParts(k :: GCNKernel)

Returns the dimension of the filter space underlying the kernel object, i.e. the
number of weight matrices required per layer.
"""
numParts(:: GCNKernel) = 1

computeMatrices(:: GCNKernel) = nothing

"""
    IdentityKernel

`GCNKernel` subtype that corresponds to a single kernel matrix that is simply
the identity matrix.
"""
struct IdentityKernel <: GCNKernel end

mutable struct FixedMatrixKernel <: GCNKernel
    matrices :: Vector{Matrix{Float64}}
end
numParts(kernel :: FixedMatrixKernel) = length(kernel.matrices)
computeMatrices(kernel :: FixedMatrixKernel, ::Dataset) = kernel.matrices

mutable struct FixedLowRankKernel <: GCNKernel
    projector :: Matrix{Float64}
    diagonals :: Vector{Vector{Float64}}
end
numParts(kernel :: FixedLowRankKernel) = length(kernel.diagonals)
computeMatrices(kernel :: FixedLowRankKernel, :: Dataset) = (kernel.projector, kernel.diagonals)


"""
    PolyLaplacianKernel

`GCNKernel` subtype for a polynomial filter space. Coefficients of the basis
polynomials are stored in the `coeffs` field as a Vector of Vector{Float64}. The
`smoother` field may hold any object for which `applySmoother` is implemented.
In the TensorFlow implementation, all matrices will be stored densely. In the
case of special Laplacian structure (e.g. Hypergraphs) should use specialized
kernel types instead of this one.
"""
mutable struct PolyLaplacianKernel <: GCNKernel
    coeffs :: Vector{Vector{Float64}}
    smoother
end
PolyLaplacianKernel(coeffs :: Vector{Vector{Float64}};
        smoother = nothing) =
    PolyLaplacianKernel(coeffs, smoother)

numParts(kernel :: PolyLaplacianKernel) =
    length(kernel.coeffs)

function computeMatrices(kernel :: PolyLaplacianKernel, dataset :: Dataset)
    L = getFullLaplacian(dataset.graph, kernel.smoother)
    X = UniformScaling(1.0)
    kernelParts = Any[c[1]*X for c in kernel.coeffs]
    for i = 2:maximum([length(c) for c in kernel.coeffs])
        X = L * X
        for j in 1:length(kernel.coeffs)
            if length(kernel.coeffs[j]) >= i
                kernelParts[j] += kernel.coeffs[j][i] * X
            end
        end
    end
    return kernelParts
end

"""
	InverseLapacianKernel
"""
mutable struct InverseLaplacianKernel <: GCNKernel
	smoother
end
InverseLaplacianKernel(; smoother=nothing) = InverseLaplacianKernel(smoother)

function computeMatrices(kernel :: InverseLaplacianKernel, dataset :: Dataset)
	L = getFullLaplacian(dataset.graph, kernel.smoother)
	return pinv(Matrix(L), atol=1e-4)
end

"""
    LowRankPolyLaplacianKernel

`GCNKernel` subtype for basis functions that are low-rank approximations to
polynomials. All basis functions must be defined on the same Laplacian
eigenvalues. The `isReduced` field determines whether a reduced-order GCN is
constructed.
"""
mutable struct LowRankPolyLaplacianKernel <: GCNKernel
    coeffs :: Vector{Vector{Float64}}
    rank :: Int64
    whichEV :: Symbol
    smoother
end
LowRankPolyLaplacianKernel(coeffs :: Vector{Vector{Float64}}, rank :: Int64;
        whichEV :: Symbol = :small,
        smoother = nothing) =
    LowRankPolyLaplacianKernel(coeffs, rank, whichEV, smoother)

numParts(kernel :: LowRankPolyLaplacianKernel) =
    length(kernel.coeffs)

function computeMatrices(kernel :: LowRankPolyLaplacianKernel, dataset :: Dataset)
    λ, U = getLaplacianEigenvalues(dataset.graph,
                kernel.rank, kernel.whichEV, kernel.smoother)
	diagonals = Vector{Float64}[]
    for i = 1:numParts(kernel)
        d = zeros(length(λ))
        for j = 1:length(kernel.coeffs[i])
            d += kernel.coeffs[i][j] * λ.^(j-1)
        end
        push!(diagonals, d)
    end
	return U, diagonals
end

"""
    LowRankInvLaplacianKernel

`GCNKernel` subtype for a single filter function that gives a low-rank
approximation to the Moore-Penrose pseudo-inverse of the Laplacian. The
`isReduced` field determines whether a reduced-order GCN is constructed.
"""
mutable struct LowRankInvLaplacianKernel <: GCNKernel
    rank :: Int64
    smoother
    isReduced :: Bool
end
LowRankInvLaplacianKernel(rank :: Int64;
        smoother = nothing,
        isReduced :: Bool = false) =
    LowRankInvLaplacianKernel(rank, smoother, isReduced)

function computeMatrices(kernel :: LowRankInvLaplacianKernel, dataset :: Dataset)
	λ, U = getLaplacianEigenvalues(dataset.graph,
                kernel.rank, :smallnonzero, kernel.smoother)
    return U, minimum(λ) ./ λ
end


"""
    GCNArchitecture

Data type for all data describing the architecture of a GCN. This includes a
`GCNKernel` object describing the filter space or the kernel matrices, the
series of sizes of the layer matrices, the type of activation function, and a
regularization parameter.

In particular, this type holds all data required to create an implementation of
a GCN while being easily storable, and contains no data that is part of the
dataset or the actual implementation.
"""
mutable struct GCNArchitecture

    name :: String
    kernel :: GCNKernel
    layerWidths :: Vector{Int64}
    activation :: Activation
    regParam :: Float64

    GCNArchitecture(layerWidths :: Vector{Int64}, kernel :: GCNKernel;
            name :: String = "GCN",
            activation :: Activation = Relu(),
            regParam :: Float64 = 5e-4) =
        new(name, kernel, layerWidths, activation, regParam)

end


function checkCompatibility(arc :: GCNArchitecture, dataset :: Dataset)
    dataset.numFeatures == arc.layerWidths[1] ||
        error("Number of features in dataset $(dataset.name) does not match the first layer width")
    dataset.numLabels == arc.layerWidths[end] ||
        error("Number of classes in dataset $(dataset.name) does not match the last layer width")
end
