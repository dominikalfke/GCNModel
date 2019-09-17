

export
    GCNKernel,
    numParts,
    IdentityKernel,
    FixedMatrixKernel,
    FixedLowRankKernel,
    PolyLaplacianKernel,
    LowRankPolyLaplacianKernel,
    LowRankInvLaplacianKernel,
    GCNArchitecture

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


mutable struct FixedLowRankKernel <: GCNKernel
    projector :: Matrix{Float64}
    diagonals :: Vector{Vector{Float64}}
    isReduced :: Bool
end
numParts(kernel :: FixedLowRankKernel) = length(kernel.diagonals)

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
        smoother = nothing :: Any) =
    PolyLaplacianKernel(coeffs, smoother)

numParts(kernel :: PolyLaplacianKernel) =
    length(kernel.coeffs)

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
    isReduced :: Bool
end
LowRankPolyLaplacianKernel(coeffs :: Vector{Vector{Float64}}, rank :: Int64;
        whichEV = :small :: Symbol,
        smoother = nothing,
        isReduced = false :: Bool) =
    LowRankPolyLaplacianKernel(coeffs, rank, whichEV,
        smoother, isReduced)

numParts(kernel :: LowRankPolyLaplacianKernel) =
    length(kernel.coeffs)

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
        isReduced = false :: Bool) =
    LowRankInvLaplacianKernel(rank, smoother, isReduced)


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
    activation :: Symbol
    regParam :: Float64

    GCNArchitecture(layerWidths :: Vector{Int64}, kernel :: GCNKernel;
            name = "GCN" :: String,
            activation = :relu :: Symbol,
            regParam = 5e-4 :: Float64) =
        new(name, kernel, layerWidths, activation, regParam)

end
