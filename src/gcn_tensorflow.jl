

import TensorFlow
import Arpack
const tf = TensorFlow


export
    TFKernel,
    applyLayer,
    transformInput,
    transformOutput,
    fillFeedDict,
    TFFixedMatrixKernel,
    TFFixedLowRankKernel,
    TFPolyLaplacianKernel,
    TFLowRankPolyLaplacianKernel,
    TFLowRankInvLaplacianKernel,
    applyActivation,
    TensorFlowGCN,
    initializeRandomWeights,
    exportWeights,
    importWeights,
    saveWeights,
    loadWeights,
    createFeedDicts,
    createTrainedSession

"""
    TFKernel

Abstract supertype for objects that hold the data of the TensorFlow
implementation of a kernel.
"""
abstract type TFKernel end

"""
    tfk = TFKernel(kernel :: GCNKernel)

Construct an object of a `TFKernel` subtype based on the given subtype of
`GCNKernel`. Throws an `ArgumentError` if no specialized method is available.
"""
TFKernel(k :: GCNKernel) =
    throw(ArgumentError("No TFKernel implementation available for kernel subtype $(typeof(k))"))


struct TFIdentityKernel <: TFKernel end
TFKernel(:: IdentityKernel) = TFIdentityKernel()

"""
    Y = applyLayer(tfk :: TFKernel, XParts :: Vector{<: TensorFlow.Tensor})
    Y = applyLayer(tfk :: TFKernel, X :: TensorFlow.Tensor, weights :: Vector{TensorFlow.Variable})

Perform a layer operation, i.e. compute a TensorFlow tensor for the operation
`Y = \\sum_i K_i X_l Θ_il`. A `TensorFlowGCN` object calls the second form
of this function, which by default calls the first form with the argument
`XParts` holding the tensors for the `X_l Θ_il` products. `TFKernel` subtypes
must override one of these methods.
"""
applyLayer(tfk :: TFKernel, XParts :: Vector{<: tf.Tensor}) = sum(XParts)

applyLayer(tfk :: TFKernel, X :: tf.Tensor, weights :: Vector{tf.Variable}) =
    applyLayer(tfk, [X * Θ for Θ in weights])

"""
    input = transformInput(tfk :: TFKernel, input :: TensorFlow.Tensor)

Transform the input matrix before it is used as the input for the GCN. By
default, this just returns the input matrix. `TFKernel` subtypes for kernels
that require input transformation (e.g. kernels for reduced GCN) must override
this method.
"""
transformInput(tfk :: TFKernel, input :: tf.Tensor) = input

"""
    output = transformOutput(tfk :: TFKernel, output :: TensorFlow.Tensor)

Transform the output matrix before it is returned as the output of the GCN. By
default, this just returns the output matrix. `TFKernel` subtypes for kernels
that require output transformation (e.g. kernels for reduced GCN) must override
this method.
"""
transformOutput(tfk :: TFKernel, output :: tf.Tensor) = output

"""
    fillFeedDict(tfk :: TFKernel, dataset :: Dataset, dict :: Dict)

Extend an existing TensorFlow feed dictionary with values for the placeholders
used for the kernel object.
"""
fillFeedDict(tfk :: TFKernel, dataset :: Dataset, dict :: Dict) = nothing


### TensorFlow version of FixedMatrixKernel

mutable struct TFFixedMatrixKernel <: TFKernel
    kernel :: FixedMatrixKernel
    tensors :: Vector{tf.Tensor{Float64}}
end
TFKernel(kernel :: FixedMatrixKernel) = TFFixedMatrixKernel(kernel,
    [tf.placeholder(Float64, shape=[size(kernel.matrices[i])...], name="FixedMatrixKernel$i")
        for i in 1:length(kernel.matrices)])

function applyLayer(tfk :: TFFixedMatrixKernel, XParts :: Vector{tf.Tensor{Float64}})
    if length(XParts) == 1
        return tf.Ops.mat_mul(tfk.tensors[1], XParts[1])
    else
        return tf.Ops.add_n([tfk.tensors[i] * XParts[i] for i in 1:length(XParts)])
    end
end

function fillFeedDict(tfk :: TFFixedMatrixKernel, :: Dataset, dict :: Dict)
    for i = 1:length(tfk.tensors)
        dict[tfk.tensors[i]] = tfk.kernel.matrices[i]
    end
end

### TensorFlow version of FixedLowRankKernel

mutable struct TFFixedLowRankKernel <: TFKernel
    kernel :: FixedLowRankKernel
    projectorTensor :: tf.Tensor{Float64}
    diagTensors :: Vector{tf.Tensor{Float64}}
end
TFKernel(kernel :: FixedLowRankKernel) = TFFixedLowRankKernel(kernel,
    tf.placeholder(Float64, shape=[size(kernel.projector)...], name="FixedLowRankProjector"),
    [tf.placeholder(Float64, shape=[length(kernel.diagonals[i]), 1], name="FixedMatrixKernel$i")
        for i in 1:length(kernel.diagonals)])

function applyLayer(tfk :: TFFixedLowRankKernel, XParts :: Vector{tf.Tensor{Float64}})
    numParts = length(XParts)
    if tfk.kernel.isReduced
        parts = [tfk.diagTensors[i] .* XParts[i] for i = 1:numParts]
        return numParts == 1 ? parts[1] : tf.Ops.add_n(parts)
    else
        parts = [tfk.diagTensors[i] .* (tfk.projectorTensor' * XParts[i]) for i = 1:numParts]
        return tfk.projectorTensor * (numParts == 1 ? parts[1] : tf.Ops.add_n(parts))
    end
end

transformInput(tfk :: TFFixedLowRankKernel, input :: tf.Tensor) =
    tfk.kernel.isReduced ? tfk.projectorTensor' * input : input
transformOutput(tfk :: TFFixedLowRankKernel, output :: tf.Tensor) =
    tfk.kernel.isReduced ? tfk.projectorTensor * output : output

function fillFeedDict(tfk :: TFFixedLowRankKernel, dataset :: Dataset, dict :: Dict)
    dict[tfk.projectorTensor] = tfk.kernel.projector
    for i = 1:length(tfk.diagTensors)
        dict[tfk.diagTensors[i]] = hcat(tfk.kernel.diagonals[i])
    end
end

### TensorFlow version of PolyLaplacianKernel

"""
    TFPolyLaplacianKernel

`TFKernel` subtype for kernels of type `PolyLaplacianKernel`. Stores TensorFlow
placeholders for all the dense Kernel matrices.
"""
mutable struct TFPolyLaplacianKernel <: TFKernel
    kernel :: PolyLaplacianKernel
    tensors :: Vector{tf.Tensor}
end

TFKernel(kernel :: PolyLaplacianKernel) = TFPolyLaplacianKernel(
        kernel,
        [tf.placeholder(Float64,shape=[missing, missing], name="PolynomialKernel$i")
            for i in 1:numParts(kernel)])

function applyLayer(tfk :: TFPolyLaplacianKernel, XParts :: Vector{tf.Tensor{Float64}})
    if length(XParts) == 1
        return tf.Ops.mat_mul(tfk.tensors[1], XParts[1])
    else
        return tf.Ops.add_n([tf.Ops.mat_mul(tfk.tensors[i], XParts[i]) for i in 1:length(XParts)])
    end
end

function fillFeedDict(tfk :: TFPolyLaplacianKernel, dataset :: Dataset, dict :: Dict)
    L = getFullLaplacian(dataset.graph, tfk.kernel.smoother)
    X = UniformScaling(1.0)
    kernelParts = Any[c[1]*X for c in tfk.kernel.coeffs]
    for i = 2:maximum([length(c) for c in tfk.kernel.coeffs])
        X = L * X
        for j in 1:length(tfk.kernel.coeffs)
            if length(tfk.kernel.coeffs[j]) >= i
                kernelParts[j] += tfk.kernel.coeffs[j][i] * X
            end
        end
    end
    for i = 1:length(tfk.kernel.coeffs)
        dict[tfk.tensors[i]] = isa(kernelParts[i], UniformScaling) ?
            kernelParts[i].λ : kernelParts[i]
    end
end



### TensorFlow version of LowRankPolyLaplacianKernel

"""
    TFLowRankPolyLaplacianKernel

`TFKernel` subtype for kernels of type `LowRankPolyLaplacianKernel`. Stores
TensorFlow placeholders for the Laplacian eigenvector matrix U and the kernel
diagonals.
"""
mutable struct TFLowRankPolyLaplacianKernel <: TFKernel
    kernel :: LowRankPolyLaplacianKernel
    UTensor :: tf.Tensor
    kernelDiagTensors :: Vector{tf.Tensor}
end
function TFKernel(kernel :: LowRankPolyLaplacianKernel)
    TFLowRankPolyLaplacianKernel(kernel,
        tf.placeholder(Float64, shape=[missing, kernel.rank], name="U"),
        [tf.placeholder(Float64, shape=[kernel.rank, 1], name="KernelDiag$i")
            for i in 1:numParts(kernel)])
end

function applyLayer(tfk :: TFLowRankPolyLaplacianKernel, XParts :: Vector{tf.Tensor{Float64}})
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
transformInput(tfk :: TFLowRankPolyLaplacianKernel, input :: tf.Tensor) =
    tfk.kernel.isReduced ? tf.Ops.mat_mul(tfk.UTensor, input, transpose_a=true) : input
transformOutput(tfk :: TFLowRankPolyLaplacianKernel, output :: tf.Tensor) =
    tfk.kernel.isReduced ? tf.Ops.mat_mul(tfk.UTensor, output) : output

function fillFeedDict(tfk :: TFLowRankPolyLaplacianKernel, dataset :: Dataset, dict :: Dict)
    λ, U = getLaplacianEigenvalues(dataset.graph,
                tfk.kernel.rank, tfk.kernel.whichEV, tfk.kernel.smoother)
    dict[tfk.UTensor] = U
    for i = 1:numParts(tfk.kernel)
        d = zeros(tfk.kernel.rank)
        for j = 1:length(tfk.kernel.coeffs[i])
            d += tfk.kernel.coeffs[i][j] * λ.^(j-1)
        end
        dict[tfk.kernelDiagTensors[i]] = hcat(d)
    end
end


### TensorFlow version of LowRankInvLaplacianKernel

"""
    TFLowRankInvLaplacianKernel

`TFKernel` subtype for kernels of type `LowRankInvLaplacianKernel`. Stores
TensorFlow placeholders for the Laplacian eigenvector matrix U and the inverted
Laplacian eigenvalues.
"""
mutable struct TFLowRankInvLaplacianKernel <: TFKernel
    kernel :: LowRankInvLaplacianKernel
    UTensor :: tf.Tensor
    kernelDiagTensor :: tf.Tensor
end
function TFKernel(kernel :: LowRankInvLaplacianKernel)
    return TFLowRankInvLaplacianKernel(kernel,
        tf.placeholder(Float64, shape=[missing, kernel.rank], name="U"),
        tf.placeholder(Float64, shape=[kernel.rank, 1], name="kernelDiag"))
end

function applyLayer(tfk :: TFLowRankInvLaplacianKernel, XParts :: Vector{tf.Tensor{Float64}})
    if tfk.kernel.isReduced
        return tfk.kernelDiagTensor .* XParts[1]
    else
        return tfk.UTensor * (tfk.kernelDiagTensor .* (tfk.UTensor' * XParts[1]))
    end
end
transformInput(tfk :: TFLowRankInvLaplacianKernel, input :: tf.Tensor) =
    tfk.kernel.isReduced ? tf.Ops.mat_mul(tfk.UTensor, input, transpose_a=true) : input
transformOutput(tfk :: TFLowRankInvLaplacianKernel, output :: tf.Tensor) =
    tfk.kernel.isReduced ? tf.Ops.mat_mul(tfk.UTensor, output) : output

function fillFeedDict(tfk :: TFLowRankInvLaplacianKernel, dataset :: Dataset, dict :: Dict)
    λ, U = getLaplacianEigenvalues(dataset.graph,
                tfk.kernel.rank, :smallnonzero, tfk.kernel.smoother)
    dict[tfk.UTensor] = U
    dict[tfk.kernelDiagTensor] = hcat(minimum(λ) ./ λ)
end



"""
    Y = applyActivation(act :: Symbol, X :: TensorFlow.Tensor)

Apply the TensorFlow operation that performs the activation function specified
by the given Symbol.
"""
function applyActivation(act :: Symbol, X :: tf.Tensor)
    if act == :relu
        return tf.nn.relu(X)
    elseif act == :identity
        return X
    else
        throw(ArgumentError("Unknown activation specifier: $act"))
    end
end




### TensorFlow version of GCN

"""
    TensorFlowGCN

Data type that stores all data coming with the TensorFlow implementation of a
GCN architecture.

    TensorFlowGCN(arch :: GCNArchitecture; optimizer :: TensorFlow.train.Optimizer)

Constructs a `TensorFlowGCN` object based on a `GCNArchitecture` object. The
only additional information needed is the optimizer object, which by default
is an `AdamOptimizer` with a learning rate of 0.01.
"""
mutable struct TensorFlowGCN

    architecture :: GCNArchitecture

    tfKernel :: TFKernel

    features :: tf.Tensor
    labels :: tf.Tensor
    hiddenLayers :: Vector{tf.Tensor}
    weightVars :: Vector{Vector{tf.Variable}}
    output :: tf.Tensor
    classPrediction :: tf.Tensor
    loss :: tf.Tensor
    accuracy :: tf.Tensor

    optimizer :: tf.train.Optimizer
    optimizingOp :: tf.Tensor

    function TensorFlowGCN(arch :: GCNArchitecture;
            # optimizer = tf.train.AdamOptimizer(0.01) :: tf.train.Optimizer)
            optimizer = tf.train.GradientDescentOptimizer(0.2) :: tf.train.Optimizer)

        self = new(arch)

        numFeatures = arch.layerWidths[1]
        numLabels = arch.layerWidths[end]

        self.features = tf.placeholder(Float64, shape=[missing, numFeatures], name="Features")
        self.labels = tf.placeholder(Float64, shape=[missing, numLabels], name="Labels")

        self.tfKernel = TFKernel(arch.kernel)
        self.hiddenLayers = tf.Tensor[]
        self.weightVars = Vector{tf.Variable}[]

        X = self.features
        X = transformInput(self.tfKernel, X)
        push!(self.hiddenLayers, X)

        numKernelParts = numParts(arch.kernel)
        for i = 1:length(arch.layerWidths)-1
            weights = [
                    #tf.get_variable("Weight_layer$(i)_kernel$(j)",
                    #    shape=arch.layerWidths[i:i+1], dtype=Float64)
                    tf.Variable(
                        sqrt(6.0/sum(arch.layerWidths[i:i+1])) *
                            (1 .- 2*rand(Float64, arch.layerWidths[i], arch.layerWidths[i+1])),
                        name = "Weight_layer$(i)_kernel$(j)")
                for j = 1:numKernelParts]
            push!(self.weightVars, weights)

            X = applyLayer(self.tfKernel,
                [tf.Ops.mat_mul(X, Θ) for Θ in weights])
            if i < length(arch.layerWidths)-1
                X = applyActivation(arch.activation, X)
            end
            push!(self.hiddenLayers, X)
        end

        X = transformOutput(self.tfKernel, X)
        self.output = X

        normalizer = tf.reduce_sum(self.labels, axis=2) / tf.reduce_sum(self.labels)

        X = tf.nn.softmax_cross_entropy_with_logits(logits = self.output, labels = self.labels)
        X = tf.reduce_sum(X .* normalizer)

        self.loss = tf.add(X, arch.regParam * (numKernelParts == 1 ?
                tf.nn.l2_loss(self.weightVars[1][1]) :
                tf.Ops.add_n([tf.nn.l2_loss(Θ) for Θ in self.weightVars[1]])
            ))

        self.classPrediction = tf.arg_max(self.output, 2)
        trueClasses = tf.arg_max(self.labels, 2)
        correctness = tf.cast(tf.equal(self.classPrediction, trueClasses), Float64)
        self.accuracy = tf.reduce_sum(correctness .* normalizer, name="Accuracy")

        self.optimizer = optimizer
        self.optimizingOp = tf.train.minimize(self.optimizer, self.loss)

        return self
    end

end

Base.show(io :: IO, gcn :: TensorFlowGCN) = print(io,
    "$(typeof(gcn))(\"$(gcn.architecture.name)\" architecture, $(typeof(gcn.tfKernel)))")

"""
    initializeRandomWeights(gcn :: TensorFlowGCN, sess :: TensorFlow.Session)

Initializes the weight variables of the GCN in the given session with small
random values around zero.
This is necessary because of the current suboptimal implementation that does not
use TensorFlow's random initializers. Sessions where this function is not call
will all start with the same initial weights matrices, determined randomly at
the time of construction of the `TensorFlowGCN` object.
"""
function initializeRandomWeights(gcn :: TensorFlowGCN, sess :: tf.Session)
    for i = 1:length(gcn.weightVars)
        inputDim = gcn.architecture.layerWidths[i]
        outputDim = gcn.architecture.layerWidths[i+1]
        initRange = sqrt(6.0/(inputDim+outputDim))
        for Θ in gcn.weightVars[i]
            tf.run(sess, tf.assign(Θ, initRange * (1 .- 2*rand(Float64, inputDim, outputDim))))
        end
    end
end

"""
    exportWeights(gcn :: TensorFlowGCN, sess :: TensorFlow.Session)

Returns all current values of the GCN's weights variables in the given session.
Returned type is Vector{Vector{Matrix{Float64}}}.
"""
function exportWeights(gcn :: TensorFlowGCN, sess :: tf.Session)
    return [[tf.run(sess, θ) for θ in layerWeightVars]
                for layerWeightVars in gcn.weightVars]
end

"""
    importWeights(gcn :: TensorFlowGCN, sess :: TensorFlow.Session, weights :: Vector{Vector{<: Array{Float64}}})

Assign the given weight values to the GCN's  weight variables in the given
session.
"""
function importWeights(gcn :: TensorFlowGCN, sess :: tf.Session, weights :: Vector{Vector{<: Array{Float64}}})
    for i = 1:length(gcn.weightVars)
        for j = 1:length(gcn.weightVars[i])
            tf.run(sess, tf.assign(gcn.weightVars[i][j], weights[i][j]))
        end
    end
end

# function saveWeights(gcn :: TensorFlowGCN, sess :: tf.Session, file :: HDF5File, name :: String)
#     for i = 1:length(gcn.weightVars)
#         for j = 1:length(gcn.weightVars[i])
#             write(file, "$name/layer$i/weight$j", tf.run(sess, gcn.weightVars[i][j]))
#         end
#     end
# end
# function loadWeights(gcn :: TensorFlowGCN, sess :: tf.Session, file :: HDF5File, name :: String)
#     for i = 1:length(gcn.weightVars)
#         for j = 1:length(gcn.weightVars[i])
#             tf.run(sess, tf.assign(gcn.weightVars[i][j], read(file, "$name/layer$i/weight$j")))
#         end
#     end
# end

"""
    (feedDictTest, feedDictTrain) = createFeedDicts(gcn :: TensorFlowGCN, dataset :: Dataset)

Construct two feed dictionaries that can be used with `TensorFlow.run` to
evaluate the target tensors of the GCN. The two dictionaries differ in the
label information made available. The first dictionary only uses the labels of
the training nodes, and the second only uses those of the test nodes of the
dataset. Also see `getTrainLabels` and `getTestLabels`.

Note that in a typical workflow, calling this function is the first time that
any large matrices are constructed.
"""
function createFeedDicts(gcn :: TensorFlowGCN, dataset :: Dataset)
    @assert(dataset.numFeatures == gcn.architecture.layerWidths[1])
    @assert(dataset.numLabels == gcn.architecture.layerWidths[end])

    feedDictTrain = Dict{tf.Tensor, Any}(
        gcn.features => dataset.features)
    fillFeedDict(gcn.tfKernel, dataset, feedDictTrain)

    feedDictTest = copy(feedDictTrain)
    feedDictTest[gcn.labels] = getTestLabels(dataset)
    feedDictTrain[gcn.labels] = getTrainLabels(dataset)

    return feedDictTrain, feedDictTest
end

"""
    (sess, feedDictTest, setupTime, trainingTime) =
        createTrainedSession(gcn :: TensorFlowGCN, dataset :: Dataset, numIter :: Int)

Create a TensorFlow.Session object, initialize all variables in that session,
construct the feed dictionaries, and run `numIter` training iterations.
Returns a tuple holding the session object, the feed dictionary for testing,
the time required for setup (i.e. the call to `createFeedDicts`), and the total
time for the training iterations.
"""
function createTrainedSession(gcn :: TensorFlowGCN, dataset :: Dataset, numIter :: Int)

    sess = tf.Session()
    tf.run(sess, tf.global_variables_initializer())

    initializeRandomWeights(gcn, sess)

    local feedDictTrain, feedDictTest
    setupTime = @elapsed begin
        feedDictTrain, feedDictTest = createFeedDicts(gcn, dataset)
    end

    trainingTime = @elapsed begin
        for iter = 1:numIter
            tf.run(sess, gcn.optimizingOp, feedDictTrain)
        end
    end

    return sess, feedDictTest, setupTime, trainingTime
end
