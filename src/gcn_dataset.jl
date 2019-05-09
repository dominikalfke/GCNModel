

# using JLD

import Random.randcycle
import LinearAlgebra.Diagonal
import Arpack.eigs

export Dataset,
    getMaskedLabels,
    getTrainLabels,
    getTestLabels,
    getValidationLabels,
    connectedComponents,
    largestConnectedSubDataset,
    randomizeIndices!,
    randomizeUniformIndices!

"""
    Dataset

Data type that can store a graph-based dataset for semi-supervised node
classification.

`name`: Custom String describing the dataset.

`graph`: Object of a subtype of `AbstractGraph` describing node adjacency.

`features`: Feature matrix. Entry (i,j) gives the value for feature #j in node #i.

`labels`: Label matrix. Entry (i,j) gives the (typically one-hot) value for label #j in node #i.

`testInd`: List of node indices for testing results.

`trainInd`: List of node indices for training.

`validInd`: List of node indices for model validation.

`numNodes`: Number of data set entries, i.e. nodes in the graph.

`numFeatures`: Number of features for each node.

`numLabels`: Number of labels for each node.
"""
mutable struct Dataset

    name :: String

    graph :: AbstractGraph

    features :: Matrix{Float64}
    labels :: Matrix{Float64}

    testInd :: Vector{Int64}
    trainInd :: Vector{Int64}
    validInd :: Vector{Int64}

    numFeatures :: Int64
    numLabels :: Int64
    numNodes :: Int64

    function Dataset(name :: String, graph :: AbstractGraph,
            features :: Matrix{Float64}, labels :: Matrix{Float64},
            testInd :: Vector{Int64}, trainInd :: Vector{Int64}, validInd :: Vector{Int64})

        self = new(name, graph, features, labels,
                testInd, trainInd, validInd)

        self.numNodes = size(self.features, 1)
        @assert getNumNodes(graph) == self.numNodes

        self.numFeatures = size(self.features, 2)
        self.numLabels = size(self.labels, 2)

        return self
    end
end



"""
    getMaskedLabels(d :: Dataset, indices)

Returns the dataset's labels matrix where all rows are replaced by zeros, except
for those rows specified in `indices`, which can be a list of indices or a
boolean index vector.
"""
function getMaskedLabels(d :: Dataset, indices)
    labels = zeros(d.numNodes, d.numLabels)
    labels[indices, :] = d.labels[indices, :]
    return labels
end

getTrainLabels(d :: Dataset) = getMaskedLabels(d, d.trainInd)
getTestLabels(d :: Dataset) = getMaskedLabels(d, d.testInd)
getValidationLabels(d :: Dataset) = getMaskedLabels(d, d.validInd)


Base.show(io :: IO, d :: Dataset) =
    # print(io, "Dataset \"$(d.name)\" with $(d.numNodes) nodes, $(d.numFeatures) features, and $(d.numLabels) labels")
    print(io, "Dataset(\"$(d.name)\", ",
            "$(d.numNodes) nodes ($(length(d.trainInd))+$(length(d.testInd))+$(length(d.validInd))), ",
            "$(d.numFeatures) features, and $(d.numLabels) labels)")

"""
    connectedComponents(adj)

Compute the connected components of a graph given by its adjacency matrix.
Returns a tuple `(k, compSizes, comp)`, where `k` is the number of connected
components, `compSizes` is a vector of length `k` holding the number of nodes in
each components, and `comp` is a vector holding the component index of each
node.
"""
function connectedComponents(adj)
    n = size(adj, 1)
    comp = zeros(Int64, n)

    function colorNode(i, color)
        if comp[i] == 0
            comp[i] = color
            for j = 1:n
                if adj[i, j] > 0
                    colorNode(j, color)
                end
            end
        end
    end

    k = 0
    for i = 1:n
        if comp[i] == 0
            k += 1
            colorNode(i, k)
        end
    end

    compSizes = zeros(Int64, k)
    for i = 1:k
        compSizes[i] = sum(comp .== i)
    end

    return k, compSizes, comp
end

"""
    largestConnectedSubDataset(d :: Dataset)
    largestConnectedSubDataset(d :: Dataset, newName :: String)

Create a dataset from a given one by deleting all nodes but those belonging to
the largest connected component of the graph.
"""
function largestConnectedSubDataset(d :: Dataset, newName = d.name * "_largestComp" :: String)

    numComps, compSizes, compInd = connectedComponents(getAdjacency(d.graph))

    if numComps == 1
        return d
    end

    k = argmax(compSizes)
    doKeep = (compInd .== k)

    print("$(numComps) components, maximum size: $(compSizes[k])\n")

    features = d.features[doKeep, :]
    labels = d.labels[doKeep, :]
    graph = getSubGraph(d.graph, findall(doKeep))

    function transformInd(ind, doKeep)
        logical = falses(d.numNodes)
        for i in ind
            logical[i] = true
        end
        return findall(logical[doKeep])
    end

    return Dataset(newName, graph, features, labels,
        transformInd(d.testInd, doKeep),
        transformInd(d.trainInd, doKeep),
        transformInd(d.validInd, doKeep))
end

"""
    randomizeIndices!(dataset :: Dataset)
    randomizeIndices!(dataset :: Dataset, numTrainInd :: Int64)

Sets the `testInd`, `trainInd`, and `validInd` fields of a dataset object so
that the training set is made up of `n` random nodes. All remaining nodes will
be used as test nodes and there will be no validation nodes. The number `n` can
be given by the second argument `numTrainInd`. If omitted, the former number of
training nodes will be maintained.
"""
function randomizeIndices!(dataset :: Dataset, numTrainInd = length(dataset.trainInd):: Int64)
    cycle = randcycle(dataset.numNodes)
    dataset.trainInd = cycle[1:numTrainInd]
    dataset.testInd = cycle[numTrainInd+1:end]
    dataset.validInd = Int64[]
    return dataset
end

function randomizeUniformIndices!(dataset :: Dataset, numTrainIndPerClass :: Int64)
    dataset.trainInd = Int64[]
    dataset.testInd = Int64[]
    dataset.validInd = Int64[]
    for i = 1:dataset.numLabels
        ind = findall(dataset.labels[:,i][:] .!= 0)
        cycle = randcycle(length(ind))
        append!(dataset.trainInd, ind[cycle[1:numTrainIndPerClass]])
        append!(dataset.testInd, ind[cycle[numTrainIndPerClass+1:end]])
    end
    return dataset
end
