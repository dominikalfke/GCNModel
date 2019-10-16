

using JLD
using Printf

export
    Experiment,
    getAverageAccuracy,
    addRuns,
    clearResults,
    saveInJLD,
    printSummary,
    TrainingSetRandomizer,
    UniformTrainingSetRandomizer


"""
    Data type for a semi-supervised node classification experiment with a
    (possibly low-rank) graph convolutional network. Objects store the
    experimental setup as well as the results of test runs performed via the
    addRuns method.

    `datasetFile` - String holding the file name of the JLD file where the
    dataset is stored as a variable named "dataset".

"""
mutable struct Experiment

    datasetFile :: String

    architecture :: GCNArchitecture

    numTrainingIter :: Int64

    randomizer

    numRuns :: Int64

    accuracyResults :: Vector{Float64}
    trainingTimes :: Vector{Float64}
    setupTimes :: Vector{Float64}

    function Experiment(datasetFile :: String, arch :: GCNArchitecture,
                numTrainingIter :: Int64; randomizer = nothing)
        return new(datasetFile, arch, numTrainingIter, randomizer,
                0, Float64[], Float64[], Float64[])
    end
end

Base.show(io :: IO, exp :: Experiment) = print(io,
    "$(typeof(exp))($(exp.datasetFile), \"$(exp.architecture.name)\", $(exp.numTrainingIter) training iterations, $(exp.numRuns) runs done)")

"""
    getAverageAccuracy(exp :: Experiment)

Returns the mean of the accuracy results of all runs stored in the experiment
object as Float64 between 0.0 and 1.0, or NaN if no runs have yet been
conducted.
"""
getAverageAccuracy(exp :: Experiment) =
    exp.numRuns == 0 ? NaN : sum(exp.accuracyResults) / exp.numRuns


function addRunResults!(exp :: Experiment, accuracy :: Float64, setupTime :: Float64, trainingTime :: Float64)
    exp.numRuns += 1
    push!(exp.accuracyResults, accuracy)
    push!(exp.setupTimes, setupTime)
    push!(exp.trainingTimes, trainingTime)
end


"""
    clearResults(exp :: Experiment)

Remove all results of previous runs from an experiment object.
"""
function clearResults(exp :: Experiment)
    exp.numRuns = 0
    exp.accuracyResults = Float64[]
    exp.setupTimes = Float64[]
    exp.trainingTimes = Float64[]
end

"""
    saveInJLD(exp :: Experiment, filename :: String, expName = "" :: String)

Open or create an JLD file and store an experiment object and a short summary
in it under a given name, and add this name to a list of experiment names
stored in the file.
"""
function saveInJLD(exp :: Experiment, filename :: String, expName :: String = "")
    # jldopen(filename, "a") do file
    jldopen(filename, false, true, true, false, true) do file
        if isempty(expName)
            i = 0
            expName = "experiment"
            while haskey(file, expName)
                i += 1
                expName = "experiment$i"
            end
        end
        write(file, expName, exp)
        write(file, "summary_$expName/accuracy", sum(exp.accuracyResults)/exp.numRuns)
        write(file, "summary_$expName/setupTime", sum(exp.setupTimes)/exp.numRuns)
        write(file, "summary_$expName/trainingTime", sum(exp.trainingTimes)/exp.numRuns)
    end
end

"""
    printSummary(exp :: Experiment)

Print a summary of the experiment results.
"""
function printSummary(exp :: Experiment)
    N = exp.numRuns
    if N == 0
        println("% $(exp.architecture.name): no results")
        return
    end
    meanSetupTime = sum(exp.setupTimes)/N
    meanTrainingTime = sum(exp.trainingTimes)/N
    meanAccuracy = sum(exp.accuracyResults)/N
    if N == 1
        println("Results from a single experiment run with architecture \"$(exp.architecture.name)\":")
        @printf(" - Accuracy: %.4f%%\n", 100*meanAccuracy)
        @printf(" - Setup time: %.2fs\n", meanSetupTime)
        @printf(" - Training time: %.2fs\n", meanTrainingTime)
    else
        println("Average results from $N experiment runs with architecture \"$(exp.architecture.name)\":")
        @printf(" - Accuracy: %.4f%% ± %.4f\n", 100*meanAccuracy,
            100*sqrt(sum((exp.accuracyResults .- meanAccuracy).^2)/(N-1)))
        @printf(" - Setup time: %.2fs ± %.2f\n",
            meanSetupTime, sqrt(sum((exp.setupTimes .- meanSetupTime).^2)/(N-1)))
        @printf(" - Training time: %.2fs ± %.2f\n",
            meanTrainingTime, sqrt(sum((exp.trainingTimes .- meanTrainingTime).^2)/(N-1)))
    end
end

"""
    TrainingSetRandomizer

Functor struct that can be passed to the `Experiment` constructor as the
`randomizer` keyword argument. Before each run, a new set of training nodes
will be chosen completely randomly via the `randomizeTrainingSet!` function.
"""
mutable struct TrainingSetRandomizer
    numTrainingNodes :: Int64
end
(r :: TrainingSetRandomizer)(dataset :: Dataset) =
    randomizeTrainingSet!(dataset, r.numTrainingNodes)

"""
    UniformTrainingSetRandomizer

Functor struct that can be passed to the `Experiment` constructor as the
`randomizer` keyword argument. Before each run, a new set of training nodes
will be chosen, where the number of nodes from each class is fixed. The
`randomizeUniformTrainingSet!` function is called.
"""
mutable struct UniformTrainingSetRandomizer
    numTrainingNodesPerClass :: Int64
end
(r :: UniformTrainingSetRandomizer)(dataset :: Dataset) =
    randomizeUniformTrainingSet!(dataset, r.numTrainingNodesPerClass)
