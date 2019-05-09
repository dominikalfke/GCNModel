

using JLD

export
    Experiment,
    getAverageAccuracy,
    addRuns,
    clearResults,
    saveInJLD

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

    randomizedTrainingSize :: Int64

    numRuns :: Int64

    accuracyResults :: Vector{Float64}
    trainingTimes :: Vector{Float64}
    setupTimes :: Vector{Float64}

    function Experiment(datasetFile :: String, arch :: GCNArchitecture,
                numTrainingIter :: Int64, randomizedTrainingSize = 0 :: Int64)
        return new(datasetFile, arch, numTrainingIter, randomizedTrainingSize,
                0, Float64[], Float64[], Float64[])
    end
    # function Experiment(file :: HDF5File, name :: String)
    #     exp = new()
    #     for field in fieldnames(Experiment)
    #         if field == :architecture
    #             exp.architecture = GCNArchitecture(file, "$name/architecture")
    #         else
    #             setproperty!(exp, field, read(file, "$name/$field"))
    #         end
    #     end
    #     return exp
    # end
end

Base.show(io :: IO, exp :: Experiment) = print(io,
    "$(typeof(exp))($(exp.datasetFile), \"$(exp.architecture.name)\", $(exp.numTrainingIter) training iterations, $(exp.numRuns) runs done)")

# function Base.write(file :: HDF5File, name :: String, exp :: Experiment)
#     for field in fieldnames(Experiment)
#         write(file, "$name/$field", getfield(exp, field))
#     end
# end

"""
    getAverageAccuracy(exp :: Experiment)

Returns the mean of the accuracy results of all runs stored in the experiment
object as Float64 between 0.0 and 1.0, or NaN if no runs have yet been
conducted.
"""
getAverageAccuracy(exp :: Experiment) =
    exp.numRuns == 0 ? NaN : sum(exp.accuracyResults) / exp.numRuns


"""
    addRuns(exp :: Experiment, numRuns :: Int; printInterval = 0 :: Int64)

Perform a number of test runs of an experiment and store the results in the
experiment object. If `printInterval` is nonzero, accuracy summaries are printed
every few runs.
"""
function addRuns(exp :: Experiment, numRuns :: Int; printInterval = 0 :: Int64)

    @load exp.datasetFile dataset

    lastPrintRun = exp.numRuns

    for incRun = 1:numRuns

        tf.set_def_graph(tf.Graph())

        gcn = TensorFlowGCN(exp.architecture)

        if exp.randomizedTrainingSize > 0
            randomizeIndices!(dataset, exp.randomizedTrainingSize)
        end

        sess, feedDictTest, setupTime, trainingTime =
            createTrainedSession(gcn, dataset, exp.numTrainingIter)

        accuracy = tf.run(sess, gcn.accuracy, feedDictTest)

        close(sess)

        exp.numRuns += 1
        push!(exp.accuracyResults, accuracy)
        push!(exp.setupTimes, setupTime)
        push!(exp.trainingTimes, trainingTime)

        if printInterval == 1
            println("Run $(exp.numRuns))/$numRuns: Accuracy $(100*accuracy)%")
        elseif printInterval > 0 && (incRun % printInterval == 0 || incRun == numRuns)
            println("Runs $(lastPrintRun+1)-$(exp.numRuns)/$numRuns: Average accuracy $(100*sum(exp.accuracyResults[lastPrintRun+1:end])/(exp.numRuns-lastPrintRun))%")
            lastPrintRun = incRun
        end

    end
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
function saveInJLD(exp :: Experiment, filename :: String, expName = "" :: String)
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
