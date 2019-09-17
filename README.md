# GCNModel
Julia implementation of Graph Convolutional Networks as used in the pre-print to [Alfke, Stoll, **Semi-Supervised Classification on Non-Sparse Graphs Using Low-Rank Graph Convolutional Networks** (2019)](https://arxiv.org/abs/1905.10224).

## Installation
This module requires the [``TensorFlow.jl``](https://github.com/malmaud/TensorFlow.jl) package. Before laying hands on ``GCNModel``, it is best to install, build, and test the TensorFlow module first.

Since the ``GCNModel`` package is not currently available in Julia's package repository, you have to make it available to your Julia installation. One way to do this is to activate the ``GCNModel`` environment each time you start Julia before using the module. Type ``]`` in Julia to open the package command line, followed by ``activate path/to/GCNModel``. From then on, ``using GCNModel`` should work.

## Examples

For usage examples, see the [LowRankGCNExperiments](https://github.com/dominikalfke/LowRankGCNExperiments) repository.

## Dataset representation
The ``Dataset`` data type is designed for semi-supervised node classification on graphs. It holds the ``features`` and ``labels`` matrix, index vectors holding the ``trainingSet``, ``testSet``, and ``validationSet`` (though the latter is never used in our setup), as well as an object of an ``AbstractGraph`` subtype. Notable subtypes are ``Hypergraph`` and ``FullGaussianGraph``. Additional, untested subtypes are ``AdjacencyGraph``, ``SparseAdjacencyGraph``, and ``EdgeListGraph``, which remain for future work.

## Architecture representation
The ``GCNArchitecture`` data type is designed to set up and store the meta-information about the GCN architecture, without requiring any system matrices. Its main fields are the ``layerWidths`` vector holding the widths of all layers, which were called N~0~, ..., N~L~ in our paper, and an object of a subtype of the the abstract ``GCNKernel`` type. These kernel objects describe how matrix multiplications with the kernel matrices will be evaluated. Available subtypes are:
* ``FixedMatrixKernel`` -- for kernel matrices given from external sources.
* ``FixedLowRankKernel`` -- for kernel matrices in low-rank form given from external sources.
* ``PolyLaplacianKernel`` -- for a polynomial filter space. The Laplacian will be set up from the dataset at run time, and the polynomial matrices will be explicitly set up.
* ``LowRankPolyLaplacianKernel`` -- for low-rank approximations to a polynomial filter space. The Laplacian will be set up, its dominant eigenvalues will be computed, and the diagonal filter matrices will be set up all at run time. This sets up the full Laplacian, but not the full kernel matrices.
* ``LowRankInvLaplacianKernel`` -- for a one-dimensional filter space spanned by the low-rank approximation to the pseudoinverse filter function. The Laplacian will be set up, its dominant eigenvalues will be computed, and the diagonal filter matrix will be set up all at run time. This sets up the full Laplacian, but not the full kernel matrices.

For hypergraph datasets, additional subtypes exist:
* ``PolyHypergraphLaplacianKernel`` -- for a polynomial filter space, evaluated efficiently using the techniques from Sections A3.1.1 and A3.1.2 of the appendix.
* ``InvHypergraphLaplacianKernel`` -- for a one-dimensional filter space spanned by the full-rank pseudoinverse filter function, using techniques from Section A3.1.3 of the appendix.
* ``LowRankPolyHypergraphLaplacianKernel`` and ``LowRankInvHypergraphLaplacianKernel`` -- like ``LowRankPolyLaplacianKernel`` and ``LowRankInvLaplacianKernel``, but with more efficient eigenvalue computation as in Section A3.1.4 of the appendix.
*  ``PolySmoothedHypergraphLaplacianKernel`` -- for a polynomial filter space with non-hypergraph smoothing, evaluated efficiently using the techniques from Section A3.3 of the appendix.

## TensorFlow Model
If ``architecture`` is an object of type ``GCNArchitecture``, then ``gcn = TensorFlowGCN(architecture)`` automatically creates the TensorFlow model. The ``createTrainedSession`` function creates a TensorFlow session and a feed dictionary for future test evaluations. The ``gcn.output`` Tensor generates the output matrix ``Y``. The ``gcn.accuracy`` Tensor evaluates the class prediction accuracy with respect to the true labels in the passed feed dictionary.

## Automatic experiment workflow
The ``Experiment`` data type is designed to hold all setup information and run results of an experiment. It contains the architecture object and the filename of the ``JLD`` file storing the dataset, but it does not store any system matrices or TensorFlow objects itself, so it can easily be saved in a file. The ``addRuns`` function implements the training process described in detail in Section A4.3 of the appendix. The setup time, training time, and accuracy results of each run are stored directly in the experiment object. The ``printSummary`` function prints the average results in the Julia console. The ``saveInJLD`` function adds the experiment object as well as a short result summary to a ``JLD`` file.

## Custom kernel types
Information on how to write custom kernel types will be added at a later time.
