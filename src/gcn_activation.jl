

export
    Activation,
    NoActivation,
    Relu,
    ReducedRelu

abstract type Activation end


struct NoActivation <: Activation end

struct Relu <: Activation end

struct ReducedRelu <: Activation end
