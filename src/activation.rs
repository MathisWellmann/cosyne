/// Represents which transfer function to use for evaluating neural networks.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Activation {
    /// Maps input to output directly, as if there is no transfer function.
    Linear,
    /// Outputs 1 if input is greater than 0, 0 otherwise.
    Threshold,
    /// Outputs 1 if input is greater than 0, 0 if input is equal to 0, -1 otherwise. Useful
    /// for simple problems and boolean logic, as it only allows three possible output values.
    Sign,
    /// A non-linear function. This function is the most general, so it should be defaulted to.
    Sigmoid,
    /// Tanh activation function with output range [-1.0, 1.0]
    Tanh,
    /// SoftSign activation function
    SoftSign,
    /// BentIdentity activation function
    BentIdentity,
    /// Rectified linear unit activation function
    Relu,
}

impl Activation {
    /// Return the corresponding function to the Activation
    #[inline(always)]
    pub fn get_func(&self) -> fn(f64) -> f64 {
        match self {
            Activation::Linear => linear,
            Activation::Threshold => threshold,
            Activation::Sign => sign,
            Activation::Sigmoid => sigmoid,
            Activation::Tanh => tanh,
            Activation::SoftSign => soft_sign,
            Activation::BentIdentity => bent_identity,
            Activation::Relu => relu,
        }
    }

    /// parse the Activation from an int 32 value
    #[inline(always)]
    pub fn from_i32(n: i32) -> Activation {
        match n {
            0 => Activation::Linear,
            1 => Activation::Threshold,
            2 => Activation::Sign,
            3 => Activation::Sigmoid,
            4 => Activation::Tanh,
            5 => Activation::SoftSign,
            6 => Activation::BentIdentity,
            _ => Activation::Relu,
        }
    }
}

#[inline(always)]
pub fn linear(x: f64) -> f64 {
    x
}

#[inline(always)]
pub fn threshold(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

#[inline(always)]
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x == 0.0 {
        0.0
    } else {
        -1.0
    }
}

#[inline(always)]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

#[inline(always)]
pub fn soft_sign(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

#[inline(always)]
pub fn bent_identity(x: f64) -> f64 {
    (((x.powi(2) + 1.0).sqrt() - 1.0) / 2.0) + x
}

/// rectified linear unit
#[inline(always)]
pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}
