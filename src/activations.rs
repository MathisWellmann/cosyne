#[derive(Debug, Clone)]
pub enum Activation{
    Relu,
    SoftSign,
    SoftPlus,
    Tanh,
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        return match self {
            Self::Relu => relu(x),
            Self::SoftPlus => soft_plus(x),
            Self::SoftSign => soft_sign(x),
            Self::Tanh => x.tanh(),
        }
    }
}

pub fn relu(x: f64) -> f64 {
    return if x <= 0.0 {
        0.0
    } else {
        x
    }
}

pub fn relu_deriv(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        1.0
    }
}

pub fn soft_plus(x: f64) -> f64 {
    (1.0 + std::f64::consts::E.powf(x)).ln()
}

pub fn soft_plus_deriv(x: f64) -> f64 {
    1.0 / (1.0 + std::f64::consts::E.powf(-x))
}

pub fn soft_sign(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

pub fn soft_sign_deriv(x: f64) -> f64 {
    1.0 / (1.0 + x.abs()).powi(2)
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn tanh_deriv(x: f64) -> f64 {
    1.0 - (x.tanh()).powi(2)
}