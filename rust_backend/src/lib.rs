use pyo3::prelude::*;

mod binary_op;
mod reduce_op;
mod unary_op;
mod movement_op;
use binary_op::{add, mul, div, pow, sub, cmpeq};
use unary_op::{relu, log, exp, neg, reciprocal, sign};
use movement_op::{shrink, pad, flip, masked_fill};
use reduce_op::{sum, max};

#[pyfunction]
fn noop(x: Vec<f32>, stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape = out_shape.iter().product();
    let ndim = out_shape.len();
    let mut ret = vec![0.; flat_shape];

    let mut flat_stride = vec![1; ndim];
    for i in (0..ndim-1).rev() {
        flat_stride[i] = flat_stride[i+1] * out_shape[i+1];
    }

    for j in 0..flat_shape {
        let mut x_idx = 0;
        for k in 0..ndim {
            let idx = (j / flat_stride[k]) % out_shape[k];
            x_idx += idx * stride[k]
        }
        ret[j] = x[x_idx];
    }
    return ret
}

#[pymodule]
fn rust_backend(m: &Bound<'_, PyModule>)-> PyResult<()>{
    m.add_function(wrap_pyfunction!(mul,m)?)?;
    m.add_function(wrap_pyfunction!(add,m)?)?;
    m.add_function(wrap_pyfunction!(div,m)?)?;
    m.add_function(wrap_pyfunction!(pow,m)?)?;
    m.add_function(wrap_pyfunction!(sub,m)?)?;
    m.add_function(wrap_pyfunction!(cmpeq,m)?)?;

    m.add_function(wrap_pyfunction!(relu,m)?)?;
    m.add_function(wrap_pyfunction!(log,m)?)?;
    m.add_function(wrap_pyfunction!(exp,m)?)?;
    m.add_function(wrap_pyfunction!(reciprocal,m)?)?;
    m.add_function(wrap_pyfunction!(neg,m)?)?;
    m.add_function(wrap_pyfunction!(sign,m)?)?;

    m.add_function(wrap_pyfunction!(sum,m)?)?;
    m.add_function(wrap_pyfunction!(max,m)?)?;

    m.add_function(wrap_pyfunction!(shrink,m)?)?;
    m.add_function(wrap_pyfunction!(pad,m)?)?;
    m.add_function(wrap_pyfunction!(flip,m)?)?;
    m.add_function(wrap_pyfunction!(masked_fill,m)?)?;

    m.add_function(wrap_pyfunction!(noop,m)?)?;
    Ok(())
}


// /// A Python module implemented in Rust.
// #[pymodule]
// mod rust_backend {
//     use pyo3::prelude::*;
//     /// Formats the sum of two numbers as string.
//     #[pyfunction]
//     fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//         Ok((a + b).to_string())
//     }

// }
