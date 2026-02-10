// reciprocal, relu, sign, log, exp, neg
use pyo3::prelude::*;

// mod utils;
// use utils::utils::{gen_idx, gen_stride};

#[pyfunction]
pub fn sign(x: Vec<f32>, stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape = out_shape.iter().product();
    let mut ret = vec![0.; flat_shape];
    let flat_stride = gen_stride(&out_shape);
    for i in 0..flat_shape {
        let x_idx = gen_idx(i, &flat_stride, &stride, &out_shape);
        ret[i] = x[x_idx].signum();
    }
    return ret
}

#[pyfunction]
pub fn log(x: Vec<f32>, stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape = out_shape.iter().product();
    let mut ret = vec![0.; flat_shape];
    let flat_stride = gen_stride(&out_shape);
    for i in 0..flat_shape {
        let x_idx = gen_idx(i, &flat_stride, &stride, &out_shape);
        ret[i] = x[x_idx].ln(); // (x.abs + 1).ln
    }
    return ret
}

#[pyfunction]
pub fn exp(x: Vec<f32>, stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape = out_shape.iter().product();
    let mut ret = vec![0.; flat_shape];
    let flat_stride = gen_stride(&out_shape);
    for i in 0..flat_shape {
        let x_idx = gen_idx(i, &flat_stride, &stride, &out_shape);
        ret[i] = x[x_idx].exp();
    }
    return ret
}

#[pyfunction]
pub fn relu(x: Vec<f32>, stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape = out_shape.iter().product();
    let mut ret = vec![0.; flat_shape];
    let flat_stride = gen_stride(&out_shape);
    for i in 0..flat_shape {
        let x_idx = gen_idx(i, &flat_stride, &stride, &out_shape);
        ret[i] = x[x_idx].max(0.0);
    }
    return ret
}

#[pyfunction]
pub fn neg(x: Vec<f32>, stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape = out_shape.iter().product();
    let mut ret = vec![0.; flat_shape];
    let flat_stride = gen_stride(&out_shape);
    for i in 0..flat_shape {
        let x_idx = gen_idx(i, &flat_stride, &stride, &out_shape);
        ret[i] = -x[x_idx];
    }
    return ret
}

#[pyfunction]
pub fn reciprocal(x: Vec<f32>, stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape = out_shape.iter().product();
    let mut ret = vec![0.; flat_shape];
    let flat_stride = gen_stride(&out_shape);
    for i in 0..flat_shape {
        let x_idx = gen_idx(i, &flat_stride, &stride, &out_shape);
        ret[i] = 1.0/x[x_idx];
    }
    return ret
}

// generate the stride of given shape, contiguous view.
fn gen_stride(shape: &Vec<usize>) -> Vec<usize>{
    let ndim = shape.len();
    let mut stride = vec![1; ndim];

    for i in (0..ndim-1).rev() {
        stride[i] = stride[i+1] * shape[i+1];
    }
    return stride
}

// generate flat idx -> multi-dim idx -> flat idx
fn gen_idx(g_idx:usize,flat_stride:&Vec<usize>,stride:&Vec<usize>,shape:&Vec<usize>)->usize{
    let ndim = shape.len();
    let mut idx = 0;
    for i in 0..ndim {
        let d_idx = (g_idx / flat_stride[i]) % shape[i];
        idx += d_idx * stride[i];
    }
    return idx
}