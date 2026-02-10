use pyo3::prelude::*;

// add, mul, div, pow, cmpeq
#[pyfunction]
pub fn add(x: Vec<f32>, y: Vec<f32>, x_stride: Vec<usize>, 
        y_stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape: usize = out_shape.iter().product();
    let mut ret = vec![0.; flat_shape];
    let ndim = out_shape.len();

    let mut flat_stride = vec![1; ndim];
    for k in (0..ndim-1).rev(){
        flat_stride[k] = flat_stride[k+1] * out_shape[k+1];
    }

    for i in 0..flat_shape{
        let mut x_idx = 0;
        let mut y_idx = 0;
        for j in 0..ndim{
            // flat idx to multi-dim idx
            let idx = (i / flat_stride[j]) % out_shape[j];
            x_idx += idx * x_stride[j];
            y_idx += idx * y_stride[j];
        }
        ret[i] = x[x_idx] + y[y_idx];
    }
    return ret;
}

#[pyfunction]
pub fn mul(x: Vec<f32>, y: Vec<f32>, x_stride: Vec<usize>,
        y_stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape: usize = out_shape.iter().product();
    let mut ret = vec![0.; flat_shape];
    let ndim = out_shape.len();

    let mut flat_stride = vec![1; ndim];
    for k in (0..ndim-1).rev(){
        flat_stride[k] = flat_stride[k+1] * out_shape[k+1];
    }

    for i in 0..flat_shape{
        let mut x_idx = 0;
        let mut y_idx = 0;
        for j in 0..ndim{
            let idx = (i / flat_stride[j]) % out_shape[j];
            x_idx += idx * x_stride[j];
            y_idx += idx * y_stride[j];
        }
        ret[i] = x[x_idx] * y[y_idx];
    }
    return ret;
}

#[pyfunction]
pub fn div(x: Vec<f32>, y: Vec<f32>, x_stride: Vec<usize>,
        y_stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape: usize = out_shape.iter().product();
    let mut ret = vec![0.; flat_shape];
    let ndim = out_shape.len();

    let mut flat_stride = vec![1; ndim];
    for k in (0..ndim-1).rev(){
        flat_stride[k] = flat_stride[k+1] * out_shape[k+1];
    }

    for i in 0..flat_shape{
        let mut x_idx = 0;
        let mut y_idx = 0;
        for j in 0..ndim{
            let idx = (i / flat_stride[j]) % out_shape[j];
            x_idx += idx * x_stride[j];
            y_idx += idx * y_stride[j];
        }
        ret[i] = x[x_idx] / y[y_idx];
    }
    return ret;
}

#[pyfunction]
pub fn pow(x: Vec<f32>, y: Vec<f32>, x_stride: Vec<usize>,
        y_stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape: usize = out_shape.iter().product();
    let mut ret = vec![0.; flat_shape];
    let ndim = out_shape.len();

    let mut flat_stride = vec![1; ndim];
    for k in (0..ndim-1).rev(){
        flat_stride[k] = flat_stride[k+1] * out_shape[k+1];
    }

    for i in 0..flat_shape{
        let mut x_idx = 0;
        let mut y_idx = 0;
        for j in 0..ndim{
            let idx = (i / flat_stride[j]) % out_shape[j];
            x_idx += idx * x_stride[j];
            y_idx += idx * y_stride[j];
        }
        ret[i] = x[x_idx].powf(y[y_idx]);
    }
    return ret;
}

#[pyfunction]
pub fn sub(x: Vec<f32>, y: Vec<f32>, x_stride: Vec<usize>,
        y_stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape: usize = out_shape.iter().product();
    let mut ret = vec![0.; flat_shape];
    let ndim = out_shape.len();

    let mut flat_stride = vec![1; ndim];
    for k in (0..ndim-1).rev(){
        flat_stride[k] = flat_stride[k+1] * out_shape[k+1];
    }

    for i in 0..flat_shape{
        let mut x_idx = 0;
        let mut y_idx = 0;
        for j in 0..ndim{
            let idx = (i / flat_stride[j]) % out_shape[j];
            x_idx += idx * x_stride[j];
            y_idx += idx * y_stride[j];
        }
        ret[i] = x[x_idx] - y[y_idx];
    }
    return ret;
}

#[pyfunction]
pub fn cmpeq(x: Vec<f32>, y: Vec<f32>, x_stride: Vec<usize>,
        y_stride: Vec<usize>, out_shape: Vec<usize>)-> Vec<f32>{
    let flat_shape: usize = out_shape.iter().product();
    let mut ret = vec![0.; flat_shape];
    let ndim = out_shape.len();

    let mut flat_stride = vec![1; ndim];
    for k in (0..ndim-1).rev(){
        flat_stride[k] = flat_stride[k+1] * out_shape[k+1];
    }

    for i in 0..flat_shape{
        let mut x_idx = 0;
        let mut y_idx = 0;
        for j in 0..ndim{
            let idx = (i / flat_stride[j]) % out_shape[j];
            x_idx += idx * x_stride[j];
            y_idx += idx * y_stride[j];
        }
        ret[i] = if x[x_idx] == y[y_idx] {1.0} else {0.0};
    }
    return ret;
}