use pyo3::prelude::*;
// flip, shrink, pad
#[pyfunction]
pub fn shrink(x:Vec<f32>,stride: Vec<usize>,slice: Vec<(usize,usize)>,out_shape: Vec<usize>)-> Vec<f32> {
    let flat_size: usize = out_shape.iter().product();
    let mut ret = vec![0.; flat_size];
    let mut offset = vec![0; slice.len()];
    for (i,(s,_)) in slice.iter().enumerate(){
        offset[i] += s;
    }

    let flat_stride = gen_stride(&out_shape);

    for i in 0..flat_size {
        let mut x_idx = 0;
        for j in 0..out_shape.len() {
            let idx = (i / flat_stride[j]) % out_shape[j];
            x_idx += idx* stride[j] + offset[j]* stride[j];
        }
        ret[i] = x[x_idx];
    }
    ret
}

#[pyfunction]
pub fn pad(x:Vec<f32>, shape: Vec<usize>, stride: Vec<usize>, padding: Vec<(usize,usize)>,out_shape:Vec<usize>)->Vec<f32>{
    let flat_size: usize = out_shape.iter().product();
    let mut ret = vec![0.; flat_size];
    let mut offset = vec![0; padding.len()];

    for (i,(pb,_)) in padding.iter().enumerate() {
        offset[i] = -(*pb as isize);
    }
    let flat_stride = gen_stride(&out_shape);

    for i in 0..flat_size {
        let mut x_idx = 0;
        let mut valid = true;
        for j in 0..out_shape.len() {
            let idx = (i / flat_stride[j]) % out_shape[j];
            let src_idx = idx as isize + offset[j];
            // if this dimension is in padding → output stays zero
            if src_idx < 0 || src_idx >= shape[j] as isize {
                valid = false;
                break;
            }
            x_idx += src_idx as usize * stride[j];
        }
        if valid{
            ret[i] = x[x_idx];
        }
    }
    ret
}

#[pyfunction]
pub fn flip(x:Vec<f32>,stride: Vec<isize>,axes: Vec<usize>,out_shape: Vec<usize>)-> Vec<f32> {
    let flat_size:usize = out_shape.iter().product();
    let mut ret = vec![0.; flat_size];
    let mut stride = stride;

    // base pointer not 0.
    let mut base: isize = 0;
    for &ax in &axes {
        base += (out_shape[ax] as isize - 1) * stride[ax];
        stride[ax] = -stride[ax];
    }
    let flat_stride = gen_stride(&out_shape);

    for j in 0..flat_size {
        let mut x_idx = base;
        for k in 0..out_shape.len() {
            let idx = (j / flat_stride[k]) % out_shape[k];
            x_idx += idx as isize  * stride[k];
        }
        ret[j] = x[x_idx as usize];
    }
    ret
}

#[pyfunction]
pub fn masked_fill(x:Vec<f32>,stride: Vec<usize>,mask: Vec<bool>,value:f32,out_shape: Vec<usize>)-> Vec<f32>{
    let flat_size: usize = out_shape.iter().product();
    let mut ret = vec![0.; flat_size];
    let flat_stride = gen_stride(&out_shape);

    for i in 0..flat_size {
        let mut x_idx = 0;
        let mut m_idx = 0;

        for j in 0..out_shape.len(){
            let idx = (i / flat_stride[j]) % out_shape[j];
            x_idx += idx * stride[j];
            m_idx += idx * flat_stride[j];
        }
        if mask[m_idx]{
            ret[i] = value;
        }
        else{
            ret[i] = x[x_idx];
        }
    }
    ret
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