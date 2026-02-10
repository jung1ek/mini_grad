//map,filter,sum,reduce
// max, sum
// axis; param
use pyo3::prelude::*;
// from opencl code.
#[pyfunction]
pub fn sum(x: Vec<f32>, shape: Vec<usize>, stride: Vec<usize>, out_shape: Vec<usize>, axes: Vec<usize>)->Vec<f32> {
    let out_size: usize =  out_shape.iter().product();
    let mut ret = vec![0.; out_size];
    let mut reduce_size = 1;
    for ax in &axes {
        reduce_size *= shape[*ax];
    }

    let mut reduce_divs = vec![1; axes.len()];
    let mut div = 1;
    for i in (0..axes.len()).rev(){
        reduce_divs[i] = div;
        div *= shape[axes[i]];
    }
    let out_divs = gen_stride(&out_shape);

    // surviving axes in position.
    let mut out_axis_map = vec![None; shape.len()];
    let mut od = 0;
    for k in 0..shape.len() {
        if shape.len() != out_shape.len(){
            if !axes.contains(&k) {
                out_axis_map[k] = Some(od);
                od += 1;
            }
        }
        else {
            out_axis_map[k] = Some(od);
            od+=1;
        }
    }

    // reduce axes and index.
    let mut axes_map = vec![None; shape.len()];
    for (ridx, &ax) in axes.iter().enumerate() {
        axes_map[ax] = Some(ridx);
    }
   
    for i in 0..out_size {
        let mut acc = 0.0;
        for j in 0..reduce_size{
            let mut x_idx = 0;
            for k in 0..shape.len(){
                let idx = if let Some(r_idx) = axes_map[k] {
                    (j / reduce_divs[r_idx]) % shape[k]
                }
                else{
                    let od = out_axis_map[k].unwrap();
                    (i / out_divs[od]) % shape[k]
                };
                x_idx += idx * stride[k];
            }
            acc += x[x_idx];
        }
        ret[i] = acc;
    }
    return ret;
}

#[pyfunction]
pub fn max(x: Vec<f32>, shape: Vec<usize>, stride: Vec<usize>, out_shape: Vec<usize>, axes: Vec<usize>)->Vec<f32> {
    let out_size: usize =  out_shape.iter().product();
    let mut ret = vec![0.; out_size];
    let mut reduce_size = 1;
    for ax in &axes {
        reduce_size *= shape[*ax];
    }

    let mut reduce_divs = vec![1; axes.len()];
    let mut div = 1;
    for i in (0..axes.len()).rev(){
        reduce_divs[i] = div;
        div *= shape[axes[i]];
    }
    let out_divs = gen_stride(&out_shape);

    // surviving axes in position.
    let mut out_axis_map = vec![None; shape.len()];
    let mut od = 0;
    for k in 0..shape.len() {
        if shape.len() != out_shape.len(){
            if !axes.contains(&k) {
                out_axis_map[k] = Some(od);
                od += 1;
            }
        }
        else {
            out_axis_map[k] = Some(od);
            od+=1;
        }
    }

    // reduce axes and index.
    let mut axes_map = vec![None; shape.len()];
    for (ridx, &ax) in axes.iter().enumerate() {
        axes_map[ax] = Some(ridx);
    }

    for i in 0..out_size {
        let mut acc = f32::NEG_INFINITY;
        for j in 0..reduce_size{
            let mut x_idx = 0;
            for k in 0..shape.len(){
                let idx = if let Some(r_idx) = axes_map[k] {
                    (j / reduce_divs[r_idx]) % shape[k]
                }
                else{
                    let od = out_axis_map[k].unwrap();
                    (i / out_divs[od]) % shape[k]
                };
                x_idx += idx * stride[k];
            }
            acc = x[x_idx].max(acc);
        }
        ret[i] = acc;
    }
    return ret;
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