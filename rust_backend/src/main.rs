// use pyo3::prelude::*;
mod utils;
mod reduce_op;
mod movement_op;
// mod binary_op;
use utils::utils::{gen_idx, gen_stride};
use reduce_op::{sum, max};
use movement_op::{shrink, pad, flip, masked_fill};
// use binary_op::cmpeq;

fn check_idx(shape: Vec<usize>, stride: Vec<usize>){
    let flat_shape = shape.iter().product();
    let ndim = shape.len();

    let mut flat_stride = vec![1; ndim];
    for i in (0..ndim-1).rev(){
        flat_stride[i] = flat_stride[i+1]*shape[i+1];
    }

    for i in 0..flat_shape{
        let mut x_idx = 0;
        for j in 0..ndim{
            let idx = (i / flat_stride[j]) % shape[j];
            x_idx += idx * stride[j];
        }
        println!("{}",x_idx);
    }
}

fn check_gen_fn(shape: Vec<usize>, stride: Vec<usize>){
    let flat_shape = shape.iter().product();
    let flat_stride = gen_stride(&shape);
    for i in 0..flat_shape {
        let x_idx = gen_idx(i, &flat_stride, &stride, &shape);
        println!("{}",x_idx);
    }
}

fn main(){
    // let v: Vec<i32> = [1,2,3].to_vec();
    // check_gen_fn([2,5].to_vec(),[0,1].to_vec());
    println!("{:?}",sum([-0.4,-0.1,-0.001,-0.55,3.0,2.0,1.0,1.0].to_vec(),[2,2,2].to_vec(),[4,2,1].to_vec(),[2,2,1].to_vec(),[1].to_vec()));
    println!("{:?}",cmpeq([3.0,2.0,1.0,1.0].to_vec(),[-0.4,-0.1,-0.001,-0.55,3.0,2.0,1.0,1.0].to_vec(),[0,1].to_vec(),[4,1].to_vec(),[2,4].to_vec()));
    println!("{:?}",pad([1.0,2.0,3.0,4.0,3.,2.,2.,2.,2.,].to_vec(),[3,3].to_vec(),[3,1].to_vec(),[(1,1),(1,1)].to_vec(),[5,5].to_vec()));
    println!("{:?}",flip([1.0,2.0,3.0,4.0,3.,2.,2.,2.,2.,].to_vec(),[3,1].to_vec(),[1].to_vec(),[3,3].to_vec()));
    println!("{:?}",masked_fill([1.0,2.0,3.0].to_vec(),[0,1].to_vec(),[false,false,false,false,false,true,false,false,true].to_vec(),0.00001,[3,3].to_vec()));
    
}

fn cmpeq(x: Vec<f32>, y: Vec<f32>, x_stride: Vec<usize>,
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