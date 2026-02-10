pub mod utils {

    // generate the stride of given shape, contiguous view.
    pub fn gen_stride(shape: &Vec<usize>) -> Vec<usize>{
        let ndim = shape.len();
        let mut stride = vec![1; ndim];

        for i in (0..ndim-1).rev() {
            stride[i] = stride[i+1] * shape[i+1];
        }
        return stride
    }

    // generate flat idx -> multi-dim idx -> flat idx
    pub fn gen_idx(gidx:usize,flat_stride:&Vec<usize>,stride:&Vec<usize>,shape:&Vec<usize>)->usize{
        let ndim = shape.len();
        let mut idx = 0;
        for i in 0..ndim {
            let d_idx = (gidx / flat_stride[i]) % shape[i];
            idx += d_idx * stride[i];
        }
        return idx
    }
}
