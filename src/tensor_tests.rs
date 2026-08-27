use super::*;

#[test]
fn zeros_and_add() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let z = Tensor::zeros(vec![2, 2]).unwrap();
    let s = a.add(&z).unwrap();
    assert_eq!(s.as_slice(), a.as_slice());
    assert_eq!(s.elem_count(), 4);
    assert_eq!(s.rank(), 2);
    assert_eq!(s.shape().to_string(), "[2, 2]");
}

#[test]
fn reshape_is_zero_copy() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = a.reshape(vec![4]).unwrap();
    assert_eq!(a.as_slice().as_ptr(), b.as_slice().as_ptr());
    assert_eq!(b.dims(), &[4]);
    assert!(a.reshape(vec![3]).is_err());
}

#[test]
fn from_vec_rejects_len_mismatch() {
    assert!(Tensor::from_vec(vec![1.0], vec![2, 2]).is_err());
}

#[test]
fn add_rejects_shape_mismatch() {
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
    let b = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
    assert!(a.add(&b).is_err());
}

#[test]
fn matmul_identity() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let i = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    let c = a.matmul(&i).unwrap();
    assert_eq!(c.as_slice(), a.as_slice());
    let r1 = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
    assert!(a.matmul(&r1).is_err());
}

#[test]
fn matmul_inner_mismatch() {
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    assert!(a.matmul(&b).is_err());
}

#[test]
fn matmul_weight_t_and_bias() {
    let x = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    let y = x.matmul_weight_t(&w).unwrap();
    assert_eq!(y.as_slice(), &[1.0, 0.0, 0.0, 1.0]);
    let bias = Tensor::from_vec(vec![1.0, -1.0], vec![2]).unwrap();
    let z = y.add_bias_2d(&bias).unwrap();
    assert!((z.as_slice()[0] - 2.0).abs() < 1e-5);
    assert!((z.as_slice()[1] + 1.0).abs() < 1e-5);
    let bad = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
    assert!(y.add_bias_2d(&bad).is_err());
    assert!(bias.add_bias_2d(&bias).is_err());
}

#[test]
fn matmul_weight_t_rejects_bad_rank() {
    let x = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
    let w = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap();
    assert!(x.matmul_weight_t(&w).is_err());
}

#[test]
fn as_mut_slice_copy_on_write() {
    let mut a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
    let clone = a.clone();
    a.as_mut_slice()[0] = 9.0;
    assert_eq!(clone.as_slice()[0], 1.0);
    assert_eq!(a.as_slice()[0], 9.0);
}

#[test]
fn add_covers_remainder_len() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
    let s = a.add(&b).unwrap();
    assert_eq!(s.as_slice(), &[2.0, 3.0, 4.0]);
}
