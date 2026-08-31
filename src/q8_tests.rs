use super::*;

#[test]
fn roundtrip_block() {
    let mut w = vec![0.0f32; 64];
    for (i, x) in w.iter_mut().enumerate() {
        *x = (i as f32) * 0.01 - 0.3;
    }
    let shape = [2usize, 32];
    let wire = quant(&w, &shape).unwrap();
    assert_eq!(wire.len(), 2 * BLOCK_Q8_0_BYTES);
    let back = dequant(&wire, &shape).unwrap();
    let mut max_err = 0.0f32;
    for (a, b) in w.iter().zip(back.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(max_err < 0.01, "{max_err}");
}

#[test]
fn should_quant_skips_norm_and_tiny() {
    assert!(!should_quant(
        "bert.encoder.layer.0.attention.output.LayerNorm.weight",
        &[768]
    ));
    assert!(!should_quant("classifier.weight", &[33, 8]));
    assert!(should_quant(
        "bert.embeddings.word_embeddings.weight",
        &[128, 32]
    ));
    assert!(should_quant("classifier.weight", &[128, 32]));
}

#[test]
fn dequant_rejects_bad_wire() {
    assert!(dequant(&[0u8; 3], &[32]).is_err());
}

#[test]
fn tiled_matmul_matches_dequant_f32() {
    let n = 160usize;
    let k = 32usize;
    let mut w = vec![0.0f32; n * k];
    for (i, x) in w.iter_mut().enumerate() {
        *x = (i as f32).sin() * 0.02;
    }
    let shape = vec![n, k];
    let q = Q8Weight::from_wire(quant(&w, &shape).unwrap(), shape).unwrap();
    let deq = q.dequant_f32().unwrap();
    let lq = crate::linear::Linear::from_q8(q, None).unwrap();
    let lf = crate::linear::Linear::from_slices(n, k, &deq, None).unwrap();
    let mut xv = vec![0.0f32; 3 * k];
    for (i, v) in xv.iter_mut().enumerate() {
        *v = (i as f32) * 0.01;
    }
    let x = crate::tensor::Tensor::from_vec(xv, vec![3, k]).unwrap();
    let yq = lq.forward(&x).unwrap();
    let yf = lf.forward(&x).unwrap();
    let mut max_err = 0.0f32;
    for (a, b) in yq.as_slice().iter().zip(yf.as_slice()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(max_err < 1e-4, "{max_err}");
}

#[test]
fn from_wire_rejects_bad_len_and_matmul_size() {
    assert!(Q8Weight::from_wire(vec![0u8; 3], vec![32]).is_err());
    let w = vec![0.0f32; 32];
    let q = Q8Weight::from_wire(quant(&w, &[32]).unwrap(), vec![32]).unwrap();
    let mut out = vec![0.0f32; 32];
    assert!(q.matmul_nt(1, &[0.0; 4], &mut out).is_err());
    assert!(crate::linear::Linear::from_q8(q, None).is_err());
}

#[test]
fn gather_rows_matches_dequant() {
    let rows = 8usize;
    let h = 32usize;
    let mut w = vec![0.0f32; rows * h];
    for (i, x) in w.iter_mut().enumerate() {
        *x = (i as f32) * 0.01 - 0.2;
    }
    let q = Q8Weight::from_wire(quant(&w, &[rows, h]).unwrap(), vec![rows, h]).unwrap();
    let deq = q.dequant_f32().unwrap();
    let ids = [0u32, 3, 7];
    let mut got = vec![0.0f32; ids.len() * h];
    q.gather_rows(&ids, &mut got).unwrap();
    for (i, &id) in ids.iter().enumerate() {
        let a = &got[i * h..(i + 1) * h];
        let b = &deq[id as usize * h..(id as usize + 1) * h];
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-6);
        }
    }
    assert!(q.gather_rows(&[99], &mut vec![0.0; h]).is_err());
    let mut set = crate::weights::WeightSet::default();
    set.q8.insert(
        "bert.embeddings.word_embeddings.weight".into(),
        Q8Weight::from_wire(quant(&w, &[rows, h]).unwrap(), vec![rows, h]).unwrap(),
    );
    let table = crate::ws::emb_from_ws(
        &set,
        "bert.embeddings.word_embeddings.weight",
        rows,
        h,
        "word",
    )
    .unwrap();
    let t = crate::ws::gather_emb(&table, &[0, 3]).unwrap();
    assert_eq!(t.dims(), &[2, h]);
    assert!(
        crate::ws::emb_from_ws(
            &set,
            "bert.embeddings.word_embeddings.weight",
            rows + 1,
            h,
            "word"
        )
        .is_err()
    );
}
