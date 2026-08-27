//! Tiny BERT checkpoint writer for tests.

use std::collections::BTreeMap;
use std::path::Path;

use safetensors::Dtype;
use safetensors::tensor::TensorView;

use crate::error::{Error, Result};
use crate::labels::process_token;
use crate::model::{CONFIG_FILE, MODEL_FILE, TOKENIZER_FILE};

/// RUPunct_small `id2label` order.
pub const ID2LABEL: [&str; 33] = [
    "UPPER_PERIOD",
    "LOWER_PERIOD",
    "UPPER_TOTAL_PERIOD",
    "UPPER_COMMA",
    "LOWER_COMMA",
    "UPPER_TOTAL_COMMA",
    "UPPER_QUESTION",
    "LOWER_QUESTION",
    "UPPER_TOTAL_QUESTION",
    "UPPER_TIRE",
    "LOWER_TIRE",
    "UPPER_TOTAL_TIRE",
    "UPPER_VOSKL",
    "LOWER_VOSKL",
    "UPPER_TOTAL_VOSKL",
    "UPPER_DVOETOCHIE",
    "LOWER_DVOETOCHIE",
    "UPPER_TOTAL_DVOETOCHIE",
    "UPPER_PERIODCOMMA",
    "LOWER_PERIODCOMMA",
    "UPPER_TOTAL_PERIODCOMMA",
    "UPPER_DEFIS",
    "LOWER_DEFIS",
    "UPPER_TOTAL_DEFIS",
    "UPPER_QUESTIONVOSKL",
    "LOWER_QUESTIONVOSKL",
    "UPPER_TOTAL_QUESTIONVOSKL",
    "UPPER_MNOGOTOCHIE",
    "LOWER_MNOGOTOCHIE",
    "UPPER_TOTAL_MNOGOTOCHIE",
    "UPPER_O",
    "LOWER_O",
    "UPPER_TOTAL_O",
];

pub const LABEL_LOWER_O: usize = 31;
pub const LABEL_UPPER_PERIOD: usize = 0;

pub struct TinySpec {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub max_pos: usize,
    pub vocab: usize,
    pub type_vocab: usize,
    pub bias_label: usize,
}

impl Default for TinySpec {
    fn default() -> Self {
        Self {
            hidden: 8,
            heads: 2,
            layers: 1,
            intermediate: 16,
            max_pos: 12,
            vocab: 256,
            type_vocab: 2,
            bias_label: LABEL_LOWER_O,
        }
    }
}

pub fn write_tiny_model(dir: &Path, spec: &TinySpec) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    write_tokenizer(dir)?;
    write_config(dir, spec)?;
    write_weights(dir, spec)?;
    Ok(())
}

fn write_tokenizer(dir: &Path) -> Result<()> {
    let path = dir.join(TOKENIZER_FILE);
    let json = r###"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 1, "content": "[UNK]", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "[CLS]", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true},
            {"id": 3, "content": "[SEP]", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": {"type": "BertNormalizer", "clean_text": true,
                       "handle_chinese_chars": true, "strip_accents": false,
                       "lowercase": false},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
            ],
            "pair": [],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}
            }
        },
        "decoder": null,
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": {
                "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
                "привет": 5, "мир": 6, "меня": 7, "зовут": 8, "анна": 9,
                "как": 10, "дела": 11, "это": 12, "слово": 13
            }
        }
    }"###;
    std::fs::write(&path, json).map_err(|e| Error::io(&path, e))
}

fn write_config(dir: &Path, spec: &TinySpec) -> Result<()> {
    let mut id2 = serde_json::Map::new();
    for (i, lab) in ID2LABEL.iter().enumerate() {
        id2.insert(i.to_string(), serde_json::Value::String((*lab).to_string()));
    }
    let cfg = serde_json::json!({
        "architectures": ["BertForTokenClassification"],
        "hidden_act": "gelu",
        "hidden_size": spec.hidden,
        "id2label": id2,
        "intermediate_size": spec.intermediate,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": spec.max_pos,
        "model_type": "bert",
        "num_attention_heads": spec.heads,
        "num_hidden_layers": spec.layers,
        "num_labels": ID2LABEL.len(),
        "type_vocab_size": spec.type_vocab,
        "vocab_size": spec.vocab
    });
    let path = dir.join(CONFIG_FILE);
    let bytes = serde_json::to_vec_pretty(&cfg)
        .map_err(|e| Error::config(format!("serialize config: {e}")))?;
    std::fs::write(&path, bytes).map_err(|e| Error::io(&path, e))
}

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            ((x >> 8) as f32 / (u32::MAX as f32)) * 0.02 - 0.01
        })
        .collect()
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

fn zeros(n: usize) -> Vec<f32> {
    vec![0.0; n]
}

fn write_weights(dir: &Path, spec: &TinySpec) -> Result<()> {
    let h = spec.hidden;
    let v = spec.vocab;
    let p = spec.max_pos;
    let t = spec.type_vocab;
    let inter = spec.intermediate;
    let nlab = ID2LABEL.len();
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let push = |t: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
                name: &str,
                shape: Vec<usize>,
                data: Vec<f32>| {
        t.push((name.to_string(), shape, data));
    };

    push(
        &mut tensors,
        "bert.embeddings.word_embeddings.weight",
        vec![v, h],
        fill(v * h, 1),
    );
    push(
        &mut tensors,
        "bert.embeddings.position_embeddings.weight",
        vec![p, h],
        fill(p * h, 2),
    );
    push(
        &mut tensors,
        "bert.embeddings.token_type_embeddings.weight",
        vec![t, h],
        fill(t * h, 3),
    );
    push(
        &mut tensors,
        "bert.embeddings.LayerNorm.weight",
        vec![h],
        ones(h),
    );
    push(
        &mut tensors,
        "bert.embeddings.LayerNorm.bias",
        vec![h],
        zeros(h),
    );

    for layer in 0..spec.layers {
        let pre = format!("bert.encoder.layer.{layer}");
        for (name, seed, rows, cols) in [
            ("attention.self.query", 10, h, h),
            ("attention.self.key", 11, h, h),
            ("attention.self.value", 12, h, h),
            ("attention.output.dense", 13, h, h),
        ] {
            push(
                &mut tensors,
                &format!("{pre}.{name}.weight"),
                vec![rows, cols],
                fill(rows * cols, seed + layer as u32 * 50),
            );
            push(
                &mut tensors,
                &format!("{pre}.{name}.bias"),
                vec![rows],
                zeros(rows),
            );
        }
        push(
            &mut tensors,
            &format!("{pre}.attention.output.LayerNorm.weight"),
            vec![h],
            ones(h),
        );
        push(
            &mut tensors,
            &format!("{pre}.attention.output.LayerNorm.bias"),
            vec![h],
            zeros(h),
        );
        push(
            &mut tensors,
            &format!("{pre}.intermediate.dense.weight"),
            vec![inter, h],
            fill(inter * h, 20),
        );
        push(
            &mut tensors,
            &format!("{pre}.intermediate.dense.bias"),
            vec![inter],
            zeros(inter),
        );
        push(
            &mut tensors,
            &format!("{pre}.output.dense.weight"),
            vec![h, inter],
            fill(h * inter, 21),
        );
        push(
            &mut tensors,
            &format!("{pre}.output.dense.bias"),
            vec![h],
            zeros(h),
        );
        push(
            &mut tensors,
            &format!("{pre}.output.LayerNorm.weight"),
            vec![h],
            ones(h),
        );
        push(
            &mut tensors,
            &format!("{pre}.output.LayerNorm.bias"),
            vec![h],
            zeros(h),
        );
    }

    let mut classifier_w = fill(nlab * h, 99);
    for x in classifier_w.iter_mut() {
        *x *= 0.001;
    }
    let mut classifier_b = vec![-100.0f32; nlab];
    if spec.bias_label < nlab {
        classifier_b[spec.bias_label] = 10.0;
    }
    push(
        &mut tensors,
        "classifier.weight",
        vec![nlab, h],
        classifier_w,
    );
    push(&mut tensors, "classifier.bias", vec![nlab], classifier_b);

    write_safetensors(&dir.join(MODEL_FILE), &tensors)
}

fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Result<()> {
    let bufs: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, data)| data.iter().flat_map(|f| f.to_le_bytes()).collect())
        .collect();
    let mut map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
    for (i, (name, shape, _)) in tensors.iter().enumerate() {
        let view = TensorView::new(Dtype::F32, shape.clone(), &bufs[i])
            .map_err(|e| Error::model(format!("tensor view {name}: {e}")))?;
        map.insert(name.clone(), view);
    }
    let bytes =
        safetensors::serialize(&map, None).map_err(|e| Error::model(format!("serialize: {e}")))?;
    std::fs::write(path, bytes).map_err(|e| Error::io(path, e))
}

pub fn expected_for_label(text: &str, label: &str) -> String {
    text.split_whitespace()
        .map(|w| process_token(w, label))
        .collect::<Vec<_>>()
        .join(" ")
}
