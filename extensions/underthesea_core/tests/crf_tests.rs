//! Integration tests for the CRF module.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::useless_vec)]

use underthesea_core::crf::{
    features::{extract_char_trigrams, AttributeIndex, LabelIndex},
    model::CRFModel,
    serialization::{CRFFormat, ModelLoader, ModelSaver},
    tagger::CRFTagger,
    trainer::{CRFTrainer, LossFunction, TrainerConfig, TrainingInstance},
};

// ============================================================================
// Feature Tests
// ============================================================================

#[test]
fn test_attribute_index_basic() {
    let mut index = AttributeIndex::new();
    assert!(index.is_empty());

    let id1 = index.get_or_insert("word=hello");
    let id2 = index.get_or_insert("word=world");

    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    assert_eq!(index.len(), 2);

    // Same attribute should get same ID
    let id3 = index.get_or_insert("word=hello");
    assert_eq!(id1, id3);
}

#[test]
fn test_label_index_basic() {
    let mut index = LabelIndex::new();

    let id1 = index.get_or_insert("B-PER");
    let id2 = index.get_or_insert("I-PER");
    let id3 = index.get_or_insert("O");

    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    assert_eq!(id3, 2);

    assert_eq!(index.get_label(0), Some("B-PER"));
    assert_eq!(index.get_label(1), Some("I-PER"));
    assert_eq!(index.get_label(2), Some("O"));
}

#[test]
fn test_char_trigrams() {
    let trigrams = extract_char_trigrams("hello");
    assert_eq!(trigrams, vec!["hel", "ell", "llo"]);

    // Vietnamese text
    let vn_trigrams = extract_char_trigrams("xin chào");
    assert!(!vn_trigrams.is_empty());
}

// ============================================================================
// Model Tests
// ============================================================================

#[test]
fn test_model_creation() {
    let model = CRFModel::new();
    assert_eq!(model.num_labels, 0);
    assert_eq!(model.num_attributes, 0);
}

#[test]
fn test_model_with_labels() {
    let model = CRFModel::with_labels(vec![
        "B-PER".to_string(),
        "I-PER".to_string(),
        "O".to_string(),
    ]);

    assert_eq!(model.num_labels, 3);
    assert_eq!(model.labels.get("B-PER"), Some(0));
    assert_eq!(model.labels.get("O"), Some(2));
}

#[test]
fn test_model_weights() {
    let mut model = CRFModel::with_labels(vec!["A".to_string(), "B".to_string()]);

    // State weights
    model.set_state_weight(0, 0, 1.5);
    model.set_state_weight(0, 1, -0.5);
    assert_eq!(model.get_state_weight(0, 0), 1.5);
    assert_eq!(model.get_state_weight(0, 1), -0.5);
    assert_eq!(model.get_state_weight(1, 0), 0.0); // Default

    // Transition weights
    model.set_transition(0, 1, 0.8);
    model.set_transition(1, 0, -0.3);
    assert_eq!(model.get_transition(0, 1), 0.8);
    assert_eq!(model.get_transition(1, 0), -0.3);
}

#[test]
fn test_model_emission_scores() {
    let mut model = CRFModel::with_labels(vec!["A".to_string(), "B".to_string()]);

    model.set_state_weight(0, 0, 1.0); // attr 0 -> label A
    model.set_state_weight(0, 1, 0.5); // attr 0 -> label B
    model.set_state_weight(1, 0, 0.3); // attr 1 -> label A

    let attr_ids = vec![0, 1];
    let scores = model.emission_scores(&attr_ids);

    assert_eq!(scores[0], 1.3); // 1.0 + 0.3
    assert_eq!(scores[1], 0.5); // 0.5 + 0.0
}

#[test]
fn test_model_regularization() {
    let mut model = CRFModel::with_labels(vec!["A".to_string(), "B".to_string()]);

    model.set_state_weight(0, 0, 2.0);
    model.set_transition(0, 1, 4.0);

    // L2 decay
    model.apply_l2_decay(0.5);
    assert_eq!(model.get_state_weight(0, 0), 1.0);
    assert_eq!(model.get_transition(0, 1), 2.0);

    // L2 norm
    let norm = model.l2_norm_squared();
    assert!(norm > 0.0);
}

// ============================================================================
// Tagger Tests
// ============================================================================

fn create_ner_model() -> CRFModel {
    let mut model = CRFModel::with_labels(vec![
        "B-PER".to_string(),
        "I-PER".to_string(),
        "B-LOC".to_string(),
        "O".to_string(),
    ]);

    // Add attributes
    let attr_john = model.attributes.get_or_insert("word=John");
    let attr_smith = model.attributes.get_or_insert("word=Smith");
    let attr_paris = model.attributes.get_or_insert("word=Paris");
    let attr_is = model.attributes.get_or_insert("word=is");
    let attr_cap = model.attributes.get_or_insert("cap=True");
    let attr_nocap = model.attributes.get_or_insert("cap=False");
    model.num_attributes = model.attributes.len();

    // State features
    model.set_state_weight(attr_john, 0, 3.0); // John -> B-PER
    model.set_state_weight(attr_smith, 1, 3.0); // Smith -> I-PER
    model.set_state_weight(attr_paris, 2, 3.0); // Paris -> B-LOC
    model.set_state_weight(attr_is, 3, 2.0); // is -> O
    model.set_state_weight(attr_cap, 0, 0.5); // capitalized -> B-PER (weak)
    model.set_state_weight(attr_nocap, 3, 0.5); // not capitalized -> O (weak)

    // Transitions
    model.set_transition(0, 1, 2.0); // B-PER -> I-PER
    model.set_transition(0, 3, 1.0); // B-PER -> O
    model.set_transition(1, 3, 1.0); // I-PER -> O
    model.set_transition(2, 3, 1.0); // B-LOC -> O
    model.set_transition(3, 0, 0.5); // O -> B-PER
    model.set_transition(3, 2, 0.5); // O -> B-LOC
    model.set_transition(3, 3, 0.5); // O -> O

    model
}

#[test]
fn test_tagger_creation() {
    let tagger = CRFTagger::new();
    assert_eq!(tagger.num_labels(), 0);

    let model = create_ner_model();
    let tagger = CRFTagger::from_model(model);
    assert_eq!(tagger.num_labels(), 4);
}

#[test]
fn test_viterbi_simple() {
    let model = create_ner_model();
    let tagger = CRFTagger::from_model(model);

    let features = vec![
        vec!["word=John".to_string(), "cap=True".to_string()],
        vec!["word=Smith".to_string(), "cap=True".to_string()],
        vec!["word=is".to_string(), "cap=False".to_string()],
    ];

    let labels = tagger.tag(&features);
    assert_eq!(labels.len(), 3);
    assert_eq!(labels[0], "B-PER");
    assert_eq!(labels[1], "I-PER");
    assert_eq!(labels[2], "O");
}

#[test]
fn test_viterbi_location() {
    let model = create_ner_model();
    let tagger = CRFTagger::from_model(model);

    let features = vec![
        vec!["word=Paris".to_string(), "cap=True".to_string()],
        vec!["word=is".to_string(), "cap=False".to_string()],
    ];

    let labels = tagger.tag(&features);
    assert_eq!(labels[0], "B-LOC");
    assert_eq!(labels[1], "O");
}

#[test]
fn test_tag_with_score() {
    let model = create_ner_model();
    let tagger = CRFTagger::from_model(model);

    let features = vec![vec!["word=John".to_string()]];

    let result = tagger.tag_with_score(&features);
    assert_eq!(result.labels.len(), 1);
    assert!(result.score > 0.0);
}

#[test]
fn test_marginals() {
    let model = create_ner_model();
    let tagger = CRFTagger::from_model(model);

    let features = vec![
        vec!["word=John".to_string()],
        vec!["word=Smith".to_string()],
    ];

    let marginals = tagger.compute_marginals(&features);

    assert_eq!(marginals.len(), 2);
    assert_eq!(marginals[0].len(), 4); // 4 labels

    // Marginals should sum to 1 at each position
    for t in 0..2 {
        let sum: f64 = marginals[t].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Marginals at position {} sum to {}",
            t,
            sum
        );
    }
}

#[test]
fn test_sequence_score() {
    let model = create_ner_model();
    let tagger = CRFTagger::from_model(model);

    let features = vec![
        vec!["word=John".to_string()],
        vec!["word=Smith".to_string()],
    ];

    let labels = vec![0, 1]; // B-PER, I-PER
    let score = tagger.score_sequence(&features, &labels);

    // Score = emission(John, B-PER) + transition(B-PER, I-PER) + emission(Smith, I-PER)
    // = 3.0 + 2.0 + 3.0 = 8.0
    assert!((score - 8.0).abs() < 1e-6);
}

// ============================================================================
// Trainer Tests
// ============================================================================

fn create_training_data() -> Vec<TrainingInstance> {
    vec![
        TrainingInstance::new(
            vec![
                vec!["word=John".to_string(), "cap=True".to_string()],
                vec!["word=Smith".to_string(), "cap=True".to_string()],
                vec!["word=is".to_string(), "cap=False".to_string()],
                vec!["word=here".to_string(), "cap=False".to_string()],
            ],
            vec![
                "B-PER".to_string(),
                "I-PER".to_string(),
                "O".to_string(),
                "O".to_string(),
            ],
        ),
        TrainingInstance::new(
            vec![
                vec!["word=Paris".to_string(), "cap=True".to_string()],
                vec!["word=is".to_string(), "cap=False".to_string()],
                vec!["word=beautiful".to_string(), "cap=False".to_string()],
            ],
            vec!["B-LOC".to_string(), "O".to_string(), "O".to_string()],
        ),
        TrainingInstance::new(
            vec![
                vec!["word=Mary".to_string(), "cap=True".to_string()],
                vec!["word=loves".to_string(), "cap=False".to_string()],
                vec!["word=London".to_string(), "cap=True".to_string()],
            ],
            vec!["B-PER".to_string(), "O".to_string(), "B-LOC".to_string()],
        ),
    ]
}

#[test]
fn test_trainer_creation() {
    let trainer = CRFTrainer::new();
    assert!(trainer.get_model().num_labels == 0);
}

#[test]
fn test_trainer_config() {
    let config = TrainerConfig {
        loss_function: LossFunction::StructuredPerceptron { learning_rate: 0.1 },
        max_iterations: 50,
        epsilon: 1e-5,
        averaging: true,
        verbose: 0,
    };

    let trainer = CRFTrainer::with_config(config);
    assert!(trainer.get_model().num_labels == 0);
}

#[test]
fn test_perceptron_training() {
    let config = TrainerConfig {
        loss_function: LossFunction::StructuredPerceptron { learning_rate: 0.1 },
        max_iterations: 20,
        verbose: 0,
        ..Default::default()
    };

    let mut trainer = CRFTrainer::with_config(config);
    let data = create_training_data();

    let model = trainer.train(&data);

    // Should have learned labels
    assert_eq!(model.num_labels, 4); // B-PER, I-PER, O, B-LOC
    assert!(model.num_state_features() > 0);
}

#[test]
fn test_nll_training() {
    let config = TrainerConfig {
        loss_function: LossFunction::LBFGS {
            l1_penalty: 0.0,
            l2_penalty: 0.01,
        },
        max_iterations: 10,
        verbose: 0,
        ..Default::default()
    };

    let mut trainer = CRFTrainer::with_config(config);
    let data = create_training_data();

    let model = trainer.train(&data);

    assert_eq!(model.num_labels, 4);
}

#[test]
fn test_training_and_inference() {
    let config = TrainerConfig {
        loss_function: LossFunction::StructuredPerceptron { learning_rate: 0.5 },
        max_iterations: 50,
        verbose: 0,
        ..Default::default()
    };

    let mut trainer = CRFTrainer::with_config(config);
    let data = create_training_data();
    let model = trainer.train(&data);

    let tagger = CRFTagger::from_model(model);

    // Test on training data
    let test_features = vec![
        vec!["word=John".to_string(), "cap=True".to_string()],
        vec!["word=Smith".to_string(), "cap=True".to_string()],
    ];

    let labels = tagger.tag(&test_features);
    assert_eq!(labels.len(), 2);
    // After training, should predict B-PER, I-PER
    assert_eq!(labels[0], "B-PER");
    assert_eq!(labels[1], "I-PER");
}

// ============================================================================
// Serialization Tests
// ============================================================================

#[test]
fn test_native_format_roundtrip() {
    let mut model = CRFModel::with_labels(vec![
        "B-PER".to_string(),
        "I-PER".to_string(),
        "O".to_string(),
    ]);

    let attr1 = model.attributes.get_or_insert("word=hello");
    model.num_attributes = model.attributes.len();
    model.set_state_weight(attr1, 0, 1.5);
    model.set_transition(0, 1, 0.8);

    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_crf_native.bin");

    // Save
    let saver = ModelSaver::new();
    saver.save(&model, &temp_path, CRFFormat::Native).unwrap();

    // Load
    let loader = ModelLoader::new();
    let loaded = loader.load(&temp_path, CRFFormat::Native).unwrap();

    // Verify
    assert_eq!(loaded.num_labels, 3);
    assert_eq!(loaded.labels.get("B-PER"), Some(0));

    let attr1_loaded = loaded.attributes.get("word=hello").unwrap();
    assert!((loaded.get_state_weight(attr1_loaded, 0) - 1.5).abs() < 1e-10);
    assert!((loaded.get_transition(0, 1) - 0.8).abs() < 1e-10);

    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_auto_format_detection() {
    let model = CRFModel::with_labels(vec!["A".to_string(), "B".to_string()]);

    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_crf_auto.bin");

    let saver = ModelSaver::new();
    saver.save(&model, &temp_path, CRFFormat::Native).unwrap();

    // Load with auto-detection
    let loader = ModelLoader::new();
    let loaded = loader.load(&temp_path, CRFFormat::Auto).unwrap();

    assert_eq!(loaded.num_labels, 2);

    std::fs::remove_file(temp_path).ok();
}

// ============================================================================
// End-to-End Tests
// ============================================================================

#[test]
fn test_full_pipeline() {
    // 1. Create training data
    let data = create_training_data();

    // 2. Train model
    let config = TrainerConfig {
        loss_function: LossFunction::StructuredPerceptron { learning_rate: 0.5 },
        max_iterations: 100,
        verbose: 0,
        averaging: false, // Disable averaging for more direct learning
        ..Default::default()
    };
    let mut trainer = CRFTrainer::with_config(config);
    let model = trainer.train(&data);

    // 3. Save model
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_full_pipeline.bin");
    let saver = ModelSaver::new();
    saver.save(&model, &temp_path, CRFFormat::Native).unwrap();

    // 4. Load model
    let loader = ModelLoader::new();
    let loaded_model = loader.load(&temp_path, CRFFormat::Auto).unwrap();

    // 5. Create tagger and predict
    let tagger = CRFTagger::from_model(loaded_model);

    let test_features = vec![
        vec!["word=Mary".to_string(), "cap=True".to_string()],
        vec!["word=loves".to_string(), "cap=False".to_string()],
        vec!["word=London".to_string(), "cap=True".to_string()],
    ];

    let labels = tagger.tag(&test_features);

    // Verify we get valid labels (model may not perfectly learn with limited data)
    assert_eq!(labels.len(), 3);
    // Check that labels are from the trained set
    let valid_labels = vec!["B-PER", "I-PER", "B-LOC", "O"];
    for label in &labels {
        assert!(
            valid_labels.contains(&label.as_str()),
            "Unexpected label: {}",
            label
        );
    }

    std::fs::remove_file(temp_path).ok();
}
