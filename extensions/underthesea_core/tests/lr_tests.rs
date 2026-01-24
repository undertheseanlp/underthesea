//! Integration tests for the LR (Logistic Regression) module.

use underthesea_core::lr::{
    model::{ClassIndex, FeatureIndex, LRModel},
    predictor::LRClassifier,
    serialization::{LRFormat, LRModelLoader, LRModelSaver},
    trainer::{LRTrainer, TrainerConfig, TrainingInstance},
};

// ============================================================================
// Model Tests
// ============================================================================

#[test]
fn test_model_creation() {
    let model = LRModel::new();
    assert_eq!(model.num_classes, 0);
    assert_eq!(model.num_features, 0);
}

#[test]
fn test_model_with_classes() {
    let model = LRModel::with_classes(vec![
        "positive".to_string(),
        "negative".to_string(),
        "neutral".to_string(),
    ]);
    assert_eq!(model.num_classes, 3);
    assert_eq!(model.classes.get("positive"), Some(0));
    assert_eq!(model.classes.get("negative"), Some(1));
    assert_eq!(model.classes.get("neutral"), Some(2));
}

#[test]
fn test_model_weights() {
    let mut model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

    model.set_weight(0, 0, 1.5);
    model.set_weight(0, 1, -0.5);
    model.set_weight(1, 0, 0.3);

    assert!((model.get_weight(0, 0) - 1.5).abs() < 1e-10);
    assert!((model.get_weight(0, 1) - (-0.5)).abs() < 1e-10);
    assert!((model.get_weight(1, 0) - 0.3).abs() < 1e-10);
    assert!((model.get_weight(1, 1) - 0.0).abs() < 1e-10);
}

#[test]
fn test_model_biases() {
    let mut model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

    model.set_bias(0, 0.5);
    model.set_bias(1, -0.3);

    assert!((model.get_bias(0) - 0.5).abs() < 1e-10);
    assert!((model.get_bias(1) - (-0.3)).abs() < 1e-10);
}

#[test]
fn test_model_logits() {
    let mut model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

    model.set_weight(0, 0, 1.0);
    model.set_weight(0, 1, 0.5);
    model.set_weight(1, 0, 0.3);
    model.set_bias(0, 0.1);
    model.set_bias(1, 0.2);

    let feature_ids = vec![0, 1];
    let logits = model.compute_logits(&feature_ids);

    // Class A: 1.0 + 0.3 + 0.1 = 1.4
    // Class B: 0.5 + 0.0 + 0.2 = 0.7
    assert!((logits[0] - 1.4).abs() < 1e-10);
    assert!((logits[1] - 0.7).abs() < 1e-10);
}

#[test]
fn test_model_regularization() {
    let mut model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

    model.set_weight(0, 0, 1.0);
    model.set_weight(0, 1, 0.1);
    model.set_bias(0, 2.0);

    // Test L2 decay
    model.apply_l2_decay(0.5);
    assert!((model.get_weight(0, 0) - 0.5).abs() < 1e-10);
    assert!((model.get_bias(0) - 1.0).abs() < 1e-10);

    // Test L1 penalty
    model.set_weight(0, 0, 1.0);
    model.set_weight(0, 1, 0.1);
    model.apply_l1_penalty(0.2);
    assert!((model.get_weight(0, 0) - 0.8).abs() < 1e-10);
    assert_eq!(model.get_weight(0, 1), 0.0); // Should be zeroed
}

// ============================================================================
// Index Tests
// ============================================================================

#[test]
fn test_class_index_basic() {
    let mut index = ClassIndex::new();
    let id1 = index.get_or_insert("A");
    let id2 = index.get_or_insert("B");
    let id3 = index.get_or_insert("A");

    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    assert_eq!(id1, id3);
    assert_eq!(index.len(), 2);
    assert_eq!(index.get("A"), Some(0));
    assert_eq!(index.get_class(0), Some("A"));
}

#[test]
fn test_feature_index_basic() {
    let mut index = FeatureIndex::new();
    let id1 = index.get_or_insert("word=hello");
    let id2 = index.get_or_insert("word=world");
    let id3 = index.get_or_insert("word=hello");

    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    assert_eq!(id1, id3);
    assert_eq!(index.len(), 2);
    assert_eq!(index.get("word=hello"), Some(0));
    assert_eq!(index.get_feature(0), Some("word=hello"));
}

// ============================================================================
// Trainer Tests
// ============================================================================

#[test]
fn test_trainer_creation() {
    let trainer = LRTrainer::new();
    assert!(trainer.get_model().num_classes == 0);
}

#[test]
fn test_trainer_config() {
    let config = TrainerConfig {
        l1_penalty: 0.1,
        l2_penalty: 0.01,
        learning_rate: 0.5,
        max_epochs: 50,
        batch_size: 32,
        tol: 1e-5,
        verbose: 0,
    };
    let trainer = LRTrainer::with_config(config);
    assert!(trainer.get_model().num_classes == 0);
}

#[test]
fn test_training_binary() {
    let data = vec![
        TrainingInstance::new(vec!["f1=a".to_string()], "A".to_string()),
        TrainingInstance::new(vec!["f1=a".to_string()], "A".to_string()),
        TrainingInstance::new(vec!["f1=b".to_string()], "B".to_string()),
        TrainingInstance::new(vec!["f1=b".to_string()], "B".to_string()),
    ];

    let config = TrainerConfig {
        max_epochs: 50,
        learning_rate: 1.0,
        verbose: 0,
        ..Default::default()
    };
    let mut trainer = LRTrainer::with_config(config);
    let model = trainer.train(&data);

    assert_eq!(model.num_classes, 2);
    assert_eq!(model.num_features, 2);

    // Check predictions
    let classifier = LRClassifier::from_model(model);
    assert_eq!(classifier.predict(&["f1=a".to_string()]), "A");
    assert_eq!(classifier.predict(&["f1=b".to_string()]), "B");
}

#[test]
fn test_training_multiclass() {
    let data = vec![
        TrainingInstance::new(vec!["color=red".to_string()], "apple".to_string()),
        TrainingInstance::new(vec!["color=yellow".to_string()], "banana".to_string()),
        TrainingInstance::new(vec!["color=orange".to_string()], "orange".to_string()),
        TrainingInstance::new(vec!["color=red".to_string()], "apple".to_string()),
        TrainingInstance::new(vec!["color=yellow".to_string()], "banana".to_string()),
        TrainingInstance::new(vec!["color=orange".to_string()], "orange".to_string()),
    ];

    let config = TrainerConfig {
        max_epochs: 100,
        learning_rate: 1.0,
        verbose: 0,
        ..Default::default()
    };
    let mut trainer = LRTrainer::with_config(config);
    let model = trainer.train(&data);

    assert_eq!(model.num_classes, 3);

    let classifier = LRClassifier::from_model(model);
    assert_eq!(classifier.predict(&["color=red".to_string()]), "apple");
    assert_eq!(classifier.predict(&["color=yellow".to_string()]), "banana");
    assert_eq!(classifier.predict(&["color=orange".to_string()]), "orange");
}

#[test]
fn test_training_with_regularization() {
    let data = vec![
        TrainingInstance::new(vec!["f1=a".to_string()], "A".to_string()),
        TrainingInstance::new(vec!["f1=b".to_string()], "B".to_string()),
    ];

    // L2 regularization
    let config = TrainerConfig {
        max_epochs: 10,
        learning_rate: 0.5,
        l2_penalty: 0.1,
        verbose: 0,
        ..Default::default()
    };
    let mut trainer = LRTrainer::with_config(config);
    let model = trainer.train(&data);
    assert!(model.l2_norm_squared() < 100.0);

    // L1 regularization
    let config = TrainerConfig {
        max_epochs: 10,
        learning_rate: 0.5,
        l1_penalty: 0.1,
        verbose: 0,
        ..Default::default()
    };
    let mut trainer = LRTrainer::with_config(config);
    let model = trainer.train(&data);
    assert!(model.num_weights() <= model.num_features * model.num_classes);
}

// ============================================================================
// Classifier Tests
// ============================================================================

#[test]
fn test_classifier_creation() {
    let classifier = LRClassifier::new();
    assert_eq!(classifier.num_classes(), 0);
}

#[test]
fn test_classifier_predict() {
    let mut model = LRModel::with_classes(vec!["positive".to_string(), "negative".to_string()]);

    let f1 = model.features.get_or_insert("word=good");
    let f2 = model.features.get_or_insert("word=bad");
    model.num_features = model.features.len();

    model.set_weight(f1, 0, 2.0); // good -> positive
    model.set_weight(f2, 1, 2.0); // bad -> negative

    let classifier = LRClassifier::from_model(model);

    assert_eq!(classifier.predict(&["word=good".to_string()]), "positive");
    assert_eq!(classifier.predict(&["word=bad".to_string()]), "negative");
}

#[test]
fn test_classifier_predict_proba() {
    let mut model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

    let f1 = model.features.get_or_insert("f1=a");
    model.num_features = model.features.len();

    model.set_weight(f1, 0, 2.0);

    let classifier = LRClassifier::from_model(model);
    let proba = classifier.predict_proba(&["f1=a".to_string()]);

    assert_eq!(proba.len(), 2);
    // Check probabilities sum to 1
    let sum: f64 = proba.iter().map(|(_, p)| p).sum();
    assert!((sum - 1.0).abs() < 1e-6);
    // Check sorted by probability descending
    assert!(proba[0].1 >= proba[1].1);
}

#[test]
fn test_classifier_predict_top_k() {
    let model = LRModel::with_classes(vec!["A".to_string(), "B".to_string(), "C".to_string()]);

    let classifier = LRClassifier::from_model(model);
    let top2 = classifier.predict_top_k(&[], 2);

    assert_eq!(top2.len(), 2);
}

// ============================================================================
// Serialization Tests
// ============================================================================

#[test]
fn test_native_format_roundtrip() {
    let mut model = LRModel::with_classes(vec![
        "positive".to_string(),
        "negative".to_string(),
        "neutral".to_string(),
    ]);

    let f1 = model.features.get_or_insert("word=hello");
    let f2 = model.features.get_or_insert("word=world");
    model.num_features = model.features.len();

    model.set_weight(f1, 0, 1.5);
    model.set_weight(f2, 1, -0.5);
    model.set_bias(0, 0.1);
    model.set_bias(1, -0.2);

    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_lr_roundtrip.bin");

    // Save
    let saver = LRModelSaver::new();
    saver.save(&model, &temp_path, LRFormat::Native).unwrap();

    // Load
    let loader = LRModelLoader::new();
    let loaded = loader.load(&temp_path, LRFormat::Native).unwrap();

    // Verify
    assert_eq!(loaded.num_classes, model.num_classes);
    assert_eq!(loaded.num_features, model.num_features);

    let f1_loaded = loaded.features.get("word=hello").unwrap();
    assert!((loaded.get_weight(f1_loaded, 0) - 1.5).abs() < 1e-10);
    assert!((loaded.get_bias(0) - 0.1).abs() < 1e-10);

    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_auto_format_detection() {
    let model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_lr_auto.bin");

    // Save in native format
    let saver = LRModelSaver::new();
    saver.save(&model, &temp_path, LRFormat::Native).unwrap();

    // Load with auto-detection
    let loader = LRModelLoader::new();
    let loaded = loader.load(&temp_path, LRFormat::Auto).unwrap();

    assert_eq!(loaded.num_classes, model.num_classes);

    std::fs::remove_file(temp_path).ok();
}

// ============================================================================
// Full Pipeline Tests
// ============================================================================

#[test]
fn test_full_pipeline() {
    // Create training data
    let data = vec![
        TrainingInstance::new(
            vec!["word=good".to_string(), "pos=adj".to_string()],
            "positive".to_string(),
        ),
        TrainingInstance::new(
            vec!["word=excellent".to_string(), "pos=adj".to_string()],
            "positive".to_string(),
        ),
        TrainingInstance::new(
            vec!["word=bad".to_string(), "pos=adj".to_string()],
            "negative".to_string(),
        ),
        TrainingInstance::new(
            vec!["word=terrible".to_string(), "pos=adj".to_string()],
            "negative".to_string(),
        ),
        TrainingInstance::new(
            vec!["word=okay".to_string(), "pos=adj".to_string()],
            "neutral".to_string(),
        ),
        TrainingInstance::new(
            vec!["word=fine".to_string(), "pos=adj".to_string()],
            "neutral".to_string(),
        ),
    ];

    // Train model
    let config = TrainerConfig {
        max_epochs: 100,
        learning_rate: 1.0,
        l2_penalty: 0.01,
        verbose: 0,
        ..Default::default()
    };
    let mut trainer = LRTrainer::with_config(config);
    let model = trainer.train(&data);

    // Save model
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_lr_pipeline.bin");

    let saver = LRModelSaver::new();
    saver.save(&model, &temp_path, LRFormat::Native).unwrap();

    // Load model
    let loader = LRModelLoader::new();
    let loaded = loader.load(&temp_path, LRFormat::Auto).unwrap();

    // Create classifier
    let classifier = LRClassifier::from_model(loaded);

    // Test predictions
    let pred = classifier.predict(&["word=good".to_string(), "pos=adj".to_string()]);
    assert_eq!(pred, "positive");

    let pred = classifier.predict(&["word=bad".to_string(), "pos=adj".to_string()]);
    assert_eq!(pred, "negative");

    let proba = classifier.predict_proba(&["word=okay".to_string(), "pos=adj".to_string()]);
    assert_eq!(proba.len(), 3);

    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_training_and_inference() {
    // Create a slightly larger dataset
    let mut data = Vec::new();
    for _ in 0..10 {
        data.push(TrainingInstance::new(
            vec!["f=pos1".to_string(), "f=pos2".to_string()],
            "positive".to_string(),
        ));
        data.push(TrainingInstance::new(
            vec!["f=neg1".to_string(), "f=neg2".to_string()],
            "negative".to_string(),
        ));
    }

    let config = TrainerConfig {
        max_epochs: 50,
        learning_rate: 0.5,
        batch_size: 4,
        verbose: 0,
        ..Default::default()
    };
    let mut trainer = LRTrainer::with_config(config);
    let model = trainer.train(&data);

    let classifier = LRClassifier::from_model(model);

    // Test on training data
    let pred1 = classifier.predict(&["f=pos1".to_string(), "f=pos2".to_string()]);
    let pred2 = classifier.predict(&["f=neg1".to_string(), "f=neg2".to_string()]);

    assert_eq!(pred1, "positive");
    assert_eq!(pred2, "negative");
}
