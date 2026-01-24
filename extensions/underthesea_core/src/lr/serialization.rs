//! LR Model serialization and deserialization.
//!
//! This module provides functionality for saving and loading LR models:
//! - Native binary format (bincode-based, fast and compact)

use super::model::LRModel;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Supported model file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LRFormat {
    /// Native binary format using bincode
    Native,

    /// Auto-detect format based on file magic bytes
    Auto,
}

/// Magic bytes for native format
const NATIVE_MAGIC: &[u8; 4] = b"UTLR"; // UnderTheSea LR

/// Model saver for writing LR models to files.
pub struct LRModelSaver;

impl LRModelSaver {
    /// Create a new model saver.
    pub fn new() -> Self {
        Self
    }

    /// Save a model to file.
    pub fn save<P: AsRef<Path>>(
        &self,
        model: &LRModel,
        path: P,
        format: LRFormat,
    ) -> Result<(), String> {
        match format {
            LRFormat::Native | LRFormat::Auto => self.save_native(model, path),
        }
    }

    /// Save model in native format.
    fn save_native<P: AsRef<Path>>(&self, model: &LRModel, path: P) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let mut writer = BufWriter::new(file);

        // Write magic
        writer
            .write_all(NATIVE_MAGIC)
            .map_err(|e| format!("Failed to write magic: {}", e))?;

        // Serialize model using bincode
        bincode::serialize_into(&mut writer, model)
            .map_err(|e| format!("Failed to serialize model: {}", e))?;

        writer
            .flush()
            .map_err(|e| format!("Failed to flush: {}", e))?;

        Ok(())
    }
}

impl Default for LRModelSaver {
    fn default() -> Self {
        Self::new()
    }
}

/// Model loader for reading LR models from files.
pub struct LRModelLoader;

impl LRModelLoader {
    /// Create a new model loader.
    pub fn new() -> Self {
        Self
    }

    /// Load a model from file.
    pub fn load<P: AsRef<Path>>(&self, path: P, format: LRFormat) -> Result<LRModel, String> {
        match format {
            LRFormat::Native => self.load_native(path),
            LRFormat::Auto => self.load_auto(path),
        }
    }

    /// Auto-detect format and load.
    fn load_auto<P: AsRef<Path>>(&self, path: P) -> Result<LRModel, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut reader = BufReader::new(file);

        // Read magic bytes
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Failed to read magic: {}", e))?;

        drop(reader);

        if &magic == NATIVE_MAGIC {
            self.load_native(path)
        } else {
            Err(format!(
                "Unknown file format: {:?}. Expected UTLR magic bytes.",
                magic
            ))
        }
    }

    /// Load model in native format.
    fn load_native<P: AsRef<Path>>(&self, path: P) -> Result<LRModel, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Failed to read magic: {}", e))?;

        if &magic != NATIVE_MAGIC {
            return Err(format!(
                "Invalid magic bytes: expected {:?}, got {:?}",
                NATIVE_MAGIC, magic
            ));
        }

        // Deserialize model
        let model: LRModel = bincode::deserialize_from(&mut reader)
            .map_err(|e| format!("Failed to deserialize model: {}", e))?;

        Ok(model)
    }
}

impl Default for LRModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model() -> LRModel {
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

        model
    }

    #[test]
    fn test_format_magic() {
        assert_eq!(NATIVE_MAGIC.len(), 4);
        assert_eq!(NATIVE_MAGIC, b"UTLR");
    }

    #[test]
    fn test_native_roundtrip() {
        let model = create_test_model();

        // Create a temporary file
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_lr_model.bin");

        // Save
        let saver = LRModelSaver::new();
        saver.save(&model, &temp_path, LRFormat::Native).unwrap();

        // Load
        let loader = LRModelLoader::new();
        let loaded = loader.load(&temp_path, LRFormat::Native).unwrap();

        // Verify
        assert_eq!(loaded.num_classes, model.num_classes);
        assert_eq!(loaded.num_features, model.num_features);

        // Check some weights
        let f1 = loaded.features.get("word=hello").unwrap();
        assert!((loaded.get_weight(f1, 0) - 1.5).abs() < 1e-10);

        // Check biases
        assert!((loaded.get_bias(0) - 0.1).abs() < 1e-10);
        assert!((loaded.get_bias(1) - (-0.2)).abs() < 1e-10);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_auto_format_detection() {
        let model = create_test_model();

        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_lr_model_auto.bin");

        // Save in native format
        let saver = LRModelSaver::new();
        saver.save(&model, &temp_path, LRFormat::Native).unwrap();

        // Load with auto-detection
        let loader = LRModelLoader::new();
        let loaded = loader.load(&temp_path, LRFormat::Auto).unwrap();

        assert_eq!(loaded.num_classes, model.num_classes);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_invalid_magic() {
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_lr_invalid.bin");

        // Write invalid magic
        let mut file = File::create(&temp_path).unwrap();
        file.write_all(b"XXXX").unwrap();
        file.flush().unwrap();
        drop(file);

        let loader = LRModelLoader::new();
        let result = loader.load(&temp_path, LRFormat::Auto);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown file format"));

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_file_not_found() {
        let loader = LRModelLoader::new();
        let result = loader.load("/nonexistent/path/model.bin", LRFormat::Auto);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to open file"));
    }
}
