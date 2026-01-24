//! CRF Model serialization and deserialization.
//!
//! This module provides functionality for saving and loading CRF models:
//! - Native binary format (bincode-based, fast and compact)
//! - CRFsuite binary format (for compatibility with existing models)

use super::model::CRFModel;
use byteorder::{LittleEndian, ReadBytesExt};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Supported model file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CRFFormat {
    /// Native binary format using bincode
    Native,

    /// CRFsuite binary format
    CRFsuite,

    /// Auto-detect format based on file magic bytes
    Auto,
}

/// Magic bytes for native format
const NATIVE_MAGIC: &[u8; 4] = b"UTCF"; // UnderTheSea CRF

/// Magic bytes for CRFsuite format
const CRFSUITE_MAGIC: &[u8; 4] = b"lCRF";

/// Header for native format (reserved for future use)
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NativeHeader {
    version: u32,
    num_labels: u32,
    num_attributes: u32,
    num_features: u32,
}

/// Model saver for writing CRF models to files.
pub struct ModelSaver;

impl ModelSaver {
    /// Create a new model saver.
    pub fn new() -> Self {
        Self
    }

    /// Save a model to file.
    pub fn save<P: AsRef<Path>>(
        &self,
        model: &CRFModel,
        path: P,
        format: CRFFormat,
    ) -> Result<(), String> {
        match format {
            CRFFormat::Native | CRFFormat::Auto => self.save_native(model, path),
            CRFFormat::CRFsuite => Err("Writing CRFsuite format is not supported".to_string()),
        }
    }

    /// Save model in native format.
    fn save_native<P: AsRef<Path>>(&self, model: &CRFModel, path: P) -> Result<(), String> {
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

impl Default for ModelSaver {
    fn default() -> Self {
        Self::new()
    }
}

/// Model loader for reading CRF models from files.
pub struct ModelLoader;

impl ModelLoader {
    /// Create a new model loader.
    pub fn new() -> Self {
        Self
    }

    /// Load a model from file.
    pub fn load<P: AsRef<Path>>(&self, path: P, format: CRFFormat) -> Result<CRFModel, String> {
        match format {
            CRFFormat::Native => self.load_native(path),
            CRFFormat::CRFsuite => self.load_crfsuite(path),
            CRFFormat::Auto => self.load_auto(path),
        }
    }

    /// Auto-detect format and load.
    fn load_auto<P: AsRef<Path>>(&self, path: P) -> Result<CRFModel, String> {
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
        } else if &magic == CRFSUITE_MAGIC {
            self.load_crfsuite(path)
        } else {
            Err(format!("Unknown file format: {:?}", magic))
        }
    }

    /// Load model in native format.
    fn load_native<P: AsRef<Path>>(&self, path: P) -> Result<CRFModel, String> {
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
        let model: CRFModel = bincode::deserialize_from(&mut reader)
            .map_err(|e| format!("Failed to deserialize model: {}", e))?;

        Ok(model)
    }

    /// Load model in CRFsuite format.
    fn load_crfsuite<P: AsRef<Path>>(&self, path: P) -> Result<CRFModel, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut reader = BufReader::new(file);

        // Read CRFsuite header
        let header = self.read_crfsuite_header(&mut reader)?;

        // Create model
        let mut model = CRFModel::new();

        // Read labels
        self.read_crfsuite_strings(
            &mut reader,
            header.off_labels,
            header.num_labels,
            &mut model,
            true,
        )?;

        // Read attributes
        self.read_crfsuite_strings(
            &mut reader,
            header.off_attrs,
            header.num_attrs,
            &mut model,
            false,
        )?;

        // Read features
        self.read_crfsuite_features(
            &mut reader,
            header.off_features,
            header.num_features,
            &mut model,
        )?;

        Ok(model)
    }

    /// Read CRFsuite file header.
    fn read_crfsuite_header(&self, reader: &mut BufReader<File>) -> Result<CRFsuiteHeader, String> {
        // Magic (4 bytes)
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Failed to read magic: {}", e))?;

        if &magic != CRFSUITE_MAGIC {
            return Err(format!("Invalid CRFsuite magic: {:?}", magic));
        }

        // Size (4 bytes)
        let size = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read size: {}", e))?;

        // Type (4 bytes) - should be "FOMC"
        let mut type_bytes = [0u8; 4];
        reader
            .read_exact(&mut type_bytes)
            .map_err(|e| format!("Failed to read type: {}", e))?;

        // Version (4 bytes)
        let version = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read version: {}", e))?;

        // Number of features (4 bytes)
        let num_features = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read num_features: {}", e))?;

        // Number of labels (4 bytes)
        let num_labels = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read num_labels: {}", e))?;

        // Number of attributes (4 bytes)
        let num_attrs = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read num_attrs: {}", e))?;

        // Offset to features (4 bytes)
        let off_features = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read off_features: {}", e))?;

        // Offset to labels (4 bytes)
        let off_labels = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read off_labels: {}", e))?;

        // Offset to attributes (4 bytes)
        let off_attrs = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read off_attrs: {}", e))?;

        // Offset to label references (4 bytes)
        let off_label_refs = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read off_label_refs: {}", e))?;

        // Offset to attribute references (4 bytes)
        let off_attr_refs = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read off_attr_refs: {}", e))?;

        Ok(CRFsuiteHeader {
            size,
            version,
            num_features,
            num_labels,
            num_attrs,
            off_features,
            off_labels,
            off_attrs,
            off_label_refs,
            off_attr_refs,
        })
    }

    /// Read strings (labels or attributes) from CRFsuite format.
    /// CRFsuite uses a CQDB (Constant Quark Database) format for strings.
    fn read_crfsuite_strings(
        &self,
        reader: &mut BufReader<File>,
        offset: u32,
        count: u32,
        model: &mut CRFModel,
        is_labels: bool,
    ) -> Result<(), String> {
        use std::io::Seek;
        use std::io::SeekFrom;

        // Seek to offset
        reader
            .seek(SeekFrom::Start(offset as u64))
            .map_err(|e| format!("Failed to seek to strings: {}", e))?;

        // Read CQDB header
        let mut cqdb_magic = [0u8; 4];
        reader
            .read_exact(&mut cqdb_magic)
            .map_err(|e| format!("Failed to read CQDB magic: {}", e))?;

        // Check for CQDB magic "CQDB"
        if &cqdb_magic != b"CQDB" {
            return Err(format!("Invalid CQDB magic: {:?}", cqdb_magic));
        }

        // CQDB header: flag (4) + bwd_size (4) + num_entries (4)
        let _flag = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read CQDB flag: {}", e))?;
        let _bwd_size = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read CQDB bwd_size: {}", e))?;
        let num_entries = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read CQDB num_entries: {}", e))?;

        // Read forward index
        let mut fwd_offsets = Vec::with_capacity(num_entries as usize);
        for _ in 0..num_entries {
            let off = reader
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("Failed to read fwd offset: {}", e))?;
            fwd_offsets.push(off);
        }

        // Read strings
        let base_offset = reader
            .stream_position()
            .map_err(|e| format!("Failed to get position: {}", e))?
            as u32;

        for (i, &str_off) in fwd_offsets.iter().enumerate() {
            if i >= count as usize {
                break;
            }

            reader
                .seek(SeekFrom::Start((base_offset + str_off) as u64))
                .map_err(|e| format!("Failed to seek to string: {}", e))?;

            // Read null-terminated string
            let mut bytes = Vec::new();
            loop {
                let mut byte = [0u8; 1];
                reader
                    .read_exact(&mut byte)
                    .map_err(|e| format!("Failed to read string byte: {}", e))?;
                if byte[0] == 0 {
                    break;
                }
                bytes.push(byte[0]);
            }

            let s = String::from_utf8_lossy(&bytes).to_string();

            if is_labels {
                model.labels.get_or_insert(&s);
            } else {
                model.attributes.get_or_insert(&s);
            }
        }

        model.num_labels = model.labels.len();
        model.num_attributes = model.attributes.len();

        Ok(())
    }

    /// Read features from CRFsuite format.
    fn read_crfsuite_features(
        &self,
        reader: &mut BufReader<File>,
        offset: u32,
        count: u32,
        model: &mut CRFModel,
    ) -> Result<(), String> {
        use std::io::Seek;
        use std::io::SeekFrom;

        // Seek to features offset
        reader
            .seek(SeekFrom::Start(offset as u64))
            .map_err(|e| format!("Failed to seek to features: {}", e))?;

        // Initialize transition weights matrix
        let num_labels = model.num_labels;
        if num_labels > 0 {
            // Force initialization
            model.set_transition(0, 0, 0.0);
        }

        // Read each feature (20 bytes each)
        for _ in 0..count {
            // Feature type (4 bytes): 0 = state, 1 = transition
            let feat_type = reader
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("Failed to read feature type: {}", e))?;

            // Source ID (4 bytes): attribute ID for state, from_label for transition
            let source = reader
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("Failed to read source: {}", e))?;

            // Target ID (4 bytes): label ID for state, to_label for transition
            let target = reader
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("Failed to read target: {}", e))?;

            // Weight (8 bytes, f64)
            let weight = reader
                .read_f64::<LittleEndian>()
                .map_err(|e| format!("Failed to read weight: {}", e))?;

            match feat_type {
                0 => {
                    // State feature: (attribute, label) -> weight
                    model.set_state_weight(source, target, weight);
                }
                1 => {
                    // Transition feature: (from_label, to_label) -> weight
                    model.set_transition(source, target, weight);
                }
                _ => {
                    // Unknown feature type, skip
                }
            }
        }

        Ok(())
    }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// CRFsuite file header structure.
#[allow(dead_code)]
#[derive(Debug)]
struct CRFsuiteHeader {
    size: u32,
    version: u32,
    num_features: u32,
    num_labels: u32,
    num_attrs: u32,
    off_features: u32,
    off_labels: u32,
    off_attrs: u32,
    off_label_refs: u32,
    off_attr_refs: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model() -> CRFModel {
        let mut model = CRFModel::with_labels(vec![
            "B-PER".to_string(),
            "I-PER".to_string(),
            "O".to_string(),
        ]);

        let attr1 = model.attributes.get_or_insert("word=hello");
        let attr2 = model.attributes.get_or_insert("word=world");
        model.num_attributes = model.attributes.len();

        model.set_state_weight(attr1, 0, 1.5);
        model.set_state_weight(attr2, 2, -0.5);
        model.set_transition(0, 1, 0.8);
        model.set_transition(1, 2, 0.3);

        model.build_features_list();

        model
    }

    #[test]
    fn test_format_detection() {
        let native = NATIVE_MAGIC;
        let crfsuite = CRFSUITE_MAGIC;

        assert_ne!(native, crfsuite);
        assert_eq!(native.len(), 4);
        assert_eq!(crfsuite.len(), 4);
    }

    #[test]
    fn test_native_roundtrip() {
        let model = create_test_model();

        // Create a temporary file
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_crf_model.bin");

        // Save
        let saver = ModelSaver::new();
        saver.save(&model, &temp_path, CRFFormat::Native).unwrap();

        // Load
        let loader = ModelLoader::new();
        let loaded = loader.load(&temp_path, CRFFormat::Native).unwrap();

        // Verify
        assert_eq!(loaded.num_labels, model.num_labels);
        assert_eq!(loaded.num_attributes, model.num_attributes);

        // Check some weights
        let attr1 = loaded.attributes.get("word=hello").unwrap();
        assert!((loaded.get_state_weight(attr1, 0) - 1.5).abs() < 1e-10);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_auto_format_detection() {
        let model = create_test_model();

        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_crf_model_auto.bin");

        // Save in native format
        let saver = ModelSaver::new();
        saver.save(&model, &temp_path, CRFFormat::Native).unwrap();

        // Load with auto-detection
        let loader = ModelLoader::new();
        let loaded = loader.load(&temp_path, CRFFormat::Auto).unwrap();

        assert_eq!(loaded.num_labels, model.num_labels);

        std::fs::remove_file(temp_path).ok();
    }
}
