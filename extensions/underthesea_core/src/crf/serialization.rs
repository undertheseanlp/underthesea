//! CRF Model serialization and deserialization.
//!
//! This module provides functionality for saving and loading CRF models:
//! - Native binary format (bincode-based, fast and compact)
//! - CRFsuite binary format (for compatibility with existing models)

use super::crfsuite_format::{write_crfsuite_model, CRFsuiteFeature};
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
            CRFFormat::CRFsuite => self.save_crfsuite(model, path),
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

    /// Save model in CRFsuite format for compatibility with python-crfsuite.
    fn save_crfsuite<P: AsRef<Path>>(&self, model: &CRFModel, path: P) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let mut writer = BufWriter::new(file);

        // Collect labels
        let labels: Vec<String> = (0..model.num_labels)
            .filter_map(|i| model.labels.get_label(i as u32).map(|s| s.to_string()))
            .collect();

        // Collect attributes
        let attributes: Vec<String> = (0..model.num_attributes)
            .filter_map(|i| model.attributes.get_attr(i as u32).map(|s| s.to_string()))
            .collect();

        // Collect features
        let mut features: Vec<CRFsuiteFeature> = Vec::new();

        // State features (type=0)
        for (&(attr_id, label_id), &weight) in model.state_weights_iter() {
            if weight.abs() > 1e-10 {
                features.push(CRFsuiteFeature {
                    feat_type: 0,
                    src: attr_id,
                    dst: label_id,
                    weight,
                });
            }
        }

        // Transition features (type=1)
        for from_label in 0..model.num_labels {
            for to_label in 0..model.num_labels {
                let weight = model.get_transition(from_label as u32, to_label as u32);
                if weight.abs() > 1e-10 {
                    features.push(CRFsuiteFeature {
                        feat_type: 1,
                        src: from_label as u32,
                        dst: to_label as u32,
                        weight,
                    });
                }
            }
        }

        // Write using the crfsuite_format module
        write_crfsuite_model(&mut writer, &labels, &attributes, &features)?;

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
    ///
    /// CQDB format:
    /// - Header (24 bytes): magic(4) + size(4) + flag(4) + byteorder(4) + bwd_size(4) + bwd_offset(4)
    /// - Table references (2048 bytes): 256 * (offset:4 + num:4)
    /// - Key/data pairs: for each entry: id(4) + ksize(4) + key_bytes(ksize)
    /// - Hash tables
    /// - Backward links: bwd_size * offset(4)
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

        let cqdb_start = offset;

        // Seek to CQDB start
        reader
            .seek(SeekFrom::Start(cqdb_start as u64))
            .map_err(|e| format!("Failed to seek to CQDB: {}", e))?;

        // Read CQDB header (24 bytes)
        let mut cqdb_magic = [0u8; 4];
        reader
            .read_exact(&mut cqdb_magic)
            .map_err(|e| format!("Failed to read CQDB magic: {}", e))?;

        if &cqdb_magic != b"CQDB" {
            return Err(format!("Invalid CQDB magic: {:?}", cqdb_magic));
        }

        let _size = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read CQDB size: {}", e))?;
        let _flag = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read CQDB flag: {}", e))?;
        let _byteorder = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read CQDB byteorder: {}", e))?;
        let bwd_size = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read CQDB bwd_size: {}", e))?;
        let bwd_offset = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read CQDB bwd_offset: {}", e))?;

        // Read backward links (id -> offset mapping)
        // bwd_offset is relative to CQDB start
        reader
            .seek(SeekFrom::Start((cqdb_start + bwd_offset) as u64))
            .map_err(|e| format!("Failed to seek to backward table: {}", e))?;

        let num_to_read = std::cmp::min(count, bwd_size);
        let mut bwd_offsets = Vec::with_capacity(num_to_read as usize);
        for _ in 0..num_to_read {
            let off = reader
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("Failed to read backward offset: {}", e))?;
            bwd_offsets.push(off);
        }

        // Read strings using backward offsets
        // Each entry at offset: id(4) + ksize(4) + key_bytes(ksize, null-terminated)
        for (id, &entry_offset) in bwd_offsets.iter().enumerate() {
            if entry_offset == 0 {
                continue; // Empty slot
            }

            // entry_offset is absolute (from file start, as stored during write)
            // But we need to adjust: writer stores offset relative to file, not CQDB
            // Actually, looking at writer: offset = CQDB_HEADER_SIZE + CQDB_TABLEREF_SIZE + data_pos
            // And that's stored as-is in backward table
            // So the offset stored is relative to CQDB start
            reader
                .seek(SeekFrom::Start((cqdb_start + entry_offset) as u64))
                .map_err(|e| {
                    format!("Failed to seek to entry at offset {}: {}", entry_offset, e)
                })?;

            // Read entry: id(4) + ksize(4) + key_bytes
            let stored_id = reader
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("Failed to read entry id: {}", e))?;

            if stored_id != id as u32 {
                // Mismatch, but continue - backward table might have different ordering
            }

            let ksize = reader
                .read_u32::<LittleEndian>()
                .map_err(|e| format!("Failed to read key size: {}", e))?;

            // Read key bytes (includes null terminator)
            let mut key_bytes = vec![0u8; ksize as usize];
            reader
                .read_exact(&mut key_bytes)
                .map_err(|e| format!("Failed to read key bytes: {}", e))?;

            // Remove null terminator if present
            if let Some(null_pos) = key_bytes.iter().position(|&b| b == 0) {
                key_bytes.truncate(null_pos);
            }

            let s = String::from_utf8_lossy(&key_bytes).to_string();

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
    ///
    /// Features chunk format:
    /// - Chunk header: magic("FEAT", 4) + size(4) + num_features(4)
    /// - Features: for each feature: type(4) + src(4) + dst(4) + weight(8)
    fn read_crfsuite_features(
        &self,
        reader: &mut BufReader<File>,
        offset: u32,
        count: u32,
        model: &mut CRFModel,
    ) -> Result<(), String> {
        use std::io::Seek;
        use std::io::SeekFrom;

        // Seek to features chunk
        reader
            .seek(SeekFrom::Start(offset as u64))
            .map_err(|e| format!("Failed to seek to features: {}", e))?;

        // Read chunk header: magic(4) + size(4) + num_features(4)
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Failed to read FEAT magic: {}", e))?;

        if &magic != b"FEAT" {
            return Err(format!("Invalid FEAT magic: {:?}", magic));
        }

        let _chunk_size = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read FEAT chunk size: {}", e))?;
        let num_features = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read num features: {}", e))?;

        // Use the count from chunk header if available, otherwise use header count
        let actual_count = if num_features > 0 {
            num_features
        } else {
            count
        };

        // Initialize transition weights matrix
        let num_labels = model.num_labels;
        if num_labels > 0 {
            // Force initialization
            model.set_transition(0, 0, 0.0);
        }

        // Read each feature (20 bytes each)
        for _ in 0..actual_count {
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
