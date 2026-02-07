use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{self, Read};

const FASTTEXT_FILEFORMAT_MAGIC: i32 = 0x2F4F16BA_u32 as i32;
const FASTTEXT_VERSION: i32 = 12;

/// Model type enum matching FastText C++ model_name
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelName {
    CBow = 1,
    SkipGram = 2,
    Supervised = 3,
}

/// Loss function enum matching FastText C++ loss_name
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossName {
    HierarchicalSoftmax = 1,
    NegativeSampling = 2,
    Softmax = 3,
    OneVsAll = 4,
}

/// FastText model hyperparameters, deserialized from the binary header.
#[derive(Debug, Clone)]
pub struct Args {
    pub dim: i32,
    pub ws: i32,
    pub epoch: i32,
    pub min_count: i32,
    pub neg: i32,
    pub word_ngrams: i32,
    pub loss: LossName,
    pub model: ModelName,
    pub bucket: i32,
    pub minn: i32,
    pub maxn: i32,
    pub lr_update_rate: i32,
    pub t: f64,
}

impl Args {
    /// Read Args from a binary stream (after magic + version).
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let dim = reader.read_i32::<LittleEndian>()?;
        let ws = reader.read_i32::<LittleEndian>()?;
        let epoch = reader.read_i32::<LittleEndian>()?;
        let min_count = reader.read_i32::<LittleEndian>()?;
        let neg = reader.read_i32::<LittleEndian>()?;
        let word_ngrams = reader.read_i32::<LittleEndian>()?;
        let loss_raw = reader.read_i32::<LittleEndian>()?;
        let model_raw = reader.read_i32::<LittleEndian>()?;
        let bucket = reader.read_i32::<LittleEndian>()?;
        let minn = reader.read_i32::<LittleEndian>()?;
        let maxn = reader.read_i32::<LittleEndian>()?;
        let lr_update_rate = reader.read_i32::<LittleEndian>()?;
        let t = reader.read_f64::<LittleEndian>()?;

        let loss = match loss_raw {
            1 => LossName::HierarchicalSoftmax,
            2 => LossName::NegativeSampling,
            3 => LossName::Softmax,
            4 => LossName::OneVsAll,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown loss type: {}", loss_raw),
                ))
            }
        };

        let model = match model_raw {
            1 => ModelName::CBow,
            2 => ModelName::SkipGram,
            3 => ModelName::Supervised,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown model type: {}", model_raw),
                ))
            }
        };

        Ok(Args {
            dim,
            ws,
            epoch,
            min_count,
            neg,
            word_ngrams,
            loss,
            model,
            bucket,
            minn,
            maxn,
            lr_update_rate,
            t,
        })
    }
}

/// Verify the FastText file magic number and version.
pub fn check_header<R: Read>(reader: &mut R) -> io::Result<()> {
    let magic = reader.read_i32::<LittleEndian>()?;
    if magic != FASTTEXT_FILEFORMAT_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Invalid FastText magic number: 0x{:08X} (expected 0x{:08X})",
                magic, FASTTEXT_FILEFORMAT_MAGIC
            ),
        ));
    }

    let version = reader.read_i32::<LittleEndian>()?;
    if version > FASTTEXT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Unsupported FastText version: {} (max supported: {})",
                version, FASTTEXT_VERSION
            ),
        ));
    }

    Ok(())
}
