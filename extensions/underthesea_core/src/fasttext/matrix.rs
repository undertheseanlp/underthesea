use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{self, Read};

/// Trait for matrix operations needed by FastText inference.
pub trait Matrix {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn add_row_to(&self, row: usize, output: &mut [f32]);
    fn dot_row(&self, vec: &[f32], row: usize) -> f32;
}

// ============================================================================
// Dense Matrix
// ============================================================================

/// Dense float32 matrix stored in row-major order.
#[derive(Debug)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl DenseMatrix {
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let rows = reader.read_i64::<LittleEndian>()? as usize;
        let cols = reader.read_i64::<LittleEndian>()? as usize;

        let n = rows * cols;
        let mut data = vec![0f32; n];
        for val in data.iter_mut() {
            *val = reader.read_f32::<LittleEndian>()?;
        }

        Ok(DenseMatrix { rows, cols, data })
    }
}

impl Matrix for DenseMatrix {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn add_row_to(&self, row: usize, output: &mut [f32]) {
        let start = row * self.cols;
        let row_data = &self.data[start..start + self.cols];
        for (o, &v) in output.iter_mut().zip(row_data.iter()) {
            *o += v;
        }
    }
    fn dot_row(&self, vec: &[f32], row: usize) -> f32 {
        let start = row * self.cols;
        let row_data = &self.data[start..start + self.cols];
        vec.iter().zip(row_data.iter()).map(|(&a, &b)| a * b).sum()
    }
}

// ============================================================================
// Product Quantization
// ============================================================================

const KSUB: usize = 256;

#[derive(Debug)]
struct ProductQuantizer {
    #[allow(dead_code)]
    dim: usize,
    nsubq: usize,
    dsub: usize,
    lastdsub: usize,
    centroids: Vec<f32>,
}

impl ProductQuantizer {
    fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let dim = reader.read_i32::<LittleEndian>()? as usize;
        let nsubq = reader.read_i32::<LittleEndian>()? as usize;
        let dsub = reader.read_i32::<LittleEndian>()? as usize;
        let lastdsub = reader.read_i32::<LittleEndian>()? as usize;

        let n = nsubq * KSUB * dsub;
        let mut centroids = vec![0f32; n];
        for val in centroids.iter_mut() {
            *val = reader.read_f32::<LittleEndian>()?;
        }

        Ok(ProductQuantizer {
            dim,
            nsubq,
            dsub,
            lastdsub,
            centroids,
        })
    }

    #[inline]
    fn get_centroid(&self, subq: usize, code: u8) -> &[f32] {
        let offset = subq * KSUB * self.dsub + (code as usize) * self.dsub;
        let d = if subq == self.nsubq - 1 {
            self.lastdsub
        } else {
            self.dsub
        };
        &self.centroids[offset..offset + d]
    }
}

/// Quantized matrix using Product Quantization.
#[derive(Debug)]
pub struct QuantizedMatrix {
    #[allow(dead_code)]
    qnorm: bool,
    rows: usize,
    cols: usize,
    codes: Vec<u8>,
    pq: ProductQuantizer,
    norm_pq: Option<ProductQuantizer>,
    norm_codes: Vec<u8>,
}

impl QuantizedMatrix {
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let qnorm = reader.read_u8()? != 0;
        let rows = reader.read_i64::<LittleEndian>()? as usize;
        let cols = reader.read_i64::<LittleEndian>()? as usize;
        let codesize = reader.read_i32::<LittleEndian>()? as usize;

        let mut codes = vec![0u8; codesize];
        reader.read_exact(&mut codes)?;

        let pq = ProductQuantizer::load(reader)?;

        let (norm_pq, norm_codes) = if qnorm {
            let mut ncodes = vec![0u8; rows];
            reader.read_exact(&mut ncodes)?;
            let npq = ProductQuantizer::load(reader)?;
            (Some(npq), ncodes)
        } else {
            (None, Vec::new())
        };

        Ok(QuantizedMatrix {
            qnorm,
            rows,
            cols,
            codes,
            pq,
            norm_pq,
            norm_codes,
        })
    }

    fn get_norm(&self, row: usize) -> f32 {
        if let Some(ref npq) = self.norm_pq {
            let code = self.norm_codes[row];
            npq.get_centroid(0, code)[0]
        } else {
            1.0
        }
    }
}

impl Matrix for QuantizedMatrix {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn add_row_to(&self, row: usize, output: &mut [f32]) {
        let norm = self.get_norm(row);
        let nsubq = self.pq.nsubq;
        let code_offset = row * nsubq;
        let mut dim_offset = 0usize;
        for subq in 0..nsubq {
            let code = self.codes[code_offset + subq];
            let centroid = self.pq.get_centroid(subq, code);
            for (i, &c) in centroid.iter().enumerate() {
                output[dim_offset + i] += norm * c;
            }
            dim_offset += if subq == nsubq - 1 {
                self.pq.lastdsub
            } else {
                self.pq.dsub
            };
        }
    }
    fn dot_row(&self, vec: &[f32], row: usize) -> f32 {
        let norm = self.get_norm(row);
        let nsubq = self.pq.nsubq;
        let code_offset = row * nsubq;
        let mut sum = 0.0f32;
        let mut dim_offset = 0usize;
        for subq in 0..nsubq {
            let code = self.codes[code_offset + subq];
            let centroid = self.pq.get_centroid(subq, code);
            for (i, &c) in centroid.iter().enumerate() {
                sum += vec[dim_offset + i] * c;
            }
            dim_offset += if subq == nsubq - 1 {
                self.pq.lastdsub
            } else {
                self.pq.dsub
            };
        }
        sum * norm
    }
}

// ============================================================================
// Enum wrapper
// ============================================================================

pub enum FastTextMatrix {
    Dense(DenseMatrix),
    Quantized(QuantizedMatrix),
}

impl FastTextMatrix {
    pub fn load<R: Read>(reader: &mut R, quant: bool) -> io::Result<Self> {
        if quant {
            Ok(FastTextMatrix::Quantized(QuantizedMatrix::load(reader)?))
        } else {
            Ok(FastTextMatrix::Dense(DenseMatrix::load(reader)?))
        }
    }
}

impl Matrix for FastTextMatrix {
    fn rows(&self) -> usize {
        match self {
            FastTextMatrix::Dense(m) => m.rows(),
            FastTextMatrix::Quantized(m) => m.rows(),
        }
    }
    fn cols(&self) -> usize {
        match self {
            FastTextMatrix::Dense(m) => m.cols(),
            FastTextMatrix::Quantized(m) => m.cols(),
        }
    }
    fn add_row_to(&self, row: usize, output: &mut [f32]) {
        match self {
            FastTextMatrix::Dense(m) => m.add_row_to(row, output),
            FastTextMatrix::Quantized(m) => m.add_row_to(row, output),
        }
    }
    fn dot_row(&self, vec: &[f32], row: usize) -> f32 {
        match self {
            FastTextMatrix::Dense(m) => m.dot_row(vec, row),
            FastTextMatrix::Quantized(m) => m.dot_row(vec, row),
        }
    }
}
