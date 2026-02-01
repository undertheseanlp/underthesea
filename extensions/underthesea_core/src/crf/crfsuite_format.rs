//! CRFsuite binary format writer.
//!
//! This module implements the CRFsuite model file format for compatibility
//! with python-crfsuite and other CRFsuite-compatible tools.

use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};

/// CRFsuite file magic
const FILEMAGIC: &[u8; 4] = b"lCRF";
/// Model type for CRF 1d
const MODELTYPE: &[u8; 4] = b"FOMC";
/// Version number
const VERSION_NUMBER: u32 = 100;
/// Header size
const HEADER_SIZE: u32 = 48;
/// Chunk header size
const CHUNK_SIZE: u32 = 12;
/// Feature size (type:4 + src:4 + dst:4 + weight:8 = 20)
const FEATURE_SIZE: u32 = 20;

/// CQDB constants
const CQDB_MAGIC: &[u8; 4] = b"CQDB";
const CQDB_BYTEORDER: u32 = 0x62445371;
const NUM_TABLES: usize = 256;
const CQDB_HEADER_SIZE: usize = 24;
const CQDB_TABLEREF_SIZE: usize = NUM_TABLES * 8; // 256 * (offset:4 + num:4)

/// Jenkins lookup3 hash function (hashlittle)
fn hashlittle(key: &[u8], initval: u32) -> u32 {
    let mut a: u32 = 0xdeadbeef_u32
        .wrapping_add(key.len() as u32)
        .wrapping_add(initval);
    let mut b: u32 = a;
    let mut c: u32 = a;

    let mut i = 0;
    while i + 12 <= key.len() {
        a = a.wrapping_add(u32::from_le_bytes([
            key[i],
            key[i + 1],
            key[i + 2],
            key[i + 3],
        ]));
        b = b.wrapping_add(u32::from_le_bytes([
            key[i + 4],
            key[i + 5],
            key[i + 6],
            key[i + 7],
        ]));
        c = c.wrapping_add(u32::from_le_bytes([
            key[i + 8],
            key[i + 9],
            key[i + 10],
            key[i + 11],
        ]));

        a = a.wrapping_sub(c);
        a ^= c.rotate_left(4);
        c = c.wrapping_add(b);
        b = b.wrapping_sub(a);
        b ^= a.rotate_left(6);
        a = a.wrapping_add(c);
        c = c.wrapping_sub(b);
        c ^= b.rotate_left(8);
        b = b.wrapping_add(a);
        a = a.wrapping_sub(c);
        a ^= c.rotate_left(16);
        c = c.wrapping_add(b);
        b = b.wrapping_sub(a);
        b ^= a.rotate_left(19);
        a = a.wrapping_add(c);
        c = c.wrapping_sub(b);
        c ^= b.rotate_left(4);
        b = b.wrapping_add(a);

        i += 12;
    }

    let remaining = key.len() - i;
    match remaining {
        12 => {
            c = c.wrapping_add(u32::from_le_bytes([
                key[i + 8],
                key[i + 9],
                key[i + 10],
                key[i + 11],
            ]));
            b = b.wrapping_add(u32::from_le_bytes([
                key[i + 4],
                key[i + 5],
                key[i + 6],
                key[i + 7],
            ]));
            a = a.wrapping_add(u32::from_le_bytes([
                key[i],
                key[i + 1],
                key[i + 2],
                key[i + 3],
            ]));
        }
        11 => {
            c = c.wrapping_add((key[i + 10] as u32) << 16);
            c = c.wrapping_add((key[i + 9] as u32) << 8);
            c = c.wrapping_add(key[i + 8] as u32);
            b = b.wrapping_add(u32::from_le_bytes([
                key[i + 4],
                key[i + 5],
                key[i + 6],
                key[i + 7],
            ]));
            a = a.wrapping_add(u32::from_le_bytes([
                key[i],
                key[i + 1],
                key[i + 2],
                key[i + 3],
            ]));
        }
        10 => {
            c = c.wrapping_add((key[i + 9] as u32) << 8);
            c = c.wrapping_add(key[i + 8] as u32);
            b = b.wrapping_add(u32::from_le_bytes([
                key[i + 4],
                key[i + 5],
                key[i + 6],
                key[i + 7],
            ]));
            a = a.wrapping_add(u32::from_le_bytes([
                key[i],
                key[i + 1],
                key[i + 2],
                key[i + 3],
            ]));
        }
        9 => {
            c = c.wrapping_add(key[i + 8] as u32);
            b = b.wrapping_add(u32::from_le_bytes([
                key[i + 4],
                key[i + 5],
                key[i + 6],
                key[i + 7],
            ]));
            a = a.wrapping_add(u32::from_le_bytes([
                key[i],
                key[i + 1],
                key[i + 2],
                key[i + 3],
            ]));
        }
        8 => {
            b = b.wrapping_add(u32::from_le_bytes([
                key[i + 4],
                key[i + 5],
                key[i + 6],
                key[i + 7],
            ]));
            a = a.wrapping_add(u32::from_le_bytes([
                key[i],
                key[i + 1],
                key[i + 2],
                key[i + 3],
            ]));
        }
        7 => {
            b = b.wrapping_add((key[i + 6] as u32) << 16);
            b = b.wrapping_add((key[i + 5] as u32) << 8);
            b = b.wrapping_add(key[i + 4] as u32);
            a = a.wrapping_add(u32::from_le_bytes([
                key[i],
                key[i + 1],
                key[i + 2],
                key[i + 3],
            ]));
        }
        6 => {
            b = b.wrapping_add((key[i + 5] as u32) << 8);
            b = b.wrapping_add(key[i + 4] as u32);
            a = a.wrapping_add(u32::from_le_bytes([
                key[i],
                key[i + 1],
                key[i + 2],
                key[i + 3],
            ]));
        }
        5 => {
            b = b.wrapping_add(key[i + 4] as u32);
            a = a.wrapping_add(u32::from_le_bytes([
                key[i],
                key[i + 1],
                key[i + 2],
                key[i + 3],
            ]));
        }
        4 => {
            a = a.wrapping_add(u32::from_le_bytes([
                key[i],
                key[i + 1],
                key[i + 2],
                key[i + 3],
            ]));
        }
        3 => {
            a = a.wrapping_add((key[i + 2] as u32) << 16);
            a = a.wrapping_add((key[i + 1] as u32) << 8);
            a = a.wrapping_add(key[i] as u32);
        }
        2 => {
            a = a.wrapping_add((key[i + 1] as u32) << 8);
            a = a.wrapping_add(key[i] as u32);
        }
        1 => {
            a = a.wrapping_add(key[i] as u32);
        }
        0 => return c,
        _ => {}
    }

    // Final mixing
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(14));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(11));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(25));
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(16));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(4));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(14));
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(24));

    c
}

/// A bucket entry for CQDB hash table
#[derive(Clone, Copy, Default)]
struct Bucket {
    hash: u32,
    offset: u32,
}

/// CQDB writer for string->id mapping
pub struct CQDBWriter {
    /// Key/data pairs buffer
    data: Vec<u8>,
    /// Hash tables (256 tables)
    tables: Vec<Vec<Bucket>>,
    /// Backward links (id -> offset)
    backward: Vec<u32>,
}

impl CQDBWriter {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            tables: (0..NUM_TABLES).map(|_| Vec::new()).collect(),
            backward: Vec::new(),
        }
    }

    /// Add a string with its ID
    pub fn put(&mut self, s: &str, id: u32) {
        // Include null terminator in key
        let key = format!("{}\0", s);
        let key_bytes = key.as_bytes();

        // Compute hash
        let hv = hashlittle(key_bytes, 0);
        let table_idx = (hv % 256) as usize;

        // Current offset (relative to data start)
        let offset = (CQDB_HEADER_SIZE + CQDB_TABLEREF_SIZE + self.data.len()) as u32;

        // Write key/data pair: id (4) + ksize (4) + key (with null)
        self.data.extend_from_slice(&id.to_le_bytes());
        self.data
            .extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        self.data.extend_from_slice(key_bytes);

        // Add to hash table
        self.tables[table_idx].push(Bucket { hash: hv, offset });

        // Add backward link
        if id as usize >= self.backward.len() {
            self.backward.resize(id as usize + 1, 0);
        }
        self.backward[id as usize] = offset;
    }

    /// Write the CQDB to a writer, returns total bytes written
    pub fn write<W: Write + Seek>(&self, writer: &mut W) -> Result<u32, String> {
        let start_pos = writer
            .stream_position()
            .map_err(|e| format!("Failed to get position: {}", e))? as u32;

        // Calculate sizes
        let data_end = CQDB_HEADER_SIZE + CQDB_TABLEREF_SIZE + self.data.len();

        // Calculate hash tables size
        let mut hash_tables_size = 0usize;
        for table in &self.tables {
            if !table.is_empty() {
                hash_tables_size += table.len() * 2 * 8; // Double size, 8 bytes per bucket
            }
        }

        let bwd_offset = data_end + hash_tables_size;
        let bwd_size = self.backward.len();
        let total_size = bwd_offset + bwd_size * 4;

        // Write header
        writer
            .write_all(CQDB_MAGIC)
            .map_err(|e| format!("Failed to write CQDB magic: {}", e))?;
        writer
            .write_u32::<LittleEndian>(total_size as u32)
            .map_err(|e| format!("Failed to write size: {}", e))?;
        writer
            .write_u32::<LittleEndian>(0) // flag
            .map_err(|e| format!("Failed to write flag: {}", e))?;
        writer
            .write_u32::<LittleEndian>(CQDB_BYTEORDER)
            .map_err(|e| format!("Failed to write byteorder: {}", e))?;
        writer
            .write_u32::<LittleEndian>(bwd_size as u32)
            .map_err(|e| format!("Failed to write bwd_size: {}", e))?;
        writer
            .write_u32::<LittleEndian>(bwd_offset as u32)
            .map_err(|e| format!("Failed to write bwd_offset: {}", e))?;

        // Write table references
        let mut table_offset = data_end as u32;
        for table in &self.tables {
            if table.is_empty() {
                writer
                    .write_u32::<LittleEndian>(0)
                    .map_err(|e| format!("Failed to write table offset: {}", e))?;
                writer
                    .write_u32::<LittleEndian>(0)
                    .map_err(|e| format!("Failed to write table num: {}", e))?;
            } else {
                let bucket_size = table.len() * 2; // Double size for open addressing
                writer
                    .write_u32::<LittleEndian>(table_offset)
                    .map_err(|e| format!("Failed to write table offset: {}", e))?;
                writer
                    .write_u32::<LittleEndian>(bucket_size as u32)
                    .map_err(|e| format!("Failed to write table num: {}", e))?;
                table_offset += (bucket_size * 8) as u32;
            }
        }

        // Write key/data pairs
        writer
            .write_all(&self.data)
            .map_err(|e| format!("Failed to write data: {}", e))?;

        // Write hash tables with open addressing
        for table in &self.tables {
            if !table.is_empty() {
                let n = table.len() * 2;
                let mut buckets = vec![Bucket::default(); n];

                for src in table {
                    let mut k = ((src.hash >> 8) as usize) % n;
                    while buckets[k].offset != 0 {
                        k = (k + 1) % n;
                    }
                    buckets[k] = *src;
                }

                for b in &buckets {
                    writer
                        .write_u32::<LittleEndian>(b.hash)
                        .map_err(|e| format!("Failed to write bucket hash: {}", e))?;
                    writer
                        .write_u32::<LittleEndian>(b.offset)
                        .map_err(|e| format!("Failed to write bucket offset: {}", e))?;
                }
            }
        }

        // Write backward links
        for &offset in &self.backward {
            writer
                .write_u32::<LittleEndian>(offset)
                .map_err(|e| format!("Failed to write backward link: {}", e))?;
        }

        let end_pos = writer
            .stream_position()
            .map_err(|e| format!("Failed to get position: {}", e))? as u32;

        Ok(end_pos - start_pos)
    }
}

/// Feature for CRFsuite format
pub struct CRFsuiteFeature {
    pub feat_type: u32, // 0 = state, 1 = transition
    pub src: u32,
    pub dst: u32,
    pub weight: f64,
}

/// Write a CRFsuite model file
pub fn write_crfsuite_model<W: Write + Seek>(
    writer: &mut W,
    labels: &[String],
    attributes: &[String],
    features: &[CRFsuiteFeature],
) -> Result<(), String> {
    // Skip header, write later
    writer
        .seek(SeekFrom::Start(HEADER_SIZE as u64))
        .map_err(|e| format!("Failed to seek: {}", e))?;

    // Write FEAT chunk
    let off_features = HEADER_SIZE;
    let feat_chunk_size = CHUNK_SIZE + (features.len() as u32) * FEATURE_SIZE;

    writer
        .write_all(b"FEAT")
        .map_err(|e| format!("Failed to write FEAT magic: {}", e))?;
    writer
        .write_u32::<LittleEndian>(feat_chunk_size)
        .map_err(|e| format!("Failed to write FEAT size: {}", e))?;
    writer
        .write_u32::<LittleEndian>(features.len() as u32)
        .map_err(|e| format!("Failed to write FEAT num: {}", e))?;

    for f in features {
        writer
            .write_u32::<LittleEndian>(f.feat_type)
            .map_err(|e| format!("Failed to write feature type: {}", e))?;
        writer
            .write_u32::<LittleEndian>(f.src)
            .map_err(|e| format!("Failed to write feature src: {}", e))?;
        writer
            .write_u32::<LittleEndian>(f.dst)
            .map_err(|e| format!("Failed to write feature dst: {}", e))?;
        writer
            .write_f64::<LittleEndian>(f.weight)
            .map_err(|e| format!("Failed to write feature weight: {}", e))?;
    }

    // Write labels CQDB
    let off_labels = writer
        .stream_position()
        .map_err(|e| format!("Failed to get position: {}", e))? as u32;

    let mut labels_cqdb = CQDBWriter::new();
    for (i, label) in labels.iter().enumerate() {
        labels_cqdb.put(label, i as u32);
    }
    labels_cqdb.write(writer)?;

    // Write attributes CQDB
    let off_attrs = writer
        .stream_position()
        .map_err(|e| format!("Failed to get position: {}", e))? as u32;

    let mut attrs_cqdb = CQDBWriter::new();
    for (i, attr) in attributes.iter().enumerate() {
        attrs_cqdb.put(attr, i as u32);
    }
    attrs_cqdb.write(writer)?;

    // Align to 4 bytes
    let pos = writer
        .stream_position()
        .map_err(|e| format!("Failed to get position: {}", e))? as u32;
    let padding = (4 - (pos % 4)) % 4;
    for _ in 0..padding {
        writer
            .write_u8(0)
            .map_err(|e| format!("Failed to write padding: {}", e))?;
    }

    // Build feature references by label
    let off_labelrefs = writer
        .stream_position()
        .map_err(|e| format!("Failed to get position: {}", e))? as u32;

    // Group features by label for labelrefs (transition features where dst=label)
    let mut label_features: Vec<Vec<u32>> = vec![Vec::new(); labels.len()];
    for (fid, f) in features.iter().enumerate() {
        if f.feat_type == 1 {
            // Transition feature: index by src (from_label)
            if (f.src as usize) < labels.len() {
                label_features[f.src as usize].push(fid as u32);
            }
        }
    }

    // Write LFRF chunk
    let labelrefs_header_size = CHUNK_SIZE + (labels.len() as u32) * 4;
    writer
        .write_all(b"LFRF")
        .map_err(|e| format!("Failed to write LFRF magic: {}", e))?;

    // Placeholder for size, will update later
    let lfrf_size_pos = writer
        .stream_position()
        .map_err(|e| format!("Failed to get position: {}", e))?;
    writer
        .write_u32::<LittleEndian>(0) // size placeholder
        .map_err(|e| format!("Failed to write LFRF size: {}", e))?;
    writer
        .write_u32::<LittleEndian>(labels.len() as u32)
        .map_err(|e| format!("Failed to write LFRF num: {}", e))?;

    // Write offset placeholders
    let offsets_pos = writer
        .stream_position()
        .map_err(|e| format!("Failed to get position: {}", e))?;
    for _ in 0..labels.len() {
        writer
            .write_u32::<LittleEndian>(0)
            .map_err(|e| format!("Failed to write offset placeholder: {}", e))?;
    }

    // Write feature references and collect offsets
    let mut label_offsets = Vec::with_capacity(labels.len());
    for fids in &label_features {
        let offset = writer
            .stream_position()
            .map_err(|e| format!("Failed to get position: {}", e))? as u32;
        label_offsets.push(offset);

        writer
            .write_u32::<LittleEndian>(fids.len() as u32)
            .map_err(|e| format!("Failed to write num features: {}", e))?;
        for &fid in fids {
            writer
                .write_u32::<LittleEndian>(fid)
                .map_err(|e| format!("Failed to write feature id: {}", e))?;
        }
    }

    let lfrf_end = writer
        .stream_position()
        .map_err(|e| format!("Failed to get position: {}", e))? as u32;

    // Update LFRF size
    writer
        .seek(SeekFrom::Start(lfrf_size_pos))
        .map_err(|e| format!("Failed to seek: {}", e))?;
    writer
        .write_u32::<LittleEndian>(lfrf_end - off_labelrefs)
        .map_err(|e| format!("Failed to write LFRF size: {}", e))?;

    // Update offsets
    writer
        .seek(SeekFrom::Start(offsets_pos))
        .map_err(|e| format!("Failed to seek: {}", e))?;
    for offset in &label_offsets {
        writer
            .write_u32::<LittleEndian>(*offset)
            .map_err(|e| format!("Failed to write offset: {}", e))?;
    }

    writer
        .seek(SeekFrom::Start(lfrf_end as u64))
        .map_err(|e| format!("Failed to seek: {}", e))?;

    // Align to 4 bytes
    let pos = lfrf_end;
    let padding = (4 - (pos % 4)) % 4;
    for _ in 0..padding {
        writer
            .write_u8(0)
            .map_err(|e| format!("Failed to write padding: {}", e))?;
    }

    // Build feature references by attribute
    let off_attrrefs = writer
        .stream_position()
        .map_err(|e| format!("Failed to get position: {}", e))? as u32;

    // Group features by attribute for attrrefs (state features where src=attr)
    let mut attr_features: Vec<Vec<u32>> = vec![Vec::new(); attributes.len()];
    for (fid, f) in features.iter().enumerate() {
        if f.feat_type == 0 {
            // State feature: index by src (attr_id)
            if (f.src as usize) < attributes.len() {
                attr_features[f.src as usize].push(fid as u32);
            }
        }
    }

    // Write AFRF chunk
    writer
        .write_all(b"AFRF")
        .map_err(|e| format!("Failed to write AFRF magic: {}", e))?;

    let afrf_size_pos = writer
        .stream_position()
        .map_err(|e| format!("Failed to get position: {}", e))?;
    writer
        .write_u32::<LittleEndian>(0) // size placeholder
        .map_err(|e| format!("Failed to write AFRF size: {}", e))?;
    writer
        .write_u32::<LittleEndian>(attributes.len() as u32)
        .map_err(|e| format!("Failed to write AFRF num: {}", e))?;

    // Write offset placeholders
    let attr_offsets_pos = writer
        .stream_position()
        .map_err(|e| format!("Failed to get position: {}", e))?;
    for _ in 0..attributes.len() {
        writer
            .write_u32::<LittleEndian>(0)
            .map_err(|e| format!("Failed to write offset placeholder: {}", e))?;
    }

    // Write feature references and collect offsets
    let mut attr_offsets = Vec::with_capacity(attributes.len());
    for fids in &attr_features {
        let offset = writer
            .stream_position()
            .map_err(|e| format!("Failed to get position: {}", e))? as u32;
        attr_offsets.push(offset);

        writer
            .write_u32::<LittleEndian>(fids.len() as u32)
            .map_err(|e| format!("Failed to write num features: {}", e))?;
        for &fid in fids {
            writer
                .write_u32::<LittleEndian>(fid)
                .map_err(|e| format!("Failed to write feature id: {}", e))?;
        }
    }

    let afrf_end = writer
        .stream_position()
        .map_err(|e| format!("Failed to get position: {}", e))? as u32;

    // Update AFRF size
    writer
        .seek(SeekFrom::Start(afrf_size_pos))
        .map_err(|e| format!("Failed to seek: {}", e))?;
    writer
        .write_u32::<LittleEndian>(afrf_end - off_attrrefs)
        .map_err(|e| format!("Failed to write AFRF size: {}", e))?;

    // Update offsets
    writer
        .seek(SeekFrom::Start(attr_offsets_pos))
        .map_err(|e| format!("Failed to seek: {}", e))?;
    for offset in &attr_offsets {
        writer
            .write_u32::<LittleEndian>(*offset)
            .map_err(|e| format!("Failed to write offset: {}", e))?;
    }

    writer
        .seek(SeekFrom::Start(afrf_end as u64))
        .map_err(|e| format!("Failed to seek: {}", e))?;

    let file_size = afrf_end;

    // Write header
    writer
        .seek(SeekFrom::Start(0))
        .map_err(|e| format!("Failed to seek: {}", e))?;

    writer
        .write_all(FILEMAGIC)
        .map_err(|e| format!("Failed to write magic: {}", e))?;
    writer
        .write_u32::<LittleEndian>(file_size)
        .map_err(|e| format!("Failed to write size: {}", e))?;
    writer
        .write_all(MODELTYPE)
        .map_err(|e| format!("Failed to write type: {}", e))?;
    writer
        .write_u32::<LittleEndian>(VERSION_NUMBER)
        .map_err(|e| format!("Failed to write version: {}", e))?;
    writer
        .write_u32::<LittleEndian>(features.len() as u32)
        .map_err(|e| format!("Failed to write num_features: {}", e))?;
    writer
        .write_u32::<LittleEndian>(labels.len() as u32)
        .map_err(|e| format!("Failed to write num_labels: {}", e))?;
    writer
        .write_u32::<LittleEndian>(attributes.len() as u32)
        .map_err(|e| format!("Failed to write num_attrs: {}", e))?;
    writer
        .write_u32::<LittleEndian>(off_features)
        .map_err(|e| format!("Failed to write off_features: {}", e))?;
    writer
        .write_u32::<LittleEndian>(off_labels)
        .map_err(|e| format!("Failed to write off_labels: {}", e))?;
    writer
        .write_u32::<LittleEndian>(off_attrs)
        .map_err(|e| format!("Failed to write off_attrs: {}", e))?;
    writer
        .write_u32::<LittleEndian>(off_labelrefs)
        .map_err(|e| format!("Failed to write off_labelrefs: {}", e))?;
    writer
        .write_u32::<LittleEndian>(off_attrrefs)
        .map_err(|e| format!("Failed to write off_attrrefs: {}", e))?;

    // Seek to end
    writer
        .seek(SeekFrom::Start(file_size as u64))
        .map_err(|e| format!("Failed to seek: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_hashlittle() {
        // Test with known values
        let h1 = hashlittle(b"test\0", 0);
        let h2 = hashlittle(b"test\0", 0);
        assert_eq!(h1, h2);

        let h3 = hashlittle(b"other\0", 0);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_cqdb_writer() {
        let mut cqdb = CQDBWriter::new();
        cqdb.put("B", 0);
        cqdb.put("I", 1);

        let mut buffer = Cursor::new(Vec::new());
        let size = cqdb.write(&mut buffer).unwrap();

        assert!(size > 0);
        let data = buffer.into_inner();
        assert_eq!(&data[0..4], b"CQDB");
    }
}
