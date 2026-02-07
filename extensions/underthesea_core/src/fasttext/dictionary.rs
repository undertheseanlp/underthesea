use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{self, Read};

use super::args::Args;
use super::hash::fasttext_hash;

/// Entry type: word or label
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntryType {
    Word = 0,
    Label = 1,
}

/// A single dictionary entry (word or label).
#[derive(Debug, Clone)]
pub struct Entry {
    pub word: String,
    pub count: i64,
    pub entry_type: EntryType,
}

/// The FastText dictionary: vocabulary + n-gram generation.
pub struct Dictionary {
    entries: Vec<Entry>,
    /// word2int: hash-table mapping word string to entry index.
    /// Uses open addressing with linear probing, matching C++ implementation.
    word2int: Vec<i32>,
    nwords: i32,
    nlabels: i32,
    /// Total number of word tokens seen during training
    pub ntokens: i64,
    /// Pruned vocab index: maps old bucket hash -> new row index.
    /// Non-empty for quantized (.ftz) models where the input matrix was pruned.
    pruneidx: HashMap<i32, i32>,
    // Hyperparameters needed for n-gram computation
    bucket: i32,
    minn: i32,
    maxn: i32,
    word_ngrams: i32,
}

const MAX_VOCAB_SIZE: usize = 30000000;
const HASH_TABLE_SIZE: usize = MAX_VOCAB_SIZE;

impl Dictionary {
    /// Read the dictionary from a binary stream.
    pub fn load<R: Read>(reader: &mut R, args: &Args) -> io::Result<Self> {
        let vocab_size = reader.read_i32::<LittleEndian>()?;
        let nwords = reader.read_i32::<LittleEndian>()?;
        let nlabels = reader.read_i32::<LittleEndian>()?;
        let ntokens = reader.read_i64::<LittleEndian>()?;
        let pruneidx_size = reader.read_i64::<LittleEndian>()?;

        let mut entries = Vec::with_capacity(vocab_size as usize);
        for _ in 0..vocab_size {
            let word = read_null_terminated_string(reader)?;
            let count = reader.read_i64::<LittleEndian>()?;
            let type_raw = reader.read_i8()?;
            let entry_type = if type_raw == 0 {
                EntryType::Word
            } else {
                EntryType::Label
            };
            entries.push(Entry {
                word,
                count,
                entry_type,
            });
        }

        // Read prune index as a HashMap for O(1) lookup
        let mut pruneidx = HashMap::with_capacity(pruneidx_size as usize);
        for _ in 0..pruneidx_size {
            let first = reader.read_i32::<LittleEndian>()?;
            let second = reader.read_i32::<LittleEndian>()?;
            pruneidx.insert(first, second);
        }

        // Build word2int hash table
        let mut word2int = vec![-1i32; HASH_TABLE_SIZE];
        for (i, entry) in entries.iter().enumerate() {
            let mut h = find_hash(&word2int, &entry.word);
            // Linear probing - find empty slot
            while word2int[h] != -1 {
                h = (h + 1) % HASH_TABLE_SIZE;
            }
            word2int[h] = i as i32;
        }

        Ok(Dictionary {
            entries,
            word2int,
            nwords,
            nlabels,
            ntokens,
            pruneidx,
            bucket: args.bucket,
            minn: args.minn,
            maxn: args.maxn,
            word_ngrams: args.word_ngrams,
        })
    }

    /// Number of words (excluding labels) in the dictionary.
    pub fn nwords(&self) -> i32 {
        self.nwords
    }

    /// Number of labels in the dictionary.
    pub fn nlabels(&self) -> i32 {
        self.nlabels
    }

    /// Get the label string for the given label index (0-based among labels).
    pub fn get_label(&self, lid: i32) -> &str {
        let idx = self.nwords + lid;
        &self.entries[idx as usize].word
    }

    /// Get all label strings.
    pub fn get_labels(&self) -> Vec<String> {
        (0..self.nlabels)
            .map(|i| self.get_label(i).to_string())
            .collect()
    }

    /// Get label counts (frequencies) for building the HS Huffman tree.
    pub fn get_label_counts(&self) -> Vec<i64> {
        (0..self.nlabels)
            .map(|i| self.entries[(self.nwords + i) as usize].count)
            .collect()
    }

    /// Look up a word in the hash table. Returns the entry index or -1.
    pub fn get_id(&self, word: &str) -> i32 {
        let mut h = fasttext_hash(word.as_bytes()) as usize % HASH_TABLE_SIZE;
        loop {
            let id = self.word2int[h];
            if id == -1 {
                return -1;
            }
            if self.entries[id as usize].word == word {
                return id;
            }
            h = (h + 1) % HASH_TABLE_SIZE;
        }
    }

    /// Map a hash bucket ID to a feature ID, respecting pruneidx.
    fn push_hash(&self, features: &mut Vec<i32>, bucket_hash: i32) {
        if self.pruneidx.is_empty() {
            features.push(self.nwords + bucket_hash);
        } else if let Some(&mapped) = self.pruneidx.get(&bucket_hash) {
            features.push(self.nwords + mapped);
        }
    }

    /// Compute all input features for a line of text (whitespace-split tokens).
    /// Matches the C++ FastText getLine() behavior.
    pub fn get_line_features(&self, text: &str) -> Vec<i32> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let mut word_ids: Vec<i32> = Vec::new();
        let mut features: Vec<i32> = Vec::new();

        for token in &tokens {
            let word_id = self.get_id(token);
            if word_id >= 0 {
                word_ids.push(word_id);
                features.push(word_id);
            }

            // Character n-grams: skip for EOS token (C++ initNgrams skips EOS)
            if self.minn > 0 && *token != "</s>" {
                self.compute_char_ngrams(token, &mut features);
            }
        }

        // Word n-grams (bigrams, trigrams, etc.)
        if self.word_ngrams > 1 && !word_ids.is_empty() {
            self.compute_word_ngrams(&word_ids, &mut features);
        }

        // Add EOS token - only word ID, no char n-grams (C++ initNgrams skips EOS)
        let eos_id = self.get_id("</s>");
        if eos_id >= 0 {
            features.push(eos_id);
        }

        features
    }

    /// Compute character n-gram hashes for a word.
    fn compute_char_ngrams(&self, word: &str, features: &mut Vec<i32>) {
        let mut bounded = String::with_capacity(word.len() + 2);
        bounded.push('<');
        bounded.push_str(word);
        bounded.push('>');

        let bytes = bounded.as_bytes();
        let len = bytes.len();

        let mut char_boundaries: Vec<usize> = Vec::new();
        let mut i = 0;
        while i < len {
            char_boundaries.push(i);
            i += utf8_char_len(bytes[i]);
        }
        char_boundaries.push(len);

        let nchars = char_boundaries.len() - 1;

        for n in self.minn as usize..=self.maxn as usize {
            if n > nchars {
                break;
            }
            for start_char in 0..=(nchars - n) {
                let start = char_boundaries[start_char];
                let end = char_boundaries[start_char + n];

                // Skip the full word with markers
                if start_char == 0 && start_char + n == nchars {
                    continue;
                }

                let ngram = &bytes[start..end];
                let h = fasttext_hash(ngram);
                let bucket_hash = (h as i64 % self.bucket as i64) as i32;
                self.push_hash(features, bucket_hash);
            }
        }
    }

    /// Compute word n-gram hashes (bigrams, trigrams, etc.).
    fn compute_word_ngrams(&self, word_ids: &[i32], features: &mut Vec<i32>) {
        for i in 0..word_ids.len() {
            let mut h: u64 = word_ids[i] as u64;
            for (j, &wid) in word_ids.iter().enumerate().skip(i + 1) {
                if j - i >= self.word_ngrams as usize {
                    break;
                }
                h = h.wrapping_mul(116049371).wrapping_add(wid as u64);
                let bucket_hash = (h as i64 % self.bucket as i64) as i32;
                self.push_hash(features, bucket_hash);
            }
        }
    }
}

/// Read a null-terminated UTF-8 string from a binary stream.
fn read_null_terminated_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let mut bytes = Vec::new();
    loop {
        let b = reader.read_u8()?;
        if b == 0 {
            break;
        }
        bytes.push(b);
    }
    String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Find the hash slot for a word (open addressing with linear probing).
fn find_hash(word2int: &[i32], word: &str) -> usize {
    let mut h = fasttext_hash(word.as_bytes()) as usize % HASH_TABLE_SIZE;
    loop {
        if word2int[h] == -1 {
            return h;
        }
        h = (h + 1) % HASH_TABLE_SIZE;
    }
}

/// Get the byte length of a UTF-8 character from its first byte.
fn utf8_char_len(first_byte: u8) -> usize {
    if first_byte < 0x80 {
        1
    } else if first_byte < 0xE0 {
        2
    } else if first_byte < 0xF0 {
        3
    } else {
        4
    }
}
