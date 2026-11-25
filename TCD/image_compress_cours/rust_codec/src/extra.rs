//! Étages avancés (BWT, LZ78, LZW, arithmétique) et compression d'images
//! par tuiles. Ce module est inclus par `main.rs` et utilise ses primitives.

use super::*;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

const BWT_MAX_BYTES: usize = 32 * 1024 * 1024;
const BWT_AUTO_LIMIT: usize = 2 * 1024 * 1024;
const LZ_AUTO_LIMIT: usize = 4 * 1024 * 1024;
const ARITH_AUTO_LIMIT: usize = 32 * 1024 * 1024;
const LZ78_MAX_DICT: usize = 1 << 20;
const LZW_MAX_CODE: u32 = 1 << 20;
const ARITH_TOTAL_LIMIT: u32 = 1 << 15;

// ── Choix automatique de chaînes ──────────────────────────────────────────

pub fn auto_candidates(input_len: usize) -> Vec<Vec<Stage>> {
    let mut candidates = vec![
        vec![Stage::Lz77, Stage::Huffman],
        vec![Stage::Rle, Stage::Huffman],
        vec![Stage::Huffman],
        vec![Stage::Lz77],
        vec![Stage::Rle],
        vec![Stage::Delta, Stage::Huffman],
    ];
    if input_len <= BWT_AUTO_LIMIT {
        candidates.push(vec![Stage::Bwt, Stage::Rle, Stage::Huffman]);
        candidates.push(vec![Stage::Bwt, Stage::Huffman]);
        candidates.push(vec![Stage::Bwt, Stage::Rle, Stage::Arithmetic]);
    }
    if input_len <= LZ_AUTO_LIMIT {
        candidates.push(vec![Stage::Lz78, Stage::Huffman]);
        candidates.push(vec![Stage::Lzw, Stage::Huffman]);
        candidates.push(vec![Stage::Lz78]);
        candidates.push(vec![Stage::Lzw]);
        candidates.push(vec![Stage::Lz78, Stage::Arithmetic]);
        candidates.push(vec![Stage::Lzw, Stage::Arithmetic]);
    }
    if input_len <= ARITH_AUTO_LIMIT {
        candidates.push(vec![Stage::Lz77, Stage::Arithmetic]);
        candidates.push(vec![Stage::Rle, Stage::Arithmetic]);
        candidates.push(vec![Stage::Arithmetic]);
    }
    candidates
}

// ── BWT (Burrows-Wheeler) ─────────────────────────────────────────────────

pub fn bwt_encode(input: &[u8]) -> Result<(usize, Vec<u8>), String> {
    let n = input.len();
    if n == 0 {
        return Ok((0, Vec::new()));
    }
    if n > BWT_MAX_BYTES {
        return Err(format!(
            "BWT limité à {} Mo (fichier de {} Mo)",
            BWT_MAX_BYTES / (1024 * 1024),
            (n + 1024 * 1024 - 1) / (1024 * 1024)
        ));
    }
    if n > u32::MAX as usize {
        return Err("BWT : fichier trop grand pour les index 32 bits".into());
    }

    let mut sa = vec![0u32; n];
    let mut rank = vec![0u32; n];
    let mut tmp = vec![0u32; n];

    let mut count = [0u32; 256];
    for &byte in input {
        count[byte as usize] += 1;
    }
    let mut sum = 0u32;
    for slot in count.iter_mut() {
        let value = *slot;
        *slot = sum;
        sum += value;
    }
    for (index, &byte) in input.iter().enumerate() {
        let slot = count[byte as usize] as usize;
        sa[slot] = index as u32;
        count[byte as usize] += 1;
    }

    let mut classes = 1u32;
    rank[sa[0] as usize] = 0;
    for i in 1..n {
        if input[sa[i] as usize] != input[sa[i - 1] as usize] {
            classes += 1;
        }
        rank[sa[i] as usize] = classes - 1;
    }

    let mut k = 1usize;
    while k < n && classes < n as u32 && classes > 1 {
        for i in 0..n {
            let start = sa[i] as usize;
            tmp[i] = if start >= k {
                (start - k) as u32
            } else {
                (start + n - k) as u32
            };
        }

        let mut buckets = vec![0u32; classes as usize];
        for &start in &tmp {
            buckets[rank[start as usize] as usize] += 1;
        }
        let mut acc = 0u32;
        for bucket in buckets.iter_mut() {
            let value = *bucket;
            *bucket = acc;
            acc += value;
        }
        for &start in &tmp {
            let bucket = rank[start as usize] as usize;
            sa[buckets[bucket] as usize] = start;
            buckets[bucket] += 1;
        }

        tmp[sa[0] as usize] = 0;
        let mut new_classes = 1u32;
        for i in 1..n {
            let current = sa[i] as usize;
            let previous = sa[i - 1] as usize;
            let current_pair = (rank[current], rank[(current + k) % n]);
            let previous_pair = (rank[previous], rank[(previous + k) % n]);
            if current_pair != previous_pair {
                new_classes += 1;
            }
            tmp[current] = new_classes - 1;
        }
        rank.copy_from_slice(&tmp);
        classes = new_classes;
        k <<= 1;
    }

    let primary = sa
        .iter()
        .position(|&start| start == 0)
        .ok_or_else(|| "BWT : rotation primaire introuvable".to_string())?;
    let mut output = vec![0u8; n];
    for i in 0..n {
        let start = sa[i] as usize;
        output[i] = input[(start + n - 1) % n];
    }
    Ok((primary, output))
}

pub fn bwt_decode(data: &[u8], expected: usize, primary: usize) -> Result<Vec<u8>, String> {
    if data.len() != expected {
        return Err("taille BWT incohérente".into());
    }
    if expected == 0 {
        return Ok(Vec::new());
    }
    if primary >= expected {
        return Err("index primaire BWT invalide".into());
    }

    let mut counts = [0u32; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }
    let mut starts = [0u32; 256];
    let mut sum = 0u32;
    for i in 0..256 {
        starts[i] = sum;
        sum += counts[i];
    }
    let mut next = vec![0u32; expected];
    let mut occurrence = starts;
    for (index, &byte) in data.iter().enumerate() {
        let slot = occurrence[byte as usize] as usize;
        next[slot] = index as u32;
        occurrence[byte as usize] += 1;
    }

    let mut output = vec![0u8; expected];
    let mut cursor = primary;
    for byte in output.iter_mut() {
        cursor = next[cursor] as usize;
        *byte = data[cursor];
    }
    Ok(output)
}

// ── LZ78 ──────────────────────────────────────────────────────────────────

fn write_lz78_token(output: &mut Vec<u8>, index: u32, symbol: Option<u8>) {
    let tag = ((index as u64) << 1) | u64::from(symbol.is_some());
    write_varint(output, tag);
    if let Some(byte) = symbol {
        output.push(byte);
    }
}

pub fn lz78_encode(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut dictionary: HashMap<(u32, u8), u32> = HashMap::new();
    let mut next_index = 1u32;
    let mut prefix = 0u32;
    let mut output = Vec::new();

    for &byte in input {
        let key = (prefix, byte);
        if let Some(&code) = dictionary.get(&key) {
            prefix = code;
        } else {
            write_lz78_token(&mut output, prefix, Some(byte));
            if next_index < u32::MAX && dictionary.len() < LZ78_MAX_DICT {
                dictionary.insert(key, next_index);
                next_index += 1;
            }
            prefix = 0;
        }
    }
    if prefix != 0 {
        write_lz78_token(&mut output, prefix, None);
    }
    output
}

pub fn lz78_decode(data: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    let mut output = Vec::with_capacity(expected);
    let mut parents: Vec<u32> = vec![0];
    let mut symbols: Vec<u8> = vec![0];
    let mut position = 0usize;

    while output.len() < expected {
        let tag = read_varint(data, &mut position)?;
        let has_symbol = tag & 1 == 1;
        let index = (tag >> 1) as usize;
        if index >= parents.len() {
            return Err("index LZ78 invalide".into());
        }

        let mut chain = Vec::new();
        let mut cursor = index;
        let mut guard = 0usize;
        while cursor != 0 {
            if cursor >= parents.len() {
                return Err("dictionnaire LZ78 incohérent".into());
            }
            chain.push(symbols[cursor]);
            cursor = parents[cursor] as usize;
            guard += 1;
            if guard > parents.len() + 1 {
                return Err("cycle LZ78".into());
            }
        }
        chain.reverse();

        if has_symbol {
            if position >= data.len() {
                return Err("symbole LZ78 manquant".into());
            }
            let byte = data[position];
            position += 1;
            chain.push(byte);
            if parents.len() <= LZ78_MAX_DICT {
                parents.push(index as u32);
                symbols.push(byte);
            }
        }

        if output.len() + chain.len() > expected {
            return Err("LZ78 dépasse la taille attendue".into());
        }
        output.extend_from_slice(&chain);
    }

    if position != data.len() {
        return Err("flux LZ78 non consommé".into());
    }
    Ok(output)
}

// ── LZW ───────────────────────────────────────────────────────────────────

pub fn lzw_encode(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut dictionary: HashMap<(u32, u8), u32> = HashMap::new();
    let mut next_code = 256u32;
    let mut output = Vec::new();
    let mut prefix = input[0] as u32;

    for &byte in &input[1..] {
        let key = (prefix, byte);
        if let Some(&code) = dictionary.get(&key) {
            prefix = code;
        } else {
            write_varint(&mut output, prefix as u64);
            if next_code < LZW_MAX_CODE
                && dictionary.len() < (LZW_MAX_CODE as usize - 256)
            {
                dictionary.insert(key, next_code);
                next_code += 1;
            }
            prefix = byte as u32;
        }
    }
    write_varint(&mut output, prefix as u64);
    output
}

fn lzw_entry(code: u32, parents: &[u32], symbols: &[u8]) -> Result<Vec<u8>, String> {
    let mut chain = Vec::new();
    let mut cursor = code;
    let mut guard = 0usize;
    while cursor != u32::MAX {
        if cursor as usize >= parents.len() {
            return Err("dictionnaire LZW incohérent".into());
        }
        chain.push(symbols[cursor as usize]);
        cursor = parents[cursor as usize];
        guard += 1;
        if guard > parents.len() + 1 {
            return Err("cycle LZW".into());
        }
    }
    chain.reverse();
    Ok(chain)
}

pub fn lzw_decode(data: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    if expected == 0 {
        return if data.is_empty() {
            Ok(Vec::new())
        } else {
            Err("flux LZW non vide pour une entrée vide".into())
        };
    }

    let mut parents: Vec<u32> = vec![u32::MAX; 256];
    let mut symbols: Vec<u8> = (0..=255u8).collect();
    let mut next_code = 256u32;
    let mut position = 0usize;
    let mut output = Vec::with_capacity(expected);

    let first = read_varint(data, &mut position)? as u32;
    if first >= next_code {
        return Err("premier code LZW invalide".into());
    }
    let entry = lzw_entry(first, &parents, &symbols)?;
    if entry.len() > expected {
        return Err("LZW dépasse la taille attendue".into());
    }
    output.extend_from_slice(&entry);
    let mut previous = first;

    while output.len() < expected {
        let code = read_varint(data, &mut position)? as u32;
        let entry = if code < next_code {
            lzw_entry(code, &parents, &symbols)?
        } else if code == next_code {
            let mut value = lzw_entry(previous, &parents, &symbols)?;
            let first_byte = *value
                .first()
                .ok_or_else(|| "entrée LZW vide".to_string())?;
            value.push(first_byte);
            value
        } else {
            return Err("code LZW invalide".into());
        };

        if output.len() + entry.len() > expected {
            return Err("LZW dépasse la taille attendue".into());
        }
        output.extend_from_slice(&entry);
        if next_code < LZW_MAX_CODE {
            parents.push(previous);
            symbols.push(entry[0]);
            next_code += 1;
        }
        previous = code;
    }

    if position != data.len() {
        return Err("flux LZW non consommé".into());
    }
    Ok(output)
}

// ── Codage arithmétique statique (ordre 0) ────────────────────────────────

fn scaled_frequencies(counts: &[u64; 256], length: usize) -> [u32; 256] {
    let mut frequencies = [0u32; 256];
    if length == 0 {
        return frequencies;
    }
    if length <= ARITH_TOTAL_LIMIT as usize {
        for symbol in 0..256 {
            frequencies[symbol] = counts[symbol] as u32;
        }
        return frequencies;
    }

    let target = ARITH_TOTAL_LIMIT as u64;
    let mut total = 0u64;
    for symbol in 0..256 {
        if counts[symbol] > 0 {
            let value = ((counts[symbol] * target) / length as u64).max(1);
            frequencies[symbol] = value as u32;
            total += value;
        }
    }
    while total > target {
        let mut best: Option<usize> = None;
        for symbol in 0..256 {
            if frequencies[symbol] > 1
                && best.map_or(true, |index| frequencies[symbol] > frequencies[index])
            {
                best = Some(symbol);
            }
        }
        let Some(index) = best else { break };
        frequencies[index] -= 1;
        total -= 1;
    }
    frequencies
}

pub fn arithmetic_encode(input: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    let length = input.len();
    let mut counts = [0u64; 256];
    for &byte in input {
        counts[byte as usize] += 1;
    }
    let frequencies = scaled_frequencies(&counts, length);

    let mut metadata = length_meta(length);
    for &frequency in &frequencies {
        write_varint(&mut metadata, frequency as u64);
    }
    if length == 0 {
        return Ok((metadata, Vec::new()));
    }

    let mut cumulative = [0u32; 257];
    for symbol in 0..256 {
        cumulative[symbol + 1] = cumulative[symbol] + frequencies[symbol];
    }
    let total = cumulative[256];
    if total == 0 {
        return Err("table arithmétique vide".into());
    }

    let mut writer = BitWriter::new();
    let mut low = 0u64;
    let mut high = 0xFFFF_FFFFu64;
    let mut pending = 0u32;

    for &byte in input {
        let symbol = byte as usize;
        let range = high - low + 1;
        high = low + range * cumulative[symbol + 1] as u64 / total as u64 - 1;
        low += range * cumulative[symbol] as u64 / total as u64;

        loop {
            if high < 0x8000_0000 {
                writer.write(0, 1);
                for _ in 0..pending {
                    writer.write(1, 1);
                }
                pending = 0;
            } else if low >= 0x8000_0000 {
                writer.write(1, 1);
                for _ in 0..pending {
                    writer.write(0, 1);
                }
                pending = 0;
                low -= 0x8000_0000;
                high -= 0x8000_0000;
            } else if low >= 0x4000_0000 && high < 0xC000_0000 {
                pending += 1;
                low -= 0x4000_0000;
                high -= 0x4000_0000;
            } else {
                break;
            }
            low <<= 1;
            high = (high << 1) | 1;
        }
    }

    pending += 1;
    if low < 0x4000_0000 {
        writer.write(0, 1);
        for _ in 0..pending {
            writer.write(1, 1);
        }
    } else {
        writer.write(1, 1);
        for _ in 0..pending {
            writer.write(0, 1);
        }
    }
    Ok((metadata, writer.finish()))
}

struct PaddedBitReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> PaddedBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn read_bit(&mut self) -> u8 {
        let byte_index = self.position / 8;
        let bit_index = 7 - (self.position % 8);
        self.position += 1;
        if byte_index < self.data.len() {
            (self.data[byte_index] >> bit_index) & 1
        } else {
            0
        }
    }
}

pub fn arithmetic_decode(
    data: &[u8],
    expected: usize,
    metadata: &[u8],
) -> Result<Vec<u8>, String> {
    let mut position = 0usize;
    let mut frequencies = [0u32; 256];
    for frequency in frequencies.iter_mut() {
        *frequency = read_varint(metadata, &mut position)? as u32;
    }
    if position != metadata.len() {
        return Err("métadonnées arithmétiques non consommées".into());
    }
    let mut cumulative = [0u32; 257];
    for symbol in 0..256 {
        cumulative[symbol + 1] = cumulative[symbol] + frequencies[symbol];
    }
    let total = cumulative[256];
    if expected == 0 {
        return Ok(Vec::new());
    }
    if total == 0 {
        return Err("table arithmétique vide".into());
    }

    let mut lookup = vec![0u8; total as usize];
    for symbol in 0..256 {
        for value in cumulative[symbol]..cumulative[symbol + 1] {
            lookup[value as usize] = symbol as u8;
        }
    }

    let mut reader = PaddedBitReader::new(data);
    let mut low = 0u64;
    let mut high = 0xFFFF_FFFFu64;
    let mut code = 0u64;
    for _ in 0..32 {
        code = (code << 1) | reader.read_bit() as u64;
    }

    let mut output = Vec::with_capacity(expected);
    for _ in 0..expected {
        let range = high - low + 1;
        let value = ((code - low + 1) * total as u64 - 1) / range;
        if value >= total as u64 {
            return Err("valeur arithmétique invalide".into());
        }
        let symbol = lookup[value as usize];
        output.push(symbol);
        let index = symbol as usize;
        high = low + range * cumulative[index + 1] as u64 / total as u64 - 1;
        low += range * cumulative[index] as u64 / total as u64;

        loop {
            if high < 0x8000_0000 {
                // rien à soustraire
            } else if low >= 0x8000_0000 {
                code -= 0x8000_0000;
                low -= 0x8000_0000;
                high -= 0x8000_0000;
            } else if low >= 0x4000_0000 && high < 0xC000_0000 {
                code -= 0x4000_0000;
                low -= 0x4000_0000;
                high -= 0x4000_0000;
            } else {
                break;
            }
            low <<= 1;
            high = (high << 1) | 1;
            code = ((code << 1) | reader.read_bit() as u64) & 0xFFFF_FFFF;
        }
    }
    Ok(output)
}

// ── Compression d'images RGB par tuiles ───────────────────────────────────

const TILE_MAGIC: &[u8; 6] = b"RTIL1\n";
const TILE_VERSION: u8 = 1;
const TILE_MAX_SIZE: usize = 4096;

fn tile_dims(width: usize, height: usize, tile: usize, index: usize) -> (usize, usize, usize, usize) {
    let tiles_x = (width + tile - 1) / tile;
    let ty = index / tiles_x;
    let tx = index % tiles_x;
    let tw = (width - tx * tile).min(tile);
    let th = (height - ty * tile).min(tile);
    (tx, ty, tw, th)
}

fn tile_candidates(pipeline_spec: &str, _max_tile_bytes: usize) -> Result<Vec<Vec<Stage>>, String> {
    if pipeline_spec.eq_ignore_ascii_case("auto") {
        let candidates = vec![
            vec![Stage::Delta, Stage::Rle, Stage::Huffman],
            vec![Stage::Delta, Stage::Huffman],
            vec![Stage::Rle, Stage::Huffman],
            vec![Stage::Lz77, Stage::Huffman],
            vec![Stage::Huffman],
        ];
        Ok(candidates)
    } else {
        Ok(vec![parse_pipeline(pipeline_spec)?])
    }
}

fn encode_tile_block(tile: &[u8], stages: &[Stage]) -> Result<(Vec<u8>, usize), String> {
    let mut current: Option<Vec<u8>> = None;
    let mut metas: Vec<(u8, Vec<u8>)> = Vec::with_capacity(stages.len());
    for (index, stage) in stages.iter().enumerate() {
        let source = if index == 0 {
            tile
        } else {
            current.as_deref().unwrap_or(&[])
        };
        let (meta, data) = encode_stage(*stage, source)?;
        metas.push((stage.id(), meta));
        current = Some(data);
    }
    let payload = current.unwrap_or_default();
    let mut block = Vec::with_capacity(payload.len() + 32);
    block.extend_from_slice(&(tile.len() as u32).to_le_bytes());
    block.push(stages.len() as u8);
    for (id, meta) in &metas {
        block.push(*id);
        block.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        block.extend_from_slice(meta);
    }
    block.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    block.extend_from_slice(&payload);
    Ok((block, payload.len()))
}

fn decode_tile_block(
    data: &[u8],
    position: &mut usize,
    expected: usize,
) -> Result<Vec<u8>, String> {
    let raw_len = read_u32(data, position)? as usize;
    if raw_len != expected {
        return Err(format!(
            "taille de tuile incohérente : {raw_len} au lieu de {expected}"
        ));
    }
    if *position >= data.len() {
        return Err("tuile tronquée".into());
    }
    let stage_count = data[*position] as usize;
    *position += 1;
    let mut stages: Vec<(u8, Vec<u8>)> = Vec::with_capacity(stage_count);
    for _ in 0..stage_count {
        if *position >= data.len() {
            return Err("étage de tuile tronqué".into());
        }
        let id = data[*position];
        *position += 1;
        let meta_len = read_u32(data, position)? as usize;
        if *position + meta_len > data.len() {
            return Err("métadonnées de tuile tronquées".into());
        }
        let meta = data[*position..*position + meta_len].to_vec();
        *position += meta_len;
        stages.push((id, meta));
    }
    let payload_len = read_u32(data, position)? as usize;
    if *position + payload_len > data.len() {
        return Err("payload de tuile tronqué".into());
    }
    let mut current = data[*position..*position + payload_len].to_vec();
    *position += payload_len;
    for (id, meta) in stages.iter().rev() {
        current = decode_stage(*id, meta, &current)?;
    }
    if current.len() != raw_len {
        return Err("taille décompressée de tuile incohérente".into());
    }
    Ok(current)
}

fn build_tile_container(
    width: usize,
    height: usize,
    tile: usize,
    original_len: usize,
    hash: u64,
    blocks: &[Vec<u8>],
) -> Vec<u8> {
    let mut output = Vec::with_capacity(original_len / 2 + 1024);
    output.extend_from_slice(TILE_MAGIC);
    output.push(TILE_VERSION);
    output.extend_from_slice(&(width as u32).to_le_bytes());
    output.extend_from_slice(&(height as u32).to_le_bytes());
    output.extend_from_slice(&(tile as u16).to_le_bytes());
    output.extend_from_slice(&(original_len as u64).to_le_bytes());
    output.extend_from_slice(&hash.to_le_bytes());
    output.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    for block in blocks {
        output.extend_from_slice(block);
    }
    output
}

fn decompress_tile_container(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut position = 0usize;
    if data.len() < TILE_MAGIC.len() || &data[..TILE_MAGIC.len()] != TILE_MAGIC {
        return Err("ce fichier n'est pas un conteneur .rtile".into());
    }
    position += TILE_MAGIC.len();
    if position >= data.len() {
        return Err("conteneur .rtile tronqué".into());
    }
    let version = data[position];
    position += 1;
    if version != TILE_VERSION {
        return Err(format!("version .rtile non prise en charge : {version}"));
    }
    let width = read_u32(data, &mut position)? as usize;
    let height = read_u32(data, &mut position)? as usize;
    let tile = read_u16(data, &mut position)? as usize;
    let original_len = read_u64(data, &mut position)? as usize;
    let expected_hash = read_u64(data, &mut position)?;
    let tile_count = read_u32(data, &mut position)? as usize;
    if tile == 0 {
        return Err("taille de tuile nulle".into());
    }
    let expected_tiles = if width == 0 || height == 0 {
        0
    } else {
        ((width + tile - 1) / tile) * ((height + tile - 1) / tile)
    };
    if tile_count != expected_tiles {
        return Err(format!(
            "nombre de tuiles incohérent : {tile_count} au lieu de {expected_tiles}"
        ));
    }

    let mut output = vec![0u8; original_len];
    for index in 0..tile_count {
        let (tx, ty, tw, th) = tile_dims(width, height, tile, index);
        let block = decode_tile_block(data, &mut position, tw * th * 3)?;
        for row in 0..th {
            let source = row * tw * 3;
            let destination = ((ty * tile + row) * width + tx * tile) * 3;
            output[destination..destination + tw * 3]
                .copy_from_slice(&block[source..source + tw * 3]);
        }
    }
    if position != data.len() {
        return Err("conteneur .rtile non consommé".into());
    }
    if output.len() != original_len {
        return Err(format!(
            "taille d'image incohérente : {} au lieu de {original_len}",
            output.len()
        ));
    }
    if fnv1a(&output) != expected_hash {
        return Err("empreinte de l'image décompressée invalide".into());
    }
    Ok(output)
}

pub fn tile_compress_cmd(
    input_path: &str,
    width_arg: &str,
    height_arg: &str,
    tile_arg: &str,
    output_path: &str,
    pipeline_spec: &str,
) -> Result<(), String> {
    let width: usize = width_arg
        .parse()
        .map_err(|_| format!("largeur invalide : {width_arg}"))?;
    let height: usize = height_arg
        .parse()
        .map_err(|_| format!("hauteur invalide : {height_arg}"))?;
    let tile: usize = tile_arg
        .parse()
        .map_err(|_| format!("taille de tuile invalide : {tile_arg}"))?;
    if width == 0 || height == 0 {
        return Err("dimensions d'image nulles".into());
    }
    if tile == 0 || tile > TILE_MAX_SIZE {
        return Err(format!("taille de tuile hors limites (1..={TILE_MAX_SIZE})"));
    }
    let expected_len = width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| "dimensions d'image trop grandes".to_string())?;
    let raw = fs::read(input_path)
        .map_err(|e| format!("lecture impossible de {input_path} : {e}"))?;
    if raw.len() != expected_len {
        return Err(format!(
            "taille RGB incohérente : {} octets au lieu de {expected_len}",
            raw.len()
        ));
    }

    let tiles_x = (width + tile - 1) / tile;
    let tiles_y = (height + tile - 1) / tile;
    let tile_count = tiles_x * tiles_y;
    let candidates = tile_candidates(pipeline_spec, tile * tile * 3)?;
    let started = Instant::now();
    let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(tile_count);
    let mut payload_total = 0usize;
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut tile_buffer: Vec<u8> = Vec::with_capacity(tile * tile * 3);

    for index in 0..tile_count {
        let (tx, ty, tw, th) = tile_dims(width, height, tile, index);
        tile_buffer.clear();
        for row in 0..th {
            let start = ((ty * tile + row) * width + tx * tile) * 3;
            tile_buffer.extend_from_slice(&raw[start..start + tw * 3]);
        }

        let mut best: Option<(Vec<u8>, usize, String)> = None;
        for stages in &candidates {
            let (block, payload_len) = encode_tile_block(&tile_buffer, stages)?;
            let mut check_position = 0usize;
            let decoded = decode_tile_block(&block, &mut check_position, tile_buffer.len())?;
            if check_position != block.len() {
                return Err("bloc de tuile non consommé".into());
            }
            if decoded != tile_buffer {
                return Err(format!("tuile {index} non reconstruite exactement"));
            }
            let better = best
                .as_ref()
                .map(|(current, _, _)| block.len() < current.len())
                .unwrap_or(true);
            if better {
                best = Some((block, payload_len, pipeline_name(stages)));
            }
        }

        let (block, payload_len, name) =
            best.ok_or_else(|| "aucune chaîne de tuile disponible".to_string())?;
        payload_total += payload_len;
        *counts.entry(name).or_insert(0) += 1;
        blocks.push(block);
    }
    let encode_ms = started.elapsed().as_secs_f64() * 1000.0;

    let hash = fnv1a(&raw);
    let container = build_tile_container(width, height, tile, raw.len(), hash, &blocks);
    let started_decode = Instant::now();
    let decoded = decompress_tile_container(&container)?;
    let decode_ms = started_decode.elapsed().as_secs_f64() * 1000.0;
    if decoded != raw {
        return Err("vérification globale sans perte échouée".into());
    }
    fs::write(output_path, &container)
        .map_err(|e| format!("écriture impossible de {output_path} : {e}"))?;

    let mut count_items: Vec<(String, usize)> = counts.into_iter().collect();
    count_items.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let counts_text = count_items
        .iter()
        .map(|(name, count)| format!("{name}:{count}"))
        .collect::<Vec<_>>()
        .join(",");

    println!("status=ok");
    println!("tile_size={tile}");
    println!("tiles={tile_count}");
    println!("input_bytes={}", raw.len());
    println!("output_bytes={}", container.len());
    println!("payload_bytes={payload_total}");
    println!(
        "container_overhead={}",
        container.len().saturating_sub(payload_total)
    );
    println!("pipeline_counts={counts_text}");
    println!("encode_ms={encode_ms:.3}");
    println!("decode_ms={decode_ms:.3}");
    println!("exact=1");
    println!("hash={hash:016x}");
    Ok(())
}

pub fn tile_decompress_cmd(input_path: &str, output_path: &str) -> Result<(), String> {
    let compressed = fs::read(input_path)
        .map_err(|e| format!("lecture impossible de {input_path} : {e}"))?;
    let decoded = decompress_tile_container(&compressed)?;
    fs::write(output_path, &decoded)
        .map_err(|e| format!("écriture impossible de {output_path} : {e}"))?;
    println!("status=ok");
    println!("output_bytes={}", decoded.len());
    println!("hash={:016x}", fnv1a(&decoded));
    Ok(())
}

pub fn tile_verify_cmd(original_path: &str, compressed_path: &str) -> Result<(), String> {
    let original = fs::read(original_path)
        .map_err(|e| format!("lecture impossible de {original_path} : {e}"))?;
    let compressed = fs::read(compressed_path)
        .map_err(|e| format!("lecture impossible de {compressed_path} : {e}"))?;
    let decoded = decompress_tile_container(&compressed)?;
    let exact = decoded == original;
    println!("status=ok");
    println!("exact={}", if exact { 1 } else { 0 });
    println!("input_bytes={}", original.len());
    println!("output_bytes={}", decoded.len());
    println!("hash_original={:016x}", fnv1a(&original));
    println!("hash_decoded={:016x}", fnv1a(&decoded));
    Ok(())
}
