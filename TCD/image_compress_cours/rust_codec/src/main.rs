//! Petit codec pédagogique sans perte, sans dépendance externe.
//!
//! Stages disponibles : Delta, RLE, BWT, LZ77, LZ78, LZW, Huffman et
//! arithmétique. Le conteneur `.rlab` stocke le pipeline, les métadonnées de
//! chaque étage et le payload final. La compression vérifie toujours le
//! round-trip. Le mode `tile` compresse une image RGB brute par tuiles, avec
//! choix adaptatif de la meilleure chaîne pour chaque tuile.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::convert::TryInto;
use std::env;
use std::fs;
use std::process;
use std::time::Instant;

mod extra;
use extra::*;

const MAGIC: &[u8; 6] = b"RLAB2\n";
const VERSION: u8 = 1;

const LZ_WINDOW: usize = 65_535;
const LZ_MIN_MATCH: usize = 4;
const LZ_MAX_MATCH: usize = 258;
const LZ_HASH_BITS: usize = 16;
const LZ_HASH_SIZE: usize = 1 << LZ_HASH_BITS;
const LZ_MAX_CHAIN: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Delta,
    Rle,
    Bwt,
    Lz77,
    Lz78,
    Lzw,
    Huffman,
    Arithmetic,
}

impl Stage {
    fn is_terminal(self) -> bool {
        matches!(self, Stage::Huffman | Stage::Arithmetic)
    }

    fn id(self) -> u8 {
        match self {
            Stage::Delta => 1,
            Stage::Rle => 2,
            Stage::Bwt => 3,
            Stage::Lz77 => 4,
            Stage::Lz78 => 5,
            Stage::Lzw => 6,
            Stage::Huffman => 7,
            Stage::Arithmetic => 8,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Stage::Delta => "delta",
            Stage::Rle => "rle",
            Stage::Bwt => "bwt",
            Stage::Lz77 => "lz77",
            Stage::Lz78 => "lz78",
            Stage::Lzw => "lzw",
            Stage::Huffman => "huffman",
            Stage::Arithmetic => "arithmetic",
        }
    }

    fn from_id(id: u8) -> Result<Self, String> {
        match id {
            1 => Ok(Stage::Delta),
            2 => Ok(Stage::Rle),
            3 => Ok(Stage::Bwt),
            4 => Ok(Stage::Lz77),
            5 => Ok(Stage::Lz78),
            6 => Ok(Stage::Lzw),
            7 => Ok(Stage::Huffman),
            8 => Ok(Stage::Arithmetic),
            _ => Err(format!("étage inconnu : {id}")),
        }
    }
}

fn parse_pipeline(spec: &str) -> Result<Vec<Stage>, String> {
    let mut stages = Vec::new();
    for part in spec.split('+') {
        let name = part.trim().to_ascii_lowercase();
        if name.is_empty() {
            continue;
        }
        let stage = match name.as_str() {
            "delta" => Stage::Delta,
            "rle" => Stage::Rle,
            "bwt" => Stage::Bwt,
            "lz77" | "lz" => Stage::Lz77,
            "lz78" => Stage::Lz78,
            "lzw" => Stage::Lzw,
            "huffman" => Stage::Huffman,
            "arithmetic" | "arith" | "arithmetique" => Stage::Arithmetic,
            other => {
                return Err(format!(
                    "étage Rust inconnu « {other} » (disponibles : delta, rle, bwt, lz77, lz78, lzw, huffman, arithmetic)"
                ))
            }
        };
        stages.push(stage);
    }
    if stages.is_empty() {
        return Err("pipeline vide".into());
    }
    for (i, stage) in stages.iter().enumerate() {
        if stage.is_terminal() && i + 1 != stages.len() {
            return Err("Huffman/Arithmétique doit être la dernière étape du pipeline".into());
        }
    }
    Ok(stages)
}

fn pipeline_name(stages: &[Stage]) -> String {
    stages
        .iter()
        .map(|stage| stage.name())
        .collect::<Vec<_>>()
        .join("+")
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("compress") if args.len() == 5 => {
            compress_cmd(&args[2], &args[3], &args[4])
        }
        Some("decompress") if args.len() == 4 => {
            decompress_cmd(&args[2], &args[3])
        }
        Some("verify") if args.len() == 4 => {
            verify_cmd(&args[2], &args[3])
        }
        Some("compare") if args.len() == 4 => {
            compare_cmd(&args[2], &args[3])
        }
        Some("tile") if args.len() == 8 => tile_compress_cmd(
            &args[2], &args[3], &args[4], &args[5], &args[6], &args[7],
        ),
        Some("tile-decompress") if args.len() == 4 => {
            tile_decompress_cmd(&args[2], &args[3])
        }
        Some("tile-verify") if args.len() == 4 => {
            tile_verify_cmd(&args[2], &args[3])
        }
        _ => Err(usage()),
    };

    if let Err(message) = result {
        println!("status=error");
        println!("message={}", message.replace('\n', " "));
        process::exit(1);
    }
}

fn usage() -> String {
    "usage:\n  rust_codec compress <entrée> <sortie.rlab> <pipeline|auto>\n  rust_codec decompress <entrée.rlab> <sortie>\n  rust_codec verify <original> <entrée.rlab>\n  rust_codec compare <entrée> <étages_max 1..5>\n  rust_codec tile <rgb> <largeur> <hauteur> <tuile> <sortie.rtile> <pipeline|auto>\n  rust_codec tile-decompress <entrée.rtile> <sortie.rgb>\n  rust_codec tile-verify <original.rgb> <entrée.rtile>".into()
}

fn compress_cmd(input_path: &str, output_path: &str, pipeline_spec: &str) -> Result<(), String> {
    let input = fs::read(input_path)
        .map_err(|e| format!("lecture impossible de {input_path} : {e}"))?;

    let candidates = if pipeline_spec.eq_ignore_ascii_case("auto") {
        auto_candidates(input.len())
    } else {
        vec![parse_pipeline(pipeline_spec)?]
    };

    let mut best: Option<(Vec<u8>, Vec<Stage>, usize, f64, f64)> = None;

    for stages in candidates {
        let started = Instant::now();
        let mut current: Option<Vec<u8>> = None;
        let mut metas: Vec<(u8, Vec<u8>)> = Vec::with_capacity(stages.len());

        for (index, stage) in stages.iter().enumerate() {
            let source = if index == 0 {
                input.as_slice()
            } else {
                current.as_deref().unwrap_or(&[])
            };
            let (meta, data) = encode_stage(*stage, source)?;
            metas.push((stage.id(), meta));
            current = Some(data);
        }
        let current = current.unwrap_or_default();
        let encode_ms = started.elapsed().as_secs_f64() * 1000.0;
        let payload_bytes = current.len();
        let container = build_container(&input, &stages, &metas, &current);

        let started = Instant::now();
        let decoded = decompress_container(&container)?;
        let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
        if decoded != input {
            return Err("vérification sans perte échouée pendant la compression".into());
        }

        let better = best
            .as_ref()
            .map(|(best_container, _, _, _, _)| container.len() < best_container.len())
            .unwrap_or(true);
        if better {
            best = Some((container, stages, payload_bytes, encode_ms, decode_ms));
        }
    }

    let (container, stages, payload_bytes, encode_ms, decode_ms) =
        best.ok_or_else(|| "aucun pipeline exécuté".to_string())?;
    fs::write(output_path, &container)
        .map_err(|e| format!("écriture impossible de {output_path} : {e}"))?;

    println!("status=ok");
    println!("pipeline={}", pipeline_name(&stages));
    println!("input_bytes={}", input.len());
    println!("output_bytes={}", container.len());
    println!("payload_bytes={payload_bytes}");
    println!("container_overhead={}", container.len().saturating_sub(payload_bytes));
    println!("encode_ms={encode_ms:.3}");
    println!("decode_ms={decode_ms:.3}");
    println!("exact=1");
    println!("hash={:016x}", fnv1a(&input));
    Ok(())
}

fn decompress_cmd(input_path: &str, output_path: &str) -> Result<(), String> {
    let compressed = fs::read(input_path)
        .map_err(|e| format!("lecture impossible de {input_path} : {e}"))?;
    let decoded = decompress_container(&compressed)?;
    fs::write(output_path, &decoded)
        .map_err(|e| format!("écriture impossible de {output_path} : {e}"))?;

    println!("status=ok");
    println!("output_bytes={}", decoded.len());
    println!("hash={:016x}", fnv1a(&decoded));
    Ok(())
}

fn verify_cmd(original_path: &str, compressed_path: &str) -> Result<(), String> {
    let original = fs::read(original_path)
        .map_err(|e| format!("lecture impossible de {original_path} : {e}"))?;
    let compressed = fs::read(compressed_path)
        .map_err(|e| format!("lecture impossible de {compressed_path} : {e}"))?;
    let decoded = decompress_container(&compressed)?;
    let exact = decoded == original;

    println!("status=ok");
    println!("exact={}", if exact { 1 } else { 0 });
    println!("input_bytes={}", original.len());
    println!("output_bytes={}", decoded.len());
    println!("hash_original={:016x}", fnv1a(&original));
    println!("hash_decoded={:016x}", fnv1a(&decoded));
    Ok(())
}

// ── Comparatif exhaustif des combinaisons ─────────────────────────────────

const COMPARE_STAGES: [Stage; 5] = [
    Stage::Rle,
    Stage::Bwt,
    Stage::Lz77,
    Stage::Lz78,
    Stage::Lzw,
];
const COMPARE_MAX_BYTES: usize = 2 * 1024 * 1024;

struct CompareRow {
    label: String,
    payload_bits: usize,
    encode_ms: f64,
    decode_ms: f64,
    lossless: bool,
}

fn compare_cmd(input_path: &str, max_stages_arg: &str) -> Result<(), String> {
    let max_stages: usize = max_stages_arg
        .parse()
        .map_err(|_| "le nombre d'étages doit être un entier".to_string())?;
    if !(1..=5).contains(&max_stages) {
        return Err("le comparatif accepte de 1 à 5 étages de préprocesseurs".into());
    }
    let input = fs::read(input_path)
        .map_err(|e| format!("lecture impossible de {input_path} : {e}"))?;
    if input.len() > COMPARE_MAX_BYTES {
        return Err(format!(
            "comparatif limité à {} Mo (fichier de {:.1} Mo)",
            COMPARE_MAX_BYTES / (1024 * 1024),
            input.len() as f64 / (1024.0 * 1024.0)
        ));
    }

    let started = Instant::now();
    let mut rows = Vec::with_capacity(1_000);
    let raw_bits = input.len() * 8;
    rows.push(CompareRow {
        label: "brut".into(),
        payload_bits: raw_bits,
        encode_ms: 0.0,
        decode_ms: 0.0,
        lossless: true,
    });

    let mut names = Vec::with_capacity(max_stages);
    let mut metas: Vec<(u8, Vec<u8>)> = Vec::with_capacity(max_stages);
    compare_walk(
        &input,
        &input,
        &mut names,
        &mut metas,
        0,
        0,
        max_stages,
        0.0,
        &mut rows,
    );

    rows.sort_by(|a, b| {
        b.lossless
            .cmp(&a.lossless)
            .then_with(|| a.payload_bits.cmp(&b.payload_bits))
            .then_with(|| {
                a.encode_ms
                    .partial_cmp(&b.encode_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let gain_ge0 = rows.iter().filter(|row| row.payload_bits <= raw_bits).count();
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;

    println!("status=ok");
    println!("input_bytes={}", input.len());
    println!("raw_bits={raw_bits}");
    println!("evaluated={}", rows.len());
    println!("gain_ge0={gain_ge0}");
    println!("hidden={}", rows.len() - gain_ge0);
    println!("max_stages={max_stages}");
    println!("preprocessors=rle,bwt,lz77,lz78,lzw");
    println!("total_ms={total_ms:.3}");
    for row in &rows {
        println!(
            "result={}\t{}\t{:.3}\t{:.3}\t{}",
            row.label,
            row.payload_bits,
            row.encode_ms,
            row.decode_ms,
            u8::from(row.lossless)
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compare_walk(
    original: &[u8],
    data: &[u8],
    names: &mut Vec<Stage>,
    metas: &mut Vec<(u8, Vec<u8>)>,
    used: u32,
    depth: usize,
    max_stages: usize,
    encode_ms: f64,
    rows: &mut Vec<CompareRow>,
) {
    if !names.is_empty() {
        let started = Instant::now();
        let lossless = decode_stages(data.to_vec(), metas)
            .map(|decoded| decoded == original)
            .unwrap_or(false);
        let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
        rows.push(CompareRow {
            label: pipeline_name(names),
            payload_bits: data.len() * 8,
            encode_ms,
            decode_ms,
            lossless,
        });
    }

    for coder in [Stage::Huffman, Stage::Arithmetic] {
        let started = Instant::now();
        let Ok((coder_meta, coded)) = encode_stage(coder, data) else {
            continue;
        };
        let coder_ms = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let lossless = decode_stage(coder.id(), &coder_meta, &coded)
            .and_then(|decoded| {
                if decoded != data {
                    return Err("le codeur ne restitue pas son entrée".into());
                }
                decode_stages(decoded, metas)
            })
            .map(|decoded| decoded == original)
            .unwrap_or(false);
        let decode_ms = started.elapsed().as_secs_f64() * 1000.0;

        let mut label = names.clone();
        label.push(coder);
        rows.push(CompareRow {
            label: pipeline_name(&label),
            payload_bits: coded.len() * 8,
            encode_ms: encode_ms + coder_ms,
            decode_ms,
            lossless,
        });
    }

    if depth >= max_stages {
        return;
    }

    for (index, stage) in COMPARE_STAGES.iter().enumerate() {
        if used & (1 << index) != 0 {
            continue;
        }
        let started = Instant::now();
        let Ok((meta, output)) = encode_stage(*stage, data) else {
            continue;
        };
        let stage_ms = started.elapsed().as_secs_f64() * 1000.0;
        if decode_stage(stage.id(), &meta, &output).ok().as_deref() != Some(data) {
            continue;
        }

        names.push(*stage);
        metas.push((stage.id(), meta));
        compare_walk(
            original,
            &output,
            names,
            metas,
            used | (1 << index),
            depth + 1,
            max_stages,
            encode_ms + stage_ms,
            rows,
        );
        names.pop();
        metas.pop();
    }
}

fn decode_stages(
    mut data: Vec<u8>,
    metas: &[(u8, Vec<u8>)],
) -> Result<Vec<u8>, String> {
    for (id, meta) in metas.iter().rev() {
        data = decode_stage(*id, meta, &data)?;
    }
    Ok(data)
}

fn encode_stage(stage: Stage, input: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    match stage {
        Stage::Delta => {
            let mut out = Vec::with_capacity(input.len());
            let mut previous = 0u8;
            for &byte in input {
                out.push(byte.wrapping_sub(previous));
                previous = byte;
            }
            Ok((length_meta(input.len()), out))
        }
        Stage::Rle => Ok((length_meta(input.len()), rle_encode(input))),
        Stage::Bwt => {
            let (primary, data) = bwt_encode(input)?;
            let mut meta = length_meta(input.len());
            meta.extend_from_slice(&(primary as u64).to_le_bytes());
            Ok((meta, data))
        }
        Stage::Lz77 => Ok((length_meta(input.len()), lz77_encode(input))),
        Stage::Lz78 => Ok((length_meta(input.len()), lz78_encode(input))),
        Stage::Lzw => Ok((length_meta(input.len()), lzw_encode(input))),
        Stage::Arithmetic => arithmetic_encode(input),
        Stage::Huffman => {
            let lengths = huffman_lengths(input)?;
            let data = huffman_encode(input, &lengths)?;
            let mut meta = length_meta(input.len());
            meta.extend_from_slice(&lengths);
            Ok((meta, data))
        }
    }
}

fn decode_stage(id: u8, meta: &[u8], data: &[u8]) -> Result<Vec<u8>, String> {
    if meta.len() < 8 {
        return Err("métadonnées d'étage trop courtes".into());
    }
    let expected = u64::from_le_bytes(meta[0..8].try_into().unwrap()) as usize;
    match Stage::from_id(id)? {
        Stage::Delta => {
            if data.len() != expected {
                return Err("taille Delta incohérente".into());
            }
            let mut out = Vec::with_capacity(expected);
            let mut previous = 0u8;
            for &byte in data {
                let value = byte.wrapping_add(previous);
                out.push(value);
                previous = value;
            }
            Ok(out)
        }
        Stage::Rle => rle_decode(data, expected),
        Stage::Bwt => {
            if meta.len() != 16 {
                return Err("métadonnées BWT invalides".into());
            }
            let primary =
                u64::from_le_bytes(meta[8..16].try_into().unwrap()) as usize;
            bwt_decode(data, expected, primary)
        }
        Stage::Lz77 => lz77_decode(data, expected),
        Stage::Lz78 => lz78_decode(data, expected),
        Stage::Lzw => lzw_decode(data, expected),
        Stage::Arithmetic => arithmetic_decode(data, expected, &meta[8..]),
        Stage::Huffman => {
            if meta.len() != 8 + 256 {
                return Err("métadonnées Huffman invalides".into());
            }
            let mut lengths = [0u8; 256];
            lengths.copy_from_slice(&meta[8..264]);
            huffman_decode(data, expected, &lengths)
        }
    }
}

fn length_meta(length: usize) -> Vec<u8> {
    (length as u64).to_le_bytes().to_vec()
}

fn build_container(
    original: &[u8],
    stages: &[Stage],
    metas: &[(u8, Vec<u8>)],
    payload: &[u8],
) -> Vec<u8> {
    let name = pipeline_name(stages);
    let mut out = Vec::with_capacity(payload.len() + 256);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&(original.len() as u64).to_le_bytes());
    out.extend_from_slice(&fnv1a(original).to_le_bytes());
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.push(metas.len() as u8);
    for (id, meta) in metas {
        out.push(*id);
        out.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        out.extend_from_slice(meta);
    }
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

fn decompress_container(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut pos = 0usize;
    if data.len() < MAGIC.len() || &data[..MAGIC.len()] != MAGIC {
        return Err("ce fichier n'est pas un conteneur .rlab".into());
    }
    pos += MAGIC.len();
    if pos >= data.len() {
        return Err("conteneur tronqué".into());
    }
    let version = data[pos];
    pos += 1;
    if version != VERSION {
        return Err(format!("version .rlab non prise en charge : {version}"));
    }
    let original_len = read_u64(data, &mut pos)? as usize;
    let expected_hash = read_u64(data, &mut pos)?;
    let name_len = read_u16(data, &mut pos)? as usize;
    if pos + name_len > data.len() {
        return Err("nom de pipeline tronqué".into());
    }
    pos += name_len;
    if pos >= data.len() {
        return Err("nombre d'étages manquant".into());
    }
    let stage_count = data[pos] as usize;
    pos += 1;
    let mut stages = Vec::with_capacity(stage_count);
    for _ in 0..stage_count {
        if pos >= data.len() {
            return Err("étage tronqué".into());
        }
        let id = data[pos];
        pos += 1;
        let meta_len = read_u32(data, &mut pos)? as usize;
        if pos + meta_len > data.len() {
            return Err("métadonnées d'étage tronquées".into());
        }
        let meta = data[pos..pos + meta_len].to_vec();
        pos += meta_len;
        stages.push((id, meta));
    }
    let payload_len = read_u64(data, &mut pos)? as usize;
    if pos + payload_len != data.len() {
        return Err("taille du payload incohérente".into());
    }
    let mut current = data[pos..].to_vec();
    for (id, meta) in stages.iter().rev() {
        current = decode_stage(*id, meta, &current)?;
    }
    if current.len() != original_len {
        return Err(format!(
            "taille décompressée incohérente : {} au lieu de {original_len}",
            current.len()
        ));
    }
    if fnv1a(&current) != expected_hash {
        return Err("empreinte du contenu décompressé invalide".into());
    }
    Ok(current)
}

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, String> {
    if *pos + 2 > data.len() {
        return Err("conteneur tronqué (u16)".into());
    }
    let value = u16::from_le_bytes(data[*pos..*pos + 2].try_into().unwrap());
    *pos += 2;
    Ok(value)
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() {
        return Err("conteneur tronqué (u32)".into());
    }
    let value = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(value)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > data.len() {
        return Err("conteneur tronqué (u64)".into());
    }
    let value = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(value)
}

fn fnv1a(data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

// ── RLE ────────────────────────────────────────────────────────────────────

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn read_varint(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        if *pos >= data.len() {
            return Err("flux RLE tronqué".into());
        }
        let byte = data[*pos];
        *pos += 1;
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 64 {
            return Err("varint RLE trop long".into());
        }
    }
}

fn rle_encode(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < input.len() {
        let byte = input[i];
        let mut j = i + 1;
        while j < input.len() && input[j] == byte {
            j += 1;
        }
        write_varint(&mut out, (j - i) as u64);
        out.push(byte);
        i = j;
    }
    out
}

fn rle_decode(data: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(expected);
    let mut pos = 0usize;
    while out.len() < expected {
        let count = read_varint(data, &mut pos)? as usize;
        if count == 0 {
            return Err("run RLE nul".into());
        }
        if pos >= data.len() {
            return Err("octet RLE manquant".into());
        }
        let byte = data[pos];
        pos += 1;
        if out.len() + count > expected {
            return Err("run RLE dépasse la taille attendue".into());
        }
        out.resize(out.len() + count, byte);
    }
    if pos != data.len() {
        return Err("flux RLE non consommé".into());
    }
    Ok(out)
}

// ── LZ77 (format proche de LZ4) ────────────────────────────────────────────

fn lz_hash(data: &[u8], pos: usize) -> usize {
    let value = ((data[pos] as u32) << 16)
        | ((data[pos + 1] as u32) << 8)
        | data[pos + 2] as u32;
    ((value.wrapping_mul(2_654_435_761)) >> (32 - LZ_HASH_BITS)) as usize
}

fn emit_length(out: &mut Vec<u8>, mut length: usize) {
    while length >= 255 {
        out.push(255);
        length -= 255;
    }
    out.push(length as u8);
}

fn emit_sequence(out: &mut Vec<u8>, literals: &[u8], offset: u16, match_len: usize) {
    let literal_len = literals.len();
    let match_code = match_len - LZ_MIN_MATCH;
    let token = ((literal_len.min(15) as u8) << 4) | (match_code.min(15) as u8);
    out.push(token);
    if literal_len >= 15 {
        emit_length(out, literal_len - 15);
    }
    out.extend_from_slice(literals);
    out.extend_from_slice(&offset.to_le_bytes());
    if match_code >= 15 {
        emit_length(out, match_code - 15);
    }
}

fn emit_final_literals(out: &mut Vec<u8>, literals: &[u8]) {
    let literal_len = literals.len();
    out.push((literal_len.min(15) as u8) << 4);
    if literal_len >= 15 {
        emit_length(out, literal_len - 15);
    }
    out.extend_from_slice(literals);
}

fn lz77_encode(input: &[u8]) -> Vec<u8> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    // Fenêtre glissante : la mémoire reste constante même sur un très gros fichier.
    let mut head = vec![-1i64; LZ_HASH_SIZE];
    let mut previous = vec![-1i64; LZ_WINDOW];
    let mut literals: Vec<u8> = Vec::new();
    let mut i = 0usize;

    while i < n {
        let mut best_len = 0usize;
        let mut best_offset = 0usize;

        if i + 3 <= n {
            let hash = lz_hash(input, i);
            let mut candidate = head[hash];
            let mut chain = 0usize;
            while candidate >= 0 && chain < LZ_MAX_CHAIN {
                let c = candidate as usize;
                let distance = i.saturating_sub(c);
                if distance > LZ_WINDOW {
                    break;
                }
                let max_len = (n - i).min(LZ_MAX_MATCH);
                let mut length = 0usize;
                while length < max_len && input[c + length] == input[i + length] {
                    length += 1;
                }
                if length > best_len {
                    best_len = length;
                    best_offset = distance;
                    if length == max_len {
                        break;
                    }
                }
                candidate = previous[c % LZ_WINDOW];
                chain += 1;
            }
        }

        if best_len >= LZ_MIN_MATCH {
            emit_sequence(&mut out, &literals, best_offset as u16, best_len);
            literals.clear();
            let end = (i + best_len).min(n);
            for position in i..end {
                if position + 3 <= n {
                    let hash = lz_hash(input, position);
                    previous[position % LZ_WINDOW] = head[hash];
                    head[hash] = position as i64;
                }
            }
            i += best_len;
        } else {
            literals.push(input[i]);
            if i + 3 <= n {
                let hash = lz_hash(input, i);
                previous[i % LZ_WINDOW] = head[hash];
                head[hash] = i as i64;
            }
            i += 1;
        }
    }

    if !literals.is_empty() {
        emit_final_literals(&mut out, &literals);
    }
    out
}

fn read_extended_length(data: &[u8], pos: &mut usize) -> Result<usize, String> {
    let mut total = 0usize;
    loop {
        if *pos >= data.len() {
            return Err("extension LZ77 tronquée".into());
        }
        let byte = data[*pos];
        *pos += 1;
        total += byte as usize;
        if byte != 255 {
            return Ok(total);
        }
    }
}

fn lz77_decode(data: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(expected);
    let mut pos = 0usize;

    while out.len() < expected {
        if pos >= data.len() {
            return Err("flux LZ77 tronqué".into());
        }
        let token = data[pos];
        pos += 1;

        let mut literal_len = (token >> 4) as usize;
        if literal_len == 15 {
            literal_len += read_extended_length(data, &mut pos)?;
        }
        if pos + literal_len > data.len() {
            return Err("littéraux LZ77 tronqués".into());
        }
        if out.len() + literal_len > expected {
            return Err("littéraux LZ77 dépassent la taille attendue".into());
        }
        out.extend_from_slice(&data[pos..pos + literal_len]);
        pos += literal_len;

        if out.len() == expected {
            break;
        }

        if pos + 2 > data.len() {
            return Err("offset LZ77 tronqué".into());
        }
        let offset = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if offset == 0 || offset > out.len() {
            return Err("offset LZ77 invalide".into());
        }

        let mut match_len = (token & 0x0f) as usize + LZ_MIN_MATCH;
        if token & 0x0f == 15 {
            match_len += read_extended_length(data, &mut pos)?;
        }
        if out.len() + match_len > expected {
            return Err("match LZ77 dépasse la taille attendue".into());
        }
        let start = out.len() - offset;
        for index in 0..match_len {
            let byte = out[start + index];
            out.push(byte);
        }
    }

    if pos != data.len() {
        return Err("flux LZ77 non consommé".into());
    }
    Ok(out)
}

// ── Huffman ────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct HuffmanNode {
    left: i32,
    right: i32,
    symbol: i32,
}

fn huffman_lengths(data: &[u8]) -> Result<[u8; 256], String> {
    let mut frequencies = [0u64; 256];
    for &byte in data {
        frequencies[byte as usize] += 1;
    }

    let mut nodes: Vec<HuffmanNode> = Vec::new();
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    for symbol in 0..256 {
        if frequencies[symbol] > 0 {
            nodes.push(HuffmanNode {
                left: -1,
                right: -1,
                symbol: symbol as i32,
            });
            heap.push(Reverse((frequencies[symbol], nodes.len() - 1)));
        }
    }

    let mut lengths = [0u8; 256];
    if nodes.is_empty() {
        return Ok(lengths);
    }
    if nodes.len() == 1 {
        lengths[nodes[0].symbol as usize] = 1;
        return Ok(lengths);
    }

    while heap.len() > 1 {
        let Reverse((freq_left, left)) = heap.pop().unwrap();
        let Reverse((freq_right, right)) = heap.pop().unwrap();
        nodes.push(HuffmanNode {
            left: left as i32,
            right: right as i32,
            symbol: -1,
        });
        heap.push(Reverse((freq_left + freq_right, nodes.len() - 1)));
    }

    let root = nodes.len() - 1;
    let mut stack = vec![(root, 0u8)];
    while let Some((index, depth)) = stack.pop() {
        let node = &nodes[index];
        if node.symbol >= 0 {
            if depth == 0 || depth > 32 {
                return Err("profondeur Huffman invalide (fréquences trop déséquilibrées)".into());
            }
            lengths[node.symbol as usize] = depth;
        } else {
            stack.push((node.left as usize, depth + 1));
            stack.push((node.right as usize, depth + 1));
        }
    }
    Ok(lengths)
}

fn canonical_codes(lengths: &[u8; 256]) -> Result<(Vec<u64>, Vec<u8>, u8), String> {
    let mut max_len = 0u8;
    for &length in lengths {
        max_len = max_len.max(length);
    }
    if max_len == 0 {
        return Ok((vec![0; 256], vec![0; 256], 0));
    }
    if max_len > 32 {
        return Err("code Huffman de plus de 32 bits".into());
    }

    let mut count = vec![0u32; max_len as usize + 1];
    for &length in lengths {
        if length > 0 {
            count[length as usize] += 1;
        }
    }
    let mut next_code = vec![0u64; max_len as usize + 1];
    let mut code = 0u64;
    for bits in 1..=max_len as usize {
        code = (code + count[bits - 1] as u64) << 1;
        next_code[bits] = code;
    }

    let mut codes = vec![0u64; 256];
    let mut used_lengths = vec![0u8; 256];
    for symbol in 0..256 {
        let length = lengths[symbol];
        if length > 0 {
            codes[symbol] = next_code[length as usize];
            next_code[length as usize] += 1;
            used_lengths[symbol] = length;
        }
    }
    Ok((codes, used_lengths, max_len))
}

struct BitWriter {
    out: Vec<u8>,
    accumulator: u64,
    nbits: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            accumulator: 0,
            nbits: 0,
        }
    }

    fn write(&mut self, code: u64, length: u8) {
        if length == 0 {
            return;
        }
        self.accumulator = (self.accumulator << length)
            | (code & ((1u64 << length) - 1));
        self.nbits += length;
        while self.nbits >= 8 {
            self.nbits -= 8;
            self.out.push((self.accumulator >> self.nbits) as u8);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out
                .push((self.accumulator << (8 - self.nbits)) as u8);
        }
        self.out
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    accumulator: u64,
    nbits: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            accumulator: 0,
            nbits: 0,
        }
    }

    fn fill(&mut self, wanted: u8) {
        while self.nbits < wanted && self.pos < self.data.len() {
            self.accumulator = (self.accumulator << 8) | self.data[self.pos] as u64;
            self.pos += 1;
            self.nbits += 8;
        }
    }

    fn peek(&mut self, count: u8) -> u64 {
        self.fill(count);
        if self.nbits >= count {
            (self.accumulator >> (self.nbits - count)) & ((1u64 << count) - 1)
        } else {
            let available = self.nbits;
            let value = self.accumulator & ((1u64 << available) - 1);
            value << (count - available)
        }
    }

    fn consume(&mut self, count: u8) -> Result<(), String> {
        if self.nbits < count {
            return Err("flux Huffman tronqué".into());
        }
        self.nbits -= count;
        Ok(())
    }

    fn read_bit(&mut self) -> Result<u8, String> {
        self.fill(1);
        if self.nbits == 0 {
            return Err("flux Huffman tronqué".into());
        }
        self.nbits -= 1;
        Ok(((self.accumulator >> self.nbits) & 1) as u8)
    }
}

fn huffman_encode(data: &[u8], lengths: &[u8; 256]) -> Result<Vec<u8>, String> {
    let (codes, used_lengths, _) = canonical_codes(lengths)?;
    let mut writer = BitWriter::new();
    for &byte in data {
        writer.write(codes[byte as usize], used_lengths[byte as usize]);
    }
    Ok(writer.finish())
}

fn build_decode_table(lengths: &[u8; 256], max_len: u8) -> Result<Vec<u16>, String> {
    let (codes, used_lengths, _) = canonical_codes(lengths)?;
    let size = 1usize << max_len;
    let mut table = vec![0u16; size];
    for symbol in 0..256 {
        let length = used_lengths[symbol];
        if length == 0 {
            continue;
        }
        let shift = max_len - length;
        let start = (codes[symbol] << shift) as usize;
        let count = 1usize << shift;
        let entry = ((length as u16) << 8) | symbol as u16;
        for slot in 0..count {
            table[start + slot] = entry;
        }
    }
    Ok(table)
}

fn huffman_decode(
    data: &[u8],
    expected: usize,
    lengths: &[u8; 256],
) -> Result<Vec<u8>, String> {
    if expected == 0 {
        return Ok(Vec::new());
    }
    let (_, used_lengths, max_len) = canonical_codes(lengths)?;
    if max_len == 0 {
        return Err("table Huffman vide".into());
    }
    let mut out = Vec::with_capacity(expected);

    if max_len <= 16 {
        let table = build_decode_table(lengths, max_len)?;
        let mut reader = BitReader::new(data);
        while out.len() < expected {
            let bits = reader.peek(max_len) as usize;
            let entry = table[bits];
            if entry == 0 {
                return Err("code Huffman invalide".into());
            }
            let length = (entry >> 8) as u8;
            let symbol = (entry & 0xff) as u8;
            reader.consume(length)?;
            out.push(symbol);
        }
    } else {
        let mut count = vec![0u32; max_len as usize + 1];
        for &length in used_lengths.iter() {
            if length > 0 {
                count[length as usize] += 1;
            }
        }
        let mut first_code = vec![0u64; max_len as usize + 1];
        let mut code = 0u64;
        for bits in 1..=max_len as usize {
            code = (code + count[bits - 1] as u64) << 1;
            first_code[bits] = code;
        }
        let mut first_symbol = vec![0u32; max_len as usize + 1];
        let mut offset = 0u32;
        for bits in 1..=max_len as usize {
            first_symbol[bits] = offset;
            offset += count[bits];
        }
        let mut symbols = Vec::with_capacity(offset as usize);
        for bits in 1..=max_len as usize {
            for symbol in 0..256 {
                if used_lengths[symbol] == bits as u8 {
                    symbols.push(symbol as u8);
                }
            }
        }

        let mut reader = BitReader::new(data);
        while out.len() < expected {
            let mut code = 0u64;
            let mut found = false;
            for bits in 1..=max_len as usize {
                code = (code << 1) | reader.read_bit()? as u64;
                let count_bits = count[bits];
                if count_bits > 0
                    && code >= first_code[bits]
                    && code < first_code[bits] + count_bits as u64
                {
                    let index =
                        first_symbol[bits] as usize + (code - first_code[bits]) as usize;
                    out.push(symbols[index]);
                    found = true;
                    break;
                }
            }
            if !found {
                return Err("code Huffman invalide".into());
            }
        }
    }

    Ok(out)
}
