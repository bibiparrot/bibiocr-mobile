use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig,
    OfflineTtsVitsModelConfig,
};
use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

static ENGINE: OnceLock<Mutex<Option<OfflineTts>>> = OnceLock::new();
// ponytail: Session-only cache; use disk/LRU if long documents cause memory pressure.
static AUDIO_CACHE: OnceLock<Mutex<HashMap<String, Arc<TtsAudio>>>> = OnceLock::new();

pub struct TtsAudio {
    pub samples: Vec<i16>,
    pub sample_rate: i32,
}

pub fn synthesize_sentence(
    sentence: &str,
    model_dir: &Path,
    stop: &AtomicBool,
) -> Result<Arc<TtsAudio>, String> {
    let text = sentence.trim().replace('\0', "");
    if text.is_empty() {
        return Err("There is no text to read".to_owned());
    }
    if stop.load(Ordering::Relaxed) {
        return Err("TTS stopped".to_owned());
    }
    let key = text.clone();
    let cache = AUDIO_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(audio) = cache.lock().map_err(|error| error.to_string())?.get(&key) {
        return Ok(Arc::clone(audio));
    }
    let mut engine = ENGINE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|error| error.to_string())?;
    if engine.is_none() {
        let path = |name: &str| model_dir.join(name).to_string_lossy().into_owned();
        let config = OfflineTtsConfig {
            model: OfflineTtsModelConfig {
                vits: OfflineTtsVitsModelConfig {
                    model: Some(path("melo-model.onnx")),
                    lexicon: Some(path("melo-lexicon.txt")),
                    tokens: Some(path("melo-tokens.txt")),
                    ..Default::default()
                },
                num_threads: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        *engine = Some(OfflineTts::create(&config).ok_or("Melo TTS could not load model files")?);
    }
    let engine = engine.as_mut().ok_or("Melo TTS did not initialize")?;
    let spoken = speakable_numbers(&text);
    let audio = engine
        .generate_with_config(
            &spoken,
            &GenerationConfig {
                speed: 1.0,
                ..Default::default()
            },
            None::<fn(&[f32], f32) -> bool>,
        )
        .ok_or("Melo TTS failed to generate audio")?;
    let pcm = audio
        .samples()
        .iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)
        .collect::<Vec<_>>();
    if pcm.is_empty() {
        return Err("Melo TTS generated no audio".to_owned());
    }
    let result = Arc::new(TtsAudio {
        samples: pcm,
        sample_rate: engine.sample_rate(),
    });
    cache
        .lock()
        .map_err(|error| error.to_string())?
        .insert(key, Arc::clone(&result));
    Ok(result)
}

pub fn synthesize_resilient(
    sentence: &str,
    model_dir: &Path,
    stop: &AtomicBool,
) -> Result<Vec<Arc<TtsAudio>>, String> {
    synthesize_resilient_with(sentence, |part| synthesize_sentence(part, model_dir, stop))
}

fn synthesize_resilient_with<T>(
    sentence: &str,
    mut synthesize: impl FnMut(&str) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    match synthesize(sentence) {
        Ok(audio) => Ok(vec![audio]),
        Err(error) => {
            if error != "Melo TTS failed to generate audio" {
                return Err(error);
            }
            let mut parts = sentence
                .split(|ch: char| !ch.is_alphanumeric())
                .filter(|part| !part.is_empty())
                .flat_map(|part| {
                    part.chars()
                        .collect::<Vec<_>>()
                        .chunks(24)
                        .map(|chars| chars.iter().collect::<String>())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            if parts.len() == 1 && parts[0] == sentence {
                let chars = sentence.chars().collect::<Vec<_>>();
                if chars.len() == 1 {
                    return Err(error);
                }
                let middle = chars.len() / 2;
                parts = vec![
                    chars[..middle].iter().collect(),
                    chars[middle..].iter().collect(),
                ];
            }
            let audio = parts
                .iter()
                .filter_map(|part| synthesize(part).ok())
                .collect::<Vec<_>>();
            if audio.is_empty() {
                Err(error)
            } else {
                Ok(audio)
            }
        }
    }
}

fn speakable_numbers(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if !ch.is_ascii_digit() {
            result.push(ch);
            continue;
        }
        let mut digits = ch.to_string();
        while chars.peek().is_some_and(char::is_ascii_digit) {
            digits.push(chars.next().unwrap());
        }
        result.push_str(&chinese_number(&digits));
        let mut following = chars.clone();
        if following.next() == Some(':') && following.peek().is_some_and(char::is_ascii_digit) {
            chars.next();
            let mut minute = String::new();
            while chars.peek().is_some_and(char::is_ascii_digit) {
                minute.push(chars.next().unwrap());
            }
            result.push('点');
            result.push_str(&chinese_number(&minute));
            result.push('分');
        }
    }
    result
}

fn chinese_number(digits: &str) -> String {
    const CHINESE: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
    if let Ok(number) = digits.parse::<usize>() {
        if number < 10 {
            return CHINESE[number].to_string();
        }
        if number < 100 {
            let tens = number / 10;
            let units = number % 10;
            let mut result = String::new();
            if tens != 1 {
                result.push(CHINESE[tens]);
            }
            result.push('十');
            if units != 0 {
                result.push(CHINESE[units]);
            }
            return result;
        }
    }
    digits
        .bytes()
        .map(|digit| CHINESE[usize::from(digit - b'0')])
        .collect()
}

pub fn sentences(markdown: &str) -> Vec<String> {
    sentence_chunks(&markdown_text(markdown))
}

fn sentence_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut chunk = String::new();
    let mut count = 0;
    let lines = text.split('\n').collect::<Vec<_>>();
    for (index, line) in lines.iter().enumerate() {
        let mut characters = line.trim().chars().peekable();
        while let Some(character) = characters.next() {
            chunk.push(character);
            count += 1;
            let sentence_end = matches!(character, '。' | '！' | '？' | '!' | '?' | '；' | ';')
                || (character == '.' && characters.peek().is_none_or(|next| next.is_whitespace()))
                || (matches!(character, '，' | ',') && count >= 30);
            if sentence_end || count >= 120 {
                push_sentence(&mut chunks, &mut chunk);
                count = 0;
            }
        }
        if index + 1 < lines.len() {
            // ponytail: Markdown has no page width; use OCR geometry if line boxes become available.
            let short_line =
                line.trim().chars().count() < 40 || lines[index + 1].trim().chars().count() < 40;
            if short_line || lines[index + 1].trim().is_empty() {
                push_sentence(&mut chunks, &mut chunk);
                count = 0;
            } else if !chunk.is_empty() {
                chunk.push(' ');
                count += 1;
            }
        }
    }
    push_sentence(&mut chunks, &mut chunk);
    chunks
}

fn push_sentence(chunks: &mut Vec<String>, chunk: &mut String) {
    if !chunk.trim().is_empty() {
        chunks.push(chunk.trim().to_owned());
    }
    chunk.clear();
}

pub fn markdown_text(markdown: &str) -> String {
    let mut text = String::new();
    for event in pulldown_cmark::Parser::new(markdown) {
        match event {
            pulldown_cmark::Event::Text(piece) | pulldown_cmark::Event::Code(piece) => {
                text.push_str(&piece);
            }
            pulldown_cmark::Event::SoftBreak | pulldown_cmark::Event::HardBreak => text.push('\n'),
            pulldown_cmark::Event::End(
                pulldown_cmark::TagEnd::Paragraph
                | pulldown_cmark::TagEnd::Heading(_)
                | pulldown_cmark::TagEnd::Item,
            ) => text.push_str("\n\n"),
            _ => {}
        }
    }
    text.trim().to_owned()
}

#[cfg(test)]
mod tests {
    #[test]
    fn failed_slice_does_not_silence_readable_parts() {
        let mut attempted = Vec::new();
        let audio = super::synthesize_resilient_with("你好 @ 世界", |part| {
            attempted.push(part.to_owned());
            if part.contains('@') {
                Err("Melo TTS failed to generate audio".to_owned())
            } else {
                Ok(part.to_owned())
            }
        });
        assert_eq!(audio.unwrap(), ["你好", "世界"]);
        assert_eq!(attempted, ["你好 @ 世界", "你好", "世界"]);
        let unspaced = super::synthesize_resilient_with("你好世界", |part| {
            if part == "你好世界" {
                Err("Melo TTS failed to generate audio".to_owned())
            } else {
                Ok(part.to_owned())
            }
        });
        assert_eq!(unspaced.unwrap(), ["你好", "世界"]);
        assert!(
            super::synthesize_resilient_with("@@@", |_| -> Result<(), String> {
                Err("Melo TTS failed to generate audio".to_owned())
            })
            .is_err()
        );
    }

    #[test]
    #[ignore = "loads the downloaded Melo TTS model"]
    fn real_melo_model_generates_bilingual_audio() {
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("downloads");
        let audio = super::synthesize_sentence(
            "你好, hello world!",
            &model_dir,
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(audio.sample_rate, 44_100);
        assert!(audio.samples.len() > 4_410);
        assert!(audio.samples.iter().any(|sample| *sample != 0));
        let reused = super::synthesize_sentence(
            "你好, hello world!",
            &model_dir,
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
        assert!(std::sync::Arc::ptr_eq(&audio, &reused));
        let clock = super::synthesize_sentence(
            "13:37",
            &model_dir,
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
        assert!(clock.samples.len() > 4_410);
    }

    #[test]
    fn markdown_reader_uses_document_text_not_markup() {
        assert_eq!(
            super::markdown_text("# Hello\n\n**world** [link](https://example.com)"),
            "Hello\n\nworld link"
        );
    }

    #[test]
    fn unspaced_chinese_is_split_without_losing_text() {
        let text = "中文".repeat(100);
        let chunks = super::sentence_chunks(&text);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn reading_splits_at_sentence_boundaries_without_splitting_domains() {
        assert_eq!(
            super::sentence_chunks("你好。 Hello world. Visit example.com now! 再见。"),
            ["你好。", "Hello world.", "Visit example.com now!", "再见。"]
        );
    }

    #[test]
    fn short_markdown_lines_become_sentences_but_long_wrapped_lines_continue() {
        assert_eq!(super::sentences("短行\n下一句。"), ["短行", "下一句。"]);
        let long_line = "中文".repeat(25);
        let next_long_line = "继续".repeat(25);
        assert_eq!(
            super::sentences(&format!("{long_line}\n{next_long_line}。")),
            [format!("{long_line} {next_long_line}。")]
        );
        assert_eq!(
            super::sentences(&format!("{long_line}\n继续。")),
            [long_line.clone(), "继续。".to_owned()]
        );
    }

    #[test]
    fn long_clause_uses_comma_as_a_speech_boundary() {
        let first = "中".repeat(35);
        assert_eq!(
            super::sentence_chunks(&format!("{first}，后半句。")),
            [format!("{first}，"), "后半句。".to_owned()]
        );
    }
}
