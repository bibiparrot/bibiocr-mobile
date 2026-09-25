use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelSpec {
    pub name: &'static str,
    pub file_name: &'static str,
    pub url: &'static str,
    pub expected_bytes: u64,
}

pub const MODELS: [ModelSpec; 6] = [
    ModelSpec {
        name: "VLM",
        file_name: "PaddleOCR-VL-1.6-GGUF.gguf",
        url: "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/main/PaddleOCR-VL-1.6-GGUF.gguf",
        expected_bytes: 935_769_056,
    },
    ModelSpec {
        name: "MMProj",
        file_name: "PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
        url: "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/main/PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
        expected_bytes: 881_770_560,
    },
    ModelSpec {
        name: "Layout",
        file_name: "inference.onnx",
        url: "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/main/inference.onnx",
        expected_bytes: 130_502_049,
    },
    ModelSpec {
        name: "Melo TTS (Chinese + English)",
        file_name: "melo-model.onnx",
        url: "https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/resolve/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6/model.onnx",
        expected_bytes: 170_429_550,
    },
    ModelSpec {
        name: "Melo TTS lexicon",
        file_name: "melo-lexicon.txt",
        url: "https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/resolve/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6/lexicon.txt",
        expected_bytes: 6_837_671,
    },
    ModelSpec {
        name: "Melo TTS tokens",
        file_name: "melo-tokens.txt",
        url: "https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/resolve/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6/tokens.txt",
        expected_bytes: 655,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Download,
    History,
    Capture,
    Recognizing,
    Result,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SystemInsetsPx {
    pub top: f32,
    pub bottom: f32,
}

impl SystemInsetsPx {
    #[must_use]
    pub fn from_content_bounds(window_height: i32, content_top: i32, content_bottom: i32) -> Self {
        if window_height <= 0 || content_bottom <= content_top {
            return Self::default();
        }
        let top = content_top.clamp(0, window_height);
        let bottom = window_height - content_bottom.clamp(0, window_height);
        Self {
            top: top as f32,
            bottom: bottom as f32,
        }
    }

    #[must_use]
    pub fn runtime_or_fallback(runtime: Option<Self>, fallback: Self) -> Self {
        runtime
            .filter(|insets| insets.top > 0.0 || insets.bottom > 0.0)
            .unwrap_or(fallback)
    }

    #[must_use]
    pub fn with_minimum(self, screen_height: f32) -> Self {
        let minimum = screen_height * 0.08;
        Self {
            top: self.top.max(minimum),
            bottom: self.bottom.max(minimum),
        }
    }
}

pub fn initial_screen(model_dir: &Path) -> Screen {
    if MODELS.iter().all(|model| {
        model_dir
            .join(model.file_name)
            .metadata()
            .is_ok_and(|metadata| metadata.len() == model.expected_bytes)
    }) {
        Screen::History
    } else {
        Screen::Download
    }
}

#[cfg(test)]
mod tests {
    use super::{MODELS, Screen, SystemInsetsPx, initial_screen};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn download_catalog_includes_bilingual_melo_tts_assets() {
        for file_name in ["melo-model.onnx", "melo-lexicon.txt", "melo-tokens.txt"] {
            assert!(MODELS.iter().any(|model| model.file_name == file_name));
        }
        assert!(!MODELS.iter().any(|model| model.name.starts_with("Kokoro")));
    }

    #[test]
    fn first_run_opens_download_until_all_models_exist() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-mobile-models-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();

        assert_eq!(initial_screen(&root), Screen::Download);

        for model in MODELS {
            let file = fs::File::create(root.join(model.file_name)).unwrap();
            file.set_len(model.expected_bytes).unwrap();
        }
        assert_eq!(initial_screen(&root), Screen::History);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn partial_model_does_not_count_as_installed() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-mobile-partial-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        for model in MODELS {
            let file = fs::File::create(root.join(model.file_name)).unwrap();
            file.set_len(model.expected_bytes).unwrap();
        }
        fs::File::create(root.join(MODELS[0].file_name))
            .unwrap()
            .set_len(1)
            .unwrap();

        assert_eq!(initial_screen(&root), Screen::Download);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn safe_area_comes_from_the_runtime_content_rectangle() {
        assert_eq!(
            SystemInsetsPx::from_content_bounds(2_772, 96, 2_704),
            SystemInsetsPx {
                top: 96.0,
                bottom: 68.0,
            }
        );
        assert_eq!(
            SystemInsetsPx::from_content_bounds(0, 0, 0),
            SystemInsetsPx::default()
        );
    }

    #[test]
    fn empty_runtime_insets_do_not_erase_the_last_safe_area() {
        let previous = SystemInsetsPx {
            top: 96.0,
            bottom: 68.0,
        };
        assert_eq!(
            SystemInsetsPx::runtime_or_fallback(Some(SystemInsetsPx::default()), previous),
            previous
        );
        assert_eq!(
            SystemInsetsPx::runtime_or_fallback(
                Some(SystemInsetsPx {
                    top: 108.0,
                    bottom: 72.0,
                }),
                previous,
            ),
            SystemInsetsPx {
                top: 108.0,
                bottom: 72.0,
            }
        );
    }

    #[test]
    fn safe_area_keeps_eight_percent_on_both_edges() {
        let insets = SystemInsetsPx {
            top: 100.0,
            bottom: 200.0,
        }
        .with_minimum(2000.0);
        assert_eq!(insets.top, 160.0);
        assert_eq!(insets.bottom, 200.0);
    }
}
