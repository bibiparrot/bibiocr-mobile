use crate::core::{MODELS, ModelSpec};
use crate::settings::DownloadSettings;
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::Duration,
};

#[derive(Debug)]
pub enum DownloadEvent {
    Started(usize),
    Progress(usize, u64, u64),
    Complete(usize),
    Failed(usize, String),
    Finished,
}

pub struct DownloadTask {
    pub events: Receiver<DownloadEvent>,
    paused: Arc<AtomicBool>,
}

impl DownloadTask {
    pub fn toggle_pause(&self) {
        self.paused.fetch_xor(true, Ordering::Relaxed);
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }
}

pub fn start(model_dir: PathBuf, settings: DownloadSettings, locale: &'static str) -> DownloadTask {
    let (sender, events) = mpsc::channel();
    let paused = Arc::new(AtomicBool::new(false));
    let worker_pause = Arc::clone(&paused);
    thread::spawn(move || {
        for (index, model) in MODELS.iter().enumerate() {
            if installed(model_dir.as_path(), model) {
                let _ = sender.send(DownloadEvent::Complete(index));
                continue;
            }
            let _ = sender.send(DownloadEvent::Started(index));
            if let Err(error) = download(
                model,
                &model_dir,
                index,
                &sender,
                &worker_pause,
                &settings,
                locale,
            ) {
                let _ = sender.send(DownloadEvent::Failed(index, error));
                return;
            }
            let _ = sender.send(DownloadEvent::Complete(index));
        }
        let _ = sender.send(DownloadEvent::Finished);
    });
    DownloadTask { events, paused }
}

pub fn installed(model_dir: &Path, model: &ModelSpec) -> bool {
    model_dir
        .join(model.file_name)
        .metadata()
        .is_ok_and(|metadata| metadata.len() == model.expected_bytes)
}

fn download(
    model: &ModelSpec,
    model_dir: &Path,
    index: usize,
    sender: &Sender<DownloadEvent>,
    paused: &AtomicBool,
    settings: &DownloadSettings,
    locale: &str,
) -> Result<(), String> {
    fs::create_dir_all(model_dir).map_err(|error| error.to_string())?;
    let destination = model_dir.join(model.file_name);
    let partial = destination.with_extension(format!(
        "{}.part",
        destination
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
    ));
    let offset = resume_offset(&partial, settings.resume);
    let _ = sender.send(DownloadEvent::Progress(index, offset, model.expected_bytes));
    let env_endpoint = std::env::var("HF_ENDPOINT").ok();
    let url = settings.resolve_url(model.url, locale, env_endpoint.as_deref());
    let mut request = ureq::get(&url);
    if offset > 0 {
        request = request.header("Range", format!("bytes={offset}-"));
    }
    let mut response = request.call().map_err(|error| error.to_string())?;
    let resumed = response.status().as_u16() == 206 && offset > 0;
    let mut downloaded = if resumed { offset } else { 0 };
    let content_length = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let total = content_length.map_or(model.expected_bytes, |length| downloaded + length);
    let mut output = OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(&partial)
        .map_err(|error| error.to_string())?;
    let mut reader = response.body_mut().as_reader();
    let mut buffer = vec![0_u8; 256 * 1024];
    loop {
        while paused.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(100));
        }
        let count = reader
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|error| error.to_string())?;
        downloaded += count as u64;
        let _ = sender.send(DownloadEvent::Progress(index, downloaded, total));
    }
    output.flush().map_err(|error| error.to_string())?;
    if downloaded != model.expected_bytes {
        return Err(format!(
            "{} has {downloaded} bytes; expected {}",
            model.file_name, model.expected_bytes
        ));
    }
    fs::rename(partial, destination).map_err(|error| error.to_string())
}

fn resume_offset(partial: &Path, enabled: bool) -> u64 {
    if enabled {
        partial.metadata().map_or(0, |metadata| metadata.len())
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{installed, resume_offset};
    use crate::core::MODELS;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn installed_requires_the_published_size() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-download-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let model = MODELS[2];
        let file = fs::File::create(root.join(model.file_name)).unwrap();
        file.set_len(model.expected_bytes - 1).unwrap();
        assert!(!installed(&root, &model));
        file.set_len(model.expected_bytes).unwrap();
        assert!(installed(&root, &model));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_setting_controls_partial_file_offset() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-resume-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let partial = root.join("model.gguf.part");
        let file = fs::File::create(&partial).unwrap();
        file.set_len(128).unwrap();

        assert_eq!(resume_offset(&partial, true), 128);
        assert_eq!(resume_offset(&partial, false), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "downloads the real Melo model and lexicon once"]
    fn real_bilingual_tts_assets_download_and_reuse_cache() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("downloads");
        let task = super::start(root.clone(), Default::default(), "zh-CN");
        for event in task.events {
            match event {
                super::DownloadEvent::Failed(_, error) => panic!("{error}"),
                super::DownloadEvent::Finished => break,
                _ => {}
            }
        }
        assert!(MODELS[3..].iter().all(|model| installed(&root, model)));

        let offline_settings = crate::settings::DownloadSettings {
            hf_endpoint: "http://127.0.0.1:9".to_owned(),
            ..Default::default()
        };
        let cached = super::start(root, offline_settings, "zh-CN");
        for event in cached.events {
            match event {
                super::DownloadEvent::Failed(_, error) => panic!("cache missed: {error}"),
                super::DownloadEvent::Finished => break,
                _ => {}
            }
        }
    }
}
