import math
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

from config import SAMPLE_RATE, NR_STATIONARY, NR_PROP_DECREASE, GOOGLE_LANGUAGE, STT_CHUNK_SEC


#  Load / Save

def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / sr
    print(f"[OK] Loaded: {path.name} — {duration:.1f}s, sr={sr}Hz")
    return audio, sr


def save_wav(audio: np.ndarray, sr: int, path: Path) -> None:
    sf.write(str(path), audio, sr)
    print(f"[OK] Saved {path}")


def add_noise(audio: np.ndarray, snr_db: float = 22.0) -> np.ndarray:
    signal_power = np.mean(audio ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


#  Noise reduction

def denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    reduced = nr.reduce_noise(
        y=audio, sr=sr,
        stationary=NR_STATIONARY,
        prop_decrease=NR_PROP_DECREASE,
    )
    print(f"[OK] Noise reduction: stationary={NR_STATIONARY}, prop_decrease={NR_PROP_DECREASE}")
    return reduced.astype(np.float32)


#  Visualization

def save_spectrogram(audio: np.ndarray, sr: int, path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    librosa.display.waveshow(audio, sr=sr, ax=axes[0])
    axes[0].set_title(f"{title} — Waveform (амплітуда / час)")
    axes[0].set_xlabel("Час (с)")
    axes[0].set_ylabel("Амплітуда")

    X   = librosa.stft(audio)
    Xdb = librosa.amplitude_to_db(np.abs(X))
    img = librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz", ax=axes[1])
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    axes[1].set_title(f"{title} — Spectrogram (частота / час)")

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")


#  STT Google (primary) + Whisper (fallback)

def _google_stt(audio_path: Path) -> str:
    import speech_recognition as sr

    recognizer  = sr.Recognizer()
    audio_data, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)

    chunk_size   = STT_CHUNK_SEC * sample_rate
    total_chunks = math.ceil(len(audio_data) / chunk_size)
    results: list[str] = []

    for idx, i in enumerate(range(0, len(audio_data), int(chunk_size))):
        chunk = audio_data[i : i + int(chunk_size)]
        if len(chunk) < 0.5 * sample_rate:
            continue
        raw   = (chunk * 32767).astype(np.int16).tobytes()
        audio = sr.AudioData(raw, sample_rate, 2)
        try:
            part = recognizer.recognize_google(audio, language=GOOGLE_LANGUAGE)
            results.append(part)
            print(f"[INFO] Google STT chunk {idx+1}/{total_chunks}: {len(part.split())} words")
        except sr.UnknownValueError:
            print(f"[WARN] Google STT chunk {idx+1}: not recognized")
        except sr.RequestError as e:
            print(f"[WARN] Google STT request error: {e}")
            raise

    return " ".join(results)


def _whisper_stt(audio_path: Path) -> str:
    import whisper
    from config import WHISPER_MODEL, WHISPER_LANGUAGE

    model    = whisper.load_model(WHISPER_MODEL)
    audio_np, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    result   = model.transcribe(audio_np, language=WHISPER_LANGUAGE)
    return result["text"].strip()


def transcribe(audio_path: Path) -> str:
    try:
        import speech_recognition  # noqa: F401
        print(f"[INFO] Google STT (speech_recognition, lang={GOOGLE_LANGUAGE})...")
        text = _google_stt(audio_path)
        if text.strip():
            print(f"[OK] Google STT: {len(text)} chars, {len(text.split())} words")
            return text
        print("[WARN] Google STT returned empty result")
    except ImportError:
        print("[WARN] speech_recognition not installed")
    except Exception as e:
        print(f"[WARN] Google STT failed: {e}")

    try:
        import whisper  # noqa: F401
        from config import WHISPER_MODEL, WHISPER_LANGUAGE
        print(f"[INFO] Whisper fallback (model={WHISPER_MODEL}, lang={WHISPER_LANGUAGE})...")
        text = _whisper_stt(audio_path)
        print(f"[OK] Whisper: {len(text)} chars, {len(text.split())} words")
        return text
    except ImportError:
        print("[WARN] openai-whisper not installed")
    except Exception as e:
        print(f"[WARN] Whisper failed: {e}")

    return ""


#  Microphone

def record_from_mic(duration: int = 60) -> np.ndarray:
    try:
        import sounddevice as sd
        print(f"[INFO] Recording {duration}s from microphone...")
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        print("[OK] Recording complete")
        return audio.flatten()
    except ImportError:
        print("[WARN] sounddevice not available — cannot record from microphone")
        return np.array([], dtype=np.float32)
