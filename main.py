import sys
from pathlib import Path

from config import OUTPUT_DIR, OUTPUT_FILES, SAMPLE_AUDIO, RECORD_DURATION, SAMPLE_RATE
from audio_processor import (
    load_audio, denoise, save_spectrogram, save_wav,
    transcribe, record_from_mic,
)
from text_analyzer import (
    tokenize, word_frequency, text_stats,
    save_freq_chart, save_wordcloud, annotate,
)
from tts import text_to_speech


def _resolve_audio() -> Path:
    args = sys.argv[1:]

    if "--mic" in args:
        import soundfile as sf
        audio = record_from_mic(RECORD_DURATION)
        if len(audio) > 0:
            mic_path = OUTPUT_DIR / "mic_recording.wav"
            sf.write(str(mic_path), audio, SAMPLE_RATE)
            print(f"[OK] Mic recording saved: {mic_path}")
            return mic_path
        print("[WARN] Mic recording failed — falling back to sample audio")

    if args and not args[0].startswith("--"):
        path = Path(args[0])
        if path.exists():
            return path
        print(f"[WARN] File not found: {path}")

    if not SAMPLE_AUDIO.exists():
        print("[INFO] Generating sample audio...")
        from generate_sample import generate_sample
        generate_sample(SAMPLE_AUDIO)
    return SAMPLE_AUDIO


def _save_comparison(noisy_text: str, denoised_text: str) -> None:
    noisy_words    = len(noisy_text.split())
    denoised_words = len(denoised_text.split())
    diff_words     = denoised_words - noisy_words
    diff_pct       = (diff_words / max(noisy_words, 1)) * 100

    lines = [
        "=== STT Comparison: noisy vs denoised ===",
        f"noisy_words:    {noisy_words}",
        f"denoised_words: {denoised_words}",
        f"noisy_chars:    {len(noisy_text)}",
        f"denoised_chars: {len(denoised_text)}",
        f"diff_words:     {diff_words:+d}",
        f"diff_pct:       {diff_pct:+.1f}%",
    ]

    comparison = "\n".join(lines)
    OUTPUT_FILES["stt_comparison"].write_text(comparison, encoding="utf-8")
    print(f"[OK] Saved {OUTPUT_FILES['stt_comparison']}")
    print(f"[INFO] STT diff: {diff_words:+d} words ({diff_pct:+.1f}%)")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_path = _resolve_audio()
    print(f"\n[INFO] Audio source: {audio_path}")

    raw_audio, sr = load_audio(audio_path)
    save_spectrogram(raw_audio, sr, OUTPUT_FILES["spectrogram_raw"], "Вхідний аудіосигнал")

    denoised = denoise(raw_audio, sr)
    save_spectrogram(denoised, sr, OUTPUT_FILES["spectrogram_denoised"], "Аудіосигнал після фільтрації")
    save_wav(denoised, sr, OUTPUT_FILES["denoised_wav"])

    print("\n[INFO] Transcribing noisy audio (for comparison)...")
    temp_noisy = OUTPUT_DIR / "_temp_noisy.wav"
    save_wav(raw_audio, sr, temp_noisy)
    noisy_text = transcribe(temp_noisy)
    temp_noisy.unlink(missing_ok=True)   # remove temp file after use
    OUTPUT_FILES["transcription_noisy"].write_text(noisy_text, encoding="utf-8")
    print(f"[OK] Saved {OUTPUT_FILES['transcription_noisy']}")

    print("\n[INFO] Transcribing denoised audio...")
    denoised_text = transcribe(OUTPUT_FILES["denoised_wav"])

    if not denoised_text.strip():
        print("[WARN] Transcription is empty")
        denoised_text = "[Транскрипція недоступна]"

    OUTPUT_FILES["transcription"].write_text(denoised_text, encoding="utf-8")
    print(f"[OK] Saved {OUTPUT_FILES['transcription']}")

    _save_comparison(noisy_text, denoised_text)

    tokens = tokenize(denoised_text)
    stats  = text_stats(denoised_text, tokens)

    print("\n[INFO] Text statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    freq_df = word_frequency(tokens)
    freq_df.to_csv(str(OUTPUT_FILES["freq_csv"]), sep=";", index=False, encoding="utf-8")
    print(f"[OK] Saved {OUTPUT_FILES['freq_csv']}")

    save_freq_chart(freq_df, OUTPUT_FILES["freq_chart"])
    save_wordcloud(tokens, OUTPUT_FILES["wordcloud"])

    annotation = annotate(denoised_text, tokens)
    OUTPUT_FILES["annotation"].write_text(annotation, encoding="utf-8")
    print(f"[OK] Saved {OUTPUT_FILES['annotation']}")

    text_to_speech(denoised_text, OUTPUT_FILES["verification_mp3"], lang="uk")

    noisy_w    = len(noisy_text.split())
    denoised_w = len(denoised_text.split())
    print("\n=== Summary ===")
    print(f"  Source:              {audio_path}")
    print(f"  Duration:            {len(raw_audio)/sr:.1f}s")
    print(f"  Noisy transcription: {noisy_w} words")
    print(f"  Clean transcription: {denoised_w} words (+{denoised_w-noisy_w})")
    print(f"  Unique tokens:       {stats['unique_words']}")
    print(f"  Type-token ratio:    {stats['type_token_ratio']}")
    print("\nOutput files:")
    for path in OUTPUT_FILES.values():
        if path.exists():
            print(f"  [OK] {path}")
        else:
            print(f"  [MISSING] {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
