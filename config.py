from pathlib import Path

LAB_DIR = Path(__file__).parent
OUTPUT_DIR = LAB_DIR / "output"

SAMPLE_AUDIO = LAB_DIR / "sample_audio.wav"

#  STT
WHISPER_MODEL    = "small"   # tiny / base / small / medium
WHISPER_LANGUAGE = "uk"
GOOGLE_LANGUAGE  = "uk-UA"

#  Audio
SAMPLE_RATE      = 16000     # optimal for Whisper & SpeechRecognition
RECORD_DURATION  = 15        # seconds for microphone recording
STT_CHUNK_SEC    = 25        # Google STT chunk length (API limit 30s)

#  Noise reduction (noisereduce)
NR_STATIONARY    = True
NR_PROP_DECREASE = 0.80

#  Text analysis
TOP_N_WORDS  = 30
MIN_WORD_LEN = 3

OUTPUT_FILES = {
    "spectrogram_raw":       OUTPUT_DIR / "01_spectrogram_raw.png",
    "spectrogram_denoised":  OUTPUT_DIR / "02_spectrogram_denoised.png",
    "denoised_wav":          OUTPUT_DIR / "03_denoised.wav",
    "transcription_noisy":   OUTPUT_DIR / "04_transcription_noisy.txt",
    "transcription":         OUTPUT_DIR / "05_transcription_denoised.txt",
    "stt_comparison":        OUTPUT_DIR / "06_stt_comparison.txt",
    "freq_csv":              OUTPUT_DIR / "07_freq_analysis.csv",
    "freq_chart":            OUTPUT_DIR / "08_freq_chart.png",
    "wordcloud":             OUTPUT_DIR / "09_wordcloud.png",
    "annotation":            OUTPUT_DIR / "10_annotation.txt",
    "verification_mp3":      OUTPUT_DIR / "11_verification.mp3",
}
