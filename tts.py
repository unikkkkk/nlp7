from pathlib import Path


def text_to_speech(text: str, path: Path, lang: str = "uk") -> None:
    from gtts import gTTS

    # gTTS accepts max 5000 chars per call; trim if needed
    snippet = text[:4500]
    tts = gTTS(text=snippet, lang=lang, slow=False)
    tts.save(str(path))
    print(f"[OK] TTS verification saved to {path} ({len(snippet)} chars)")
