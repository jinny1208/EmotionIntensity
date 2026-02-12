import shutil
from pathlib import Path
import subprocess
import whisper


# ==========================
# CONFIG
# ==========================
ROOT_DIR = Path("/mnt/hdd/jeonyj0612/MEAD/")
OUTPUT_DIR = Path("/mnt/hdd/jeonyj0612/MEAD_processed/")
TARGET_SR = 22050

IGNORED_EMOTIONS = {"contempt", "disgusted"}
WHISPER_MODEL = "medium"  # tiny, base, small, medium, large, turbo

ERROR_LOG = OUTPUT_DIR / "ffmpeg_errors.txt"


# ==========================
# UTILS
# ==========================
def is_dir_empty(path: Path) -> bool:
    return not any(path.iterdir())


def convert_m4a_to_wav(src: Path, dst: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-vn",                 # IMPORTANT
        "-ac", "1",
        "-ar", str(TARGET_SR),
        str(dst),
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


# ==========================
# MAIN PIPELINE
# ==========================
def process_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.touch(exist_ok=True)

    print("Loading Whisper model...")
    model = whisper.load_model(WHISPER_MODEL)

    for speaker_dir in ROOT_DIR.iterdir():
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name
        audio_root = speaker_dir / "audio"

        if not audio_root.exists():
            continue

        # --------------------------------------------------
        # 1) Emotion-level cleanup
        # --------------------------------------------------
        for emotion_dir in audio_root.iterdir():
            if not emotion_dir.is_dir():
                continue

            emotion = emotion_dir.name

            if is_dir_empty(emotion_dir):
                print(f"Deleting empty emotion folder: {emotion_dir}")
                shutil.rmtree(emotion_dir)
                continue

            if emotion in IGNORED_EMOTIONS:
                continue

            # --------------------------------------------------
            # 2) Audio processing
            # --------------------------------------------------
            for level_dir in emotion_dir.iterdir():
                if not level_dir.is_dir():
                    continue

                level_name = level_dir.name.replace("_", "")

                for audio_file in level_dir.glob("*.m4a"):
                    file_id = audio_file.stem

                    out_wav_name = (
                        f"{speaker_id}_{emotion}_{level_name}_{file_id}.wav"
                    )
                    out_wav_path = OUTPUT_DIR / out_wav_name

                    print(f"Processing: {audio_file}")

                    # 2-1) Convert with ffmpeg
                    success = convert_m4a_to_wav(audio_file, out_wav_path)

                    if not success:
                        with ERROR_LOG.open("a", encoding="utf-8") as f:
                            f.write(str(audio_file) + "\n")
                        print(f"[FFMPEG ERROR] Logged: {audio_file}")
                        continue

                    # --------------------------------------------------
                    # 3) Transcription
                    # --------------------------------------------------
                    result = model.transcribe(str(out_wav_path))
                    transcript = result["text"].strip()

                    out_txt_path = out_wav_path.with_suffix(".txt")
                    out_txt_path.write_text(transcript, encoding="utf-8")


if __name__ == "__main__":
    process_dataset()
