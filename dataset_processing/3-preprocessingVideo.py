import cv2
import random
import numpy as np
from pathlib import Path


# ==========================
# CONFIG
# ==========================
ROOT_DIR = Path("/mnt/hdd/jeonyj0612/MEAD/")
OUTPUT_DIR = Path("/mnt/hdd/jeonyj0612/MEAD_processed/")
NUM_FRAMES = 6
IMG_SIZE = 224

IGNORED_EMOTIONS = {"contempt", "disgusted"}

PROBLEM_LOG = OUTPUT_DIR / "problematic_videos.txt"


# ==========================
# UTILS
# ==========================
def extract_random_frames(video_path: Path, num_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return {}   # not considered problematic

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 🔴 ONLY problematic condition
    if total_frames == 0:
        cap.release()
        return None

    frame_indices = random.sample(
        range(total_frames),
        min(num_frames, total_frames),
    )

    frames = {}
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frames[idx] = frame

    cap.release()
    return frames


# ==========================
# MAIN PIPELINE
# ==========================
def process_videos():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for speaker_dir in ROOT_DIR.iterdir():
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name
        video_root = speaker_dir / "video" / "front"

        if not video_root.exists():
            continue

        for emotion_dir in video_root.iterdir():
            if not emotion_dir.is_dir():
                continue

            emotion = emotion_dir.name
            if emotion in IGNORED_EMOTIONS:
                continue

            for level_dir in emotion_dir.iterdir():
                if not level_dir.is_dir():
                    continue

                level_name = level_dir.name.replace("_", "")  # level_1 → level1

                for video_file in level_dir.glob("*.mp4"):
                    file_id = video_file.stem

                    print(f"Processing: {video_file}")

                    frames = extract_random_frames(video_file, NUM_FRAMES)

                    # 🔴 total_frames == 0 → log & skip
                    if frames is None:
                        with open(PROBLEM_LOG, "a") as f:
                            f.write(str(video_file) + "\n")
                        print(f"Skipped (0 frames): {video_file}")
                        continue

                    # Normal case
                    for i, frame in enumerate(frames.values(), start=1):
                        out_name = (
                            f"{speaker_id}_{emotion}_{level_name}_{file_id}_fg{i}.png"
                        )
                        out_path = OUTPUT_DIR / out_name
                        cv2.imwrite(str(out_path), frame)


if __name__ == "__main__":
    process_videos()
