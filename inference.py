"""
inference.py  —  Generate emotional speech from face images using finetuned MEAD model.

Reads a test filelist, generates audio for each line using the face image
as conditioning, and saves as:
    {output_dir}/{lineNum}_{emotion}_{intensity}.wav

Usage:
    python inference.py \
        --config   configs/mead.json \
        --checkpoint /path/to/G_XXXXX.pth \
        --filelist /path/to/test_filelist.txt \
        --output_dir /path/to/output \
        --device cuda:0
"""

import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.io.wavfile import write
from tqdm import tqdm

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence_infer, cleaned_text_to_sequence

from utils import load_wav_to_torch
from mel_processing import spectrogram_torch


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True,  help="Path to config json")
    p.add_argument("--checkpoint",  required=True,  help="Path to G_XXXXX.pth")
    p.add_argument("--filelist",    required=True,  help="Path to test filelist")
    p.add_argument("--output_dir",  required=True,  help="Directory to save .wav files")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--noise_scale",   type=float, default=0.667)
    p.add_argument("--noise_scale_w", type=float, default=0.8)
    p.add_argument("--length_scale",  type=float, default=1.0)
    p.add_argument("--max_len",       type=int,   default=2000,
                   help="Max output frames (increase for long sentences)")
    p.add_argument("--cleaned_text",  action="store_true", default=True,
                   help="Input text is already phonemized (default True for MEAD filelist)")
    return p.parse_args()


# ── text processing ───────────────────────────────────────────────────────────

def get_text(text: str, hps, cleaned: bool) -> torch.Tensor:
    if cleaned:
        text_norm = cleaned_text_to_sequence(text)
    else:
        text_norm = text_to_sequence_infer(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)


# ── filelist parsing ──────────────────────────────────────────────────────────

def parse_filelist(filelist_path: str):
    """
    Parses lines of the format:
      audio_path|image_path|text|phonemes|emotion|intensity

    Returns list of dicts with keys:
      audio_path, image_path, text, phonemes, emotion, intensity
    """
    samples = []
    with open(filelist_path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            # if len(parts) < 6:
            #     print(f"[warn] line {lineno}: expected 6 fields, got {len(parts)} — skipping")
            #     continue
            samples.append({
                "lineno"    : lineno,
                "audio_path": parts[0],
                "phonemes"  : parts[1],
            })
    return samples


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── config & model ────────────────────────────────────────────────────────
    hps = utils.get_hparams_from_file(args.config)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    net_g.eval()

    utils.load_checkpoint(args.checkpoint, net_g, None)
    print(f"[inference] loaded checkpoint: {args.checkpoint}")

    # ── filelist ──────────────────────────────────────────────────────────────
    samples = parse_filelist(args.filelist)
    print(f"[inference] {len(samples)} samples from {args.filelist}")
    print(f"[inference] output dir: {args.output_dir}")

    # ── generation loop ───────────────────────────────────────────────────────
    skipped = 0
    for sample in tqdm(samples, desc="Generating"):
        lineno    = sample["lineno"]

        # output filename: lineNum_emotion_intensity.wav
        out_name = f"{lineno:05d}.wav"
        out_path = os.path.join(args.output_dir, out_name)

        # skip already generated files (useful when resuming after crash)
        if os.path.exists(out_path):
            continue

        try:
            # text
            text_input = sample["phonemes"] if args.cleaned_text else sample["text"]
            stn = get_text(text_input, hps, cleaned=args.cleaned_text)

            ref_audio, _ = utils.load_wav_to_torch(sample["audio_path"])

            ref_audio_norm = ref_audio / 32768.0
            ref_audio_norm = ref_audio_norm.unsqueeze(0)

            spec = spectrogram_torch(
                ref_audio_norm,
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False
            )

            spec = torch.squeeze(spec, 0).to(device)

            with torch.no_grad():
                x_tst = stn.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn.size(0)]).to(device)

                audio = net_g.infer(
                    x_tst,
                    x_tst_lengths,
                    spec.unsqueeze(0),  # ✅ THIS is the key input
                    noise_scale=args.noise_scale,
                    noise_scale_w=args.noise_scale_w,
                    length_scale=args.length_scale,
                )[0][0, 0].data.cpu().float().numpy()

            write(out_path, hps.data.sampling_rate, audio)

        except Exception as e:
            print(f"[warn] line {lineno} failed: {e}")
            skipped += 1
            continue

    print(f"[inference] done. generated {len(samples) - skipped} files, skipped {skipped}")
    print(f"[inference] saved to {args.output_dir}")


if __name__ == "__main__":
    main()