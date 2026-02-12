import tarfile
from pathlib import Path


def process_audio_video_tars(root_dir: str, dry_run: bool):
    root = Path(root_dir)

    for tar_path in root.rglob("*.tar"):
        name = tar_path.name

        if name.startswith("audio"):
            target_dir = tar_path.parent / "audio"
            is_video = False
        elif name.startswith("video"):
            target_dir = tar_path.parent / "video"
            is_video = True
        else:
            continue

        print(f"\n[TAR] {tar_path}")
        print(f" -> target dir: {target_dir}")

        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tar_path) as tar:
            members = tar.getmembers()

            # Detect single top-level directory
            top_level_dirs = {
                m.name.split("/")[0]
                for m in members
                if m.name and "/" in m.name
            }

            strip_prefix = None
            if len(top_level_dirs) == 1:
                strip_prefix = next(iter(top_level_dirs)) + "/"
                print(f" -> stripping prefix: {strip_prefix}")
            else:
                print(" -> no prefix stripping")

            for member in members:
                if not member.name:
                    continue

                final_name = member.name
                if strip_prefix and final_name.startswith(strip_prefix):
                    final_name = final_name[len(strip_prefix):]

                if not final_name:
                    continue

                # 🔥 VIDEO RULE: keep only front/
                if is_video:
                    if not (
                        final_name == "front"
                        or final_name.startswith("front/")
                    ):
                        continue

                final_path = target_dir / final_name

                if dry_run:
                    print(f"    WOULD EXTRACT: {final_path}")
                else:
                    member.name = final_name
                    tar.extract(member, path=target_dir)


def main():
    process_audio_video_tars(
        root_dir="/mnt/hdd/jeonyj0612/MEAD",
        dry_run=False,
    )


if __name__ == "__main__":
    main()


###################### Delete everything except tar. 
# import argparse
# from pathlib import Path
# import shutil


# def clean_non_tar_dirs(root_dir: str, dry_run: bool):
#     root = Path(root_dir)

#     # Walk bottom-up so we can safely delete children first
#     for dir_path in sorted(
#         [p for p in root.rglob("*") if p.is_dir()],
#         key=lambda p: len(p.parts),
#         reverse=True,
#     ):
#         # Skip root itself
#         if dir_path == root:
#             continue

#         has_tar = any(f.suffix == ".tar" for f in dir_path.iterdir())

#         if not has_tar:
#             if dry_run:
#                 print(f"WOULD DELETE DIR: {dir_path}")
#             else:
#                 print(f"DELETING DIR: {dir_path}")
#                 shutil.rmtree(dir_path)


# def main():
#     # parser = argparse.ArgumentParser(
#     #     description="Delete all directories that do NOT contain .tar files"
#     # )
    
#     # args = parser.parse_args()
#     clean_non_tar_dirs(root_dir="/mnt/hdd/jeonyj0612/MEAD", dry_run = False)


# if __name__ == "__main__":
#     main()
