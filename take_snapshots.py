import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def extract_snapshots(
    video_path: str,
    num_snapshots: int,
    output_dir: str, 
    crop_rect: tuple = None,
    start_time: float = 0.0,
    end_time: float = None):
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError("The video contains no frames.")

    if end_time is None:
        end_time = total_frames / fps

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    if start_frame >= end_frame or start_frame >= total_frames:
        raise ValueError("Invalid start_time and end_time range.")

    frame_indices = np.linspace(start_frame, min(end_frame, total_frames) - 1, num=num_snapshots, dtype=int)
    saved_images = []

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            print(f"Warning: Could not read frame {frame_idx}")
            continue

        if crop_rect:
            x, y, w, h = crop_rect
            frame = frame[y:y+h, x:x+w]

        snapshot_path = snapshots_dir / f"snapshot_{i+1:03}.jpg"
        cv2.imwrite(str(snapshot_path), frame)
        saved_images.append(snapshot_path)

    cap.release()

    return saved_images

def create_summary_plot(
    image_paths,
    output_dir,
    n_rows=0
    ):
    num_images = len(image_paths)
    if n_rows > 0:
        rows = n_rows
        cols = int(np.ceil(num_images / n_rows))
    else:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if num_images > 1 else [axes]

    for ax, img_path in zip(axes, image_paths):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        # ax.set_title(img_path.name, fontsize=8)
        ax.axis('off')

    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    summary_path = Path(output_dir) / "summary_plot.jpg"
    plt.savefig(summary_path, dpi=150)
    plt.close()

    return summary_path

def process_video_to_snapshots(
    video_path: str,
    num_snapshots: int,
    output_dir: str, 
    crop_rect: tuple = None,
    start_time: float = 0.0,
    end_time: float = None,
    n_rows: int = 0
    ):
    snapshots = extract_snapshots(video_path, num_snapshots, output_dir, crop_rect, start_time, end_time)
    if snapshots:
        summary = create_summary_plot(snapshots, output_dir, n_rows)
        print(f"Saved {len(snapshots)} snapshots and summary plot to {output_dir}")
    else:
        print("No snapshots were saved.")
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract snapshots from a video and create a summary plot.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--num_snapshots", type=int, default=9, help="Number of snapshots to extract.")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save the snapshots and summary plot.")
    parser.add_argument("--crop_rect", type=int, nargs=4, metavar=("x", "y", "w", "h"), help="Crop rectangle as x, y, width, height. Example: --crop_rect 100 50 200 200")
    parser.add_argument("--start_time", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--end_time", type=float, help="End time in seconds.")
    parser.add_argument("--n_rows", type=int, default=0, help="End time in seconds.")
    
    args = parser.parse_args()
    
    output_dir = os.path.splitext(args.video_path)[0] if not args.output_dir else args.output_dir

    process_video_to_snapshots(
        args.video_path,
        args.num_snapshots,
        output_dir,
        crop_rect=args.crop_rect,
        start_time=args.start_time,
        end_time=args.end_time,
        n_rows=args.n_rows)
