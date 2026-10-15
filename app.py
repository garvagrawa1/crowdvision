"""Run CrowdVision on a webcam or video file."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from crowdvision.counting import count_person_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect people and show a count in a webcam or video."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (e.g. 0) or path to a video file (default: 0).",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics YOLO model/weights (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum confidence from 0 to 1 (default: 0.35).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save annotated video, e.g. output/result.mp4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1.")

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.model)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(
            f"Could not open source {args.source!r}. Check the webcam index or file path."
        )

    writer = None
    output_path = Path(args.output) if args.output else None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(
                frame,
                conf=args.confidence,
                classes=[0],  # COCO class 0 is person for standard COCO weights.
                verbose=False,
            )[0]
            annotated = result.plot()
            boxes = result.boxes
            raw_count = int(boxes.cls.numel()) if boxes is not None and boxes.cls is not None else 0
            person_count = count_person_detections(raw_count)

            cv2.putText(
                annotated,
                f"People detected: {person_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if output_path and writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 20.0
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or annotated.shape[1]
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or annotated.shape[0]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise SystemExit(f"Could not create output video at {output_path}")

            if writer is not None:
                writer.write(annotated)

            cv2.imshow("CrowdVision - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
