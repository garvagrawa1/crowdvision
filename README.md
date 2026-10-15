# CrowdVision

**Beginner-friendly, open-source crowd detection and people counting.**

CrowdVision uses a pretrained YOLO model and OpenCV to detect people in a video or webcam feed, draw bounding boxes, and display a live count.

> **Status:** Starter project. It uses a general-purpose pretrained model and has not been validated for safety-critical crowd management. Accuracy depends on camera angle, lighting, occlusion, and crowd density.

## Features
- Detect the `person` class in video files or a webcam stream.
- Display bounding boxes and a people count.
- Adjust confidence threshold and choose a model.
- Optionally save an annotated output video.
- Unit tests for dependency-free counting helpers.

## Quick start

### Requirements
- Python 3.10 or newer
- A webcam or video file (optional for tests)
- Internet access on first run so Ultralytics can download the selected model

### Install

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run with a webcam
```bash
python app.py --source 0
```

### Run with a video
```bash
python app.py --source path/to/video.mp4 --output output/crowdvision.mp4
```

Press **q** in the preview window to quit. Run `python app.py --help` for all options.

The first run downloads the model selected by `--model` (default: `yolo11n.pt`). Review the model provider's current documentation and license terms before commercial use.

## Run tests
```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Project layout
```text
crowdvision/
├── app.py
├── crowdvision/
│   ├── __init__.py
│   └── counting.py
├── tests/
│   └── test_counting.py
├── .github/workflows/tests.yml
├── requirements.txt
└── requirements-dev.txt
```

## Responsible use
- Use only footage you have permission to process.
- Prefer aggregate counts; avoid identifying or tracking individuals.
- Do not use this starter as the sole basis for emergency, policing, access-control, or other high-impact decisions.
- Evaluate on representative footage before relying on counts. Dense crowds and overlapping people can be missed.
- Respect local privacy laws and venue policies.

## Contributing
Issues and pull requests are welcome. Include steps to reproduce bugs and tests for code changes.

## License
This project is distributed under the MIT License; see [LICENSE](LICENSE). The pretrained model and dependencies may have separate terms.
