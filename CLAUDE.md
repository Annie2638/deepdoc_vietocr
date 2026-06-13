# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DeepDoc + VietOCR is a fast, CPU-friendly OCR tool specialized for Vietnamese documents. It is a fork of RAGFlow's DeepDoc component, extracted into a standalone repo, with PaddleOCR's text recognizer replaced by **VietOCR** for better Vietnamese accuracy. It provides three capabilities: plain OCR, Layout Recognition, and Table Structure Recognition (TSR).

## Running

There is no build step or test suite. Everything runs as Python CLI scripts against image/PDF inputs. All three entrypoints accept `--inputs` (a directory of images/PDFs, or a single file) and `--output_dir`.

```bash
# OCR: produces an annotated image + a .txt of recognized text per page
python t_ocr.py --inputs=<path> --output_dir=<out>

# Layout recognition: annotated image with region labels (Text/Title/Table/...)
python t_recognizer.py --inputs=<path> --threshold=0.2 --mode=layout --output_dir=<out>

# Table structure recognition: annotated image + a .md of the reconstructed table
python t_recognizer.py --inputs=<path> --threshold=0.2 --mode=tsr --output_dir=<out>

# Full pipeline: layout-aware document -> single concatenated Markdown (.md) per page,
# OCRing tables via TSR and reading remaining regions in reading order
python full_pipeline.py --inputs=<path> --threshold=0.5 --output_dir=<out>
```

`pip install -r requirements.txt` for dependencies. Note `vietocr` and `trio` are imported but not listed in requirements.txt — install them separately if missing.

### Important runtime behavior
- **stdout/stderr are redirected to log files.** Each entrypoint reassigns `sys.stdout`/`sys.stderr` to `log/<script>.log` (e.g. `log/t_ocr.log`, `log/full_pipeline.log`) and appends a `=== Run N ===` header per invocation. Console prints will NOT appear in the terminal — check the log file for output and tracebacks.
- These scripts force CPU: `t_ocr.py` sets `CUDA_VISIBLE_DEVICES=''`. GPU paths exist but the default deployment target is CPU-only.

## Architecture

The pipeline is a chain of ONNX models plus VietOCR, orchestrated by thin CLI wrappers. The core lives in `module/`, with shared infra in `utils/`.

### Detection → Recognition split (`module/ocr.py`)
The `OCR` class composes two stages, mirroring PaddleOCR:
- **`TextDetector`** runs the ONNX `det` model to find text-line bounding boxes (DBNet-style, with `DBPostProcess`). Preprocessing/postprocessing ops are config-driven via `create_operators` / `build_post_process`.
- **`TextRecognizer`** is the customized part: it wraps **VietOCR's `Predictor`** (not an ONNX model by default) to read each cropped box. Defaults to the `vgg_seq2seq` model with weights at `vietocr/weight/vgg_seq2seq.pth`.

`OCR.__call__(img)` returns `[(box, (text, score)), ...]`, dropping results below `drop_score` (0.5).

### Swapping the recognizer
This is the most common intended customization, and it's done by **editing source, not flags**:
- **seq2seq ↔ transformer**: in `module/ocr.py` `TextRecognizer.__init__`, switch the commented `Cfg.load_config_from_name('vgg_seq2seq')` / weights block to the `vgg_transformer` block. Transformer is slower with marginal accuracy gain.
- **PyTorch VietOCR ↔ ONNX VietOCR**: replace `from module.ocr import OCR` with `from module.ocr_onnx import OCR` in the entrypoint (commented examples are present in `t_ocr.py` / `t_recognizer.py`). Faster, slightly lower accuracy. Note `module/ocr_onnx.py` references RAGFlow-era import paths (`api.utils...`, `rag.settings`, `tool.config`) and is not wired to this repo's layout — expect to fix imports before it runs.

### Layout & Table Structure (`module/layout_recognizer.py`, `module/table_structure_recognizer.py`, `module/recognizer.py`)
Both extend the base `Recognizer` (ONNX YOLOv10 inference + NMS in `module/operators.py`). `module/__init__.py` exports `LayoutRecognizer4YOLOv10` aliased as `LayoutRecognizer`.
- `LayoutRecognizer` detects 10 region types (`Text`, `Title`, `Figure`, `Table`, etc. — see `labels` list).
- `TableStructureRecognizer` detects column/row/header/spanning-cell components, then `construct_table(...)` + the geometric matching helpers (`find_overlapped_with_threashold`, `find_horizontally_tightest_fit`, `sort_Y_firstly`, `layouts_cleanup`) reassemble OCR'd boxes into a Markdown table. This matching logic is duplicated between `t_recognizer.py:get_table_markdown` and `full_pipeline.py:extract_table_markdown` — keep them in sync when changing one.

### full_pipeline.py reading-order logic
Runs layout detection, OCRs each `Table` region into Markdown, masks out detected regions, then OCRs the remaining unmasked area as body text. Regions are reassembled by their top-`y` coordinate to preserve reading order, then concatenated into one `<name>_full.md`.

### Models (`onnx/`) and weights (`vietocr/weight/`)
ONNX models (`det`, `rec`, `layout*`, `tsr`, etc.) live in `onnx/` and are committed. If the local `onnx/` dir is missing, `OCR`/`LayoutRecognizer` fall back to `huggingface_hub.snapshot_download(repo_id="InfiniFlow/deepdoc")`. On HuggingFace download issues, set `HF_ENDPOINT=https://hf-mirror.com`. VietOCR `.pth`/`.onnx` weights live in `vietocr/weight/`.

### utils/ and RAGFlow heritage
`utils/settings.py` and `utils/db/` carry over RAGFlow's storage/DB/Redis configuration (MinIO, Elasticsearch, etc.) driven by `conf/service_conf.yaml` and env vars (`DOC_ENGINE`, `STORAGE_IMPL`). **This machinery is largely vestigial for the OCR CLIs** — the scripts here only need `get_project_base_directory`, `traversal_files`, and `PARALLEL_DEVICES`. Importing `utils.settings` may attempt to read storage config; avoid expanding that dependency surface for OCR work.

`PARALLEL_DEVICES` (= CUDA device count) controls multi-GPU fan-out in `OCR.__init__` and the `trio`-based nursery launcher in `t_ocr.py`. On CPU it collapses to a single device/sequential path.

## Conventions
- CLI help strings and some prints are in Vietnamese; code/identifiers are English.
- Paths use Windows-style separators in a few places (e.g. `r"vietocr\weight\..."`); this repo's primary target is Windows.
