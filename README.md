# ITTAMT — STRIDE-MoE OCR

Research-oriented image-to-text OCR system with a custom **STRIDE-MoE** architecture:
- convolutional local visual stem,
- transformer sequence encoder,
- sparse Mixture-of-Experts feed-forward routing,
- CTC decoding head for fast inference.

The project is optimized for Google Colab training and fast macOS inference.

## Quick start (single-script flow for Colab)

```bash
git clone <your_repo_url>
cd ITTAMT
bash scripts/run_colab.sh
```

That one script does all of this:
1. installs dependencies,
2. auto-downloads a mixed OCR training set when available,
3. generates structured synthetic OCR data automatically,
4. trains the model,
5. saves validation previews with reference and predicted text,
6. exports checkpoints and TorchScript artifact.

Artifacts and checkpoints are saved by default outside the repo in Colab under Google Drive at `/content/drive/MyDrive/ittamt/artifacts/stride_moe/`.
Dataset caches are saved outside the repo in Colab under Google Drive at `/content/drive/MyDrive/ittamt/datasets/`.
If Drive is not mounted, the script falls back to `/content/ittamt_persist/`.

## Training data mix

The training loader now mixes several OCR sources automatically:

- `Teklia/IAM-line`: handwritten English line recognition.
- `MiXaiLL76/IIIT5K_OCR`: cropped scene-text words, including number-heavy splits.
- `MiXaiLL76/TextOCR_OCR`: scene-text crops from TextOCR.
- `rth/sroie-2019-v2` / `jsdnrs/ICDAR2019-SROIE`: receipt OCR, converted into line crops.
- `naver-clova-ix/cord-v2`: structured receipts, converted into line crops.
- `nielsr/funsd`: form/document OCR, converted into line crops.
- `docling-project/DocLayNet-v1.2`: document regions with digital text cells, converted into text-bearing line crops.
- `doc-analysis/XFUND`: multilingual form OCR, imported from the official release JSON/ZIP files and cropped into line-level samples.
- structured synthetic receipts, addresses, invoices, and key-value layouts with line breaks.

The page-level datasets are converted into line-level crops because the current model is still a CTC recognizer optimized for line-style inputs.

The following corpora are intentionally not part of the default recognizer mix:

- `PubLayNet`, `TableBank`, and `DocSynth300K` are primarily layout/table analysis datasets rather than direct OCR transcription corpora.
- `DocBank` is token-level and extremely large to auto-pull into a normal Colab run.
- `Bentham`, `CVL`, and the official IAM/SROIE portals require manual download or registration instead of clean automatic mirroring.

## Validation previews

Each epoch now saves a preview image under the active output directory, typically `/content/ittamt_artifacts/stride_moe/eval_previews/` in Colab, showing:

- the validation image sample,
- the ground-truth text,
- the model prediction.

The training script also prints a few `sample[i] ref=...` and `sample[i] pred=...` lines into the Colab log for quick inspection.

## Colab notes

- Recommended GPU: **H100** (best), **A100** (great), otherwise any CUDA GPU works.
- `scripts/run_colab.sh` mounts Google Drive automatically by default and keeps datasets, checkpoints, tokenizer, and exported artifacts under `/content/drive/MyDrive/ittamt/`, so you do not redownload everything after every code change.
- `scripts/train_colab.py` now requires CUDA by default, so Colab will fail fast instead of silently running on CPU.
- `BATCH_SIZE=0` auto-scales from detected GPU VRAM. On a 96 GB card it will pick a much larger batch than the old fixed default.
- The script defaults to `NUM_WORKERS=8` and `PREFETCH_FACTOR=4` to use more host RAM and keep the GPU busier.
- The dataset mix is materialized into system RAM after loading, so Drive-backed caching mainly affects startup time; once the dataloaders are built, GPU feeding stays RAM-based rather than repeatedly streaming samples from Drive.
- For free/limited Colab, reduce workload:

```bash
EPOCHS=2 BATCH_SIZE=8 SYNTHETIC_SAMPLES=12000 bash scripts/run_colab.sh
```

For large-memory Colab instances, the defaults are already tuned to use more GPU/CPU RAM. If you still want to override them manually:

```bash
BATCH_SIZE=192 NUM_WORKERS=8 PREFETCH_FACTOR=4 PERSIST_ROOT=/content/drive/MyDrive/ittamt bash scripts/run_colab.sh
```

If you want to disable Drive mounting and keep everything ephemeral in the current runtime:

```bash
MOUNT_GOOGLE_DRIVE=0 PERSIST_ROOT=/content/ittamt_persist bash scripts/run_colab.sh
```

You can also reduce preview output volume:

```bash
python scripts/train_colab.py --preview-count 2
```

## Inference (Linux/macOS/Windows)

```bash
python scripts/infer.py --image path/to/text_image.png
```

macOS screenshot flow (future shortcut integration):

```bash
python scripts/infer.py --screenshot --image capture.png
```

## Files

- `scripts/train_colab.py`: end-to-end training entrypoint.
- `scripts/infer.py`: fast inference from file or screenshot.
- `src/ittamt/model.py`: STRIDE-MoE model.
- `src/ittamt/data.py`: dataset auto-loading + synthetic generation.
- `src/ittamt/tokenizer.py`: char-level tokenizer + CTC decode.

## Next upgrades

- Distillation into a smaller student model for sub-second CPU latency.
- ONNX/CoreML export for Apple-native acceleration.
- Real structured decoding head (`<line_break>`, `<paragraph>`) beyond plain CTC.
