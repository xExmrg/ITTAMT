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

Artifacts are saved by default in `artifacts/stride_moe/`.

## Training data mix

The training loader now mixes several OCR sources automatically:

- `Teklia/IAM-line`: handwritten English line recognition.
- `MiXaiLL76/IIIT5K_OCR`: cropped scene-text words.
- `jsdnrs/ICDAR2019-SROIE`: receipt OCR, converted into line crops.
- `naver-clova-ix/cord-v2`: structured receipts, converted into line crops.
- `nielsr/funsd`: form/document OCR, converted into line crops.
- structured synthetic receipts, addresses, invoices, and key-value layouts with line breaks.

The page-level datasets are converted into line-level crops because the current model is still a CTC recognizer optimized for line-style inputs.

## Validation previews

Each epoch now saves a preview image in `artifacts/stride_moe/eval_previews/` showing:

- the validation image sample,
- the ground-truth text,
- the model prediction.

The training script also prints a few `sample[i] ref=...` and `sample[i] pred=...` lines into the Colab log for quick inspection.

## Colab notes

- Recommended GPU: **H100** (best), **A100** (great), otherwise any CUDA GPU works.
- For free/limited Colab, reduce workload:

```bash
EPOCHS=2 BATCH_SIZE=8 SYNTHETIC_SAMPLES=12000 bash scripts/run_colab.sh
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
