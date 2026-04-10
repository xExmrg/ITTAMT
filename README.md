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
2. auto-downloads public OCR data when available (`Teklia/IAM-lines`, fallback variants),
3. generates synthetic OCR data automatically,
4. trains the model,
5. exports checkpoints and TorchScript artifact.

Artifacts are saved by default in `artifacts/stride_moe/`.

## Colab notes

- Recommended GPU: **H100** (best), **A100** (great), otherwise any CUDA GPU works.
- For free/limited Colab, reduce workload:

```bash
EPOCHS=2 BATCH_SIZE=8 SYNTHETIC_SAMPLES=12000 bash scripts/run_colab.sh
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
