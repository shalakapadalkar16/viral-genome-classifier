# Model Checkpoints

Model checkpoints are not tracked in git due to GitHub's 100MB file size limit.

## Reproduce Results

Train the model to generate checkpoints:
```bash
python scripts/train_working.py --config configs/working_config.yaml
```

## Best Model Performance

- **Architecture**: Simple CNN (66K parameters)
- **Epoch**: 17
- **Validation F1**: 0.2731
- **Validation Accuracy**: 32.35%
- **Training Time**: ~6 minutes (CPU)

## Download Pre-trained Model

Pre-trained checkpoints available on request or via:
- [Google Drive](#) (coming soon)
- [Hugging Face](#) (coming soon)
