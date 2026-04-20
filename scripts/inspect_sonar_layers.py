from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
import torch

FREEZE_LAYERS = 4  # change this to see different freeze boundaries

model_name = "sonar_speech_encoder_eng"
pipeline = SpeechToEmbeddingModelPipeline(model_name, device=torch.device("cpu"))
encoder = pipeline.model.encoder

n = len(encoder.layers)
print(f"Total layers: {n}  |  freeze_layers={FREEZE_LAYERS}\n")
for i, layer in enumerate(encoder.layers):
    status = "FROZEN " if i < FREEZE_LAYERS else "trained"
    print(f"  Layer {i:2d}: {type(layer).__name__}  [{status}]")

print(f"\n  Pooler: {type(pipeline.model.encoder_pooler).__name__}  [trained]")
print(f"\nTrainable: layers {FREEZE_LAYERS}-{n-1} + pooler  ({n - FREEZE_LAYERS} / {n} layers)")
