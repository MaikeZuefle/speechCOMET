import torch
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

m = SpeechToEmbeddingModelPipeline('sonar_speech_encoder_eng', device=torch.device('cuda'))

for secs in [10, 30, 60, 70, 80, 85, 90, 100, 120]:
    n = secs * 16000
    w = torch.zeros(1, n).cuda()  # shape (1, T) — channels x samples
    try:
        result = m.predict([w])
        print(f'{secs}s ({n} samples) -> OK')
    except Exception as e:
        print(f'{secs}s ({n} samples) -> ERROR: {e}')
