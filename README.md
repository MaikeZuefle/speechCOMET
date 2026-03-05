# speechCOMET

Then, this package can be used in Python with comet_early_exit package. The package name changed intentionally from Unbabel's package name such that they are not mutually exclusive.

## Development

Install the package locally and 
```bash
pip3 install -e .
speechcomet-train --cfg configs/models/speech_audio.yaml
speechcomet-train --cfg configs/models/speech_audiotext.yaml
speechcomet-train --cfg configs/models/speech_text.yaml


speechcomet-score ...

python3 scripts/01-get_data_text.py
```

or in Python:
```python
import speechcomet
from speechcomet import download_model

model = speechcomet.load_from_checkpoint("...") # for local model
model = speechcomet.load_from_checkpoint(download_model("...")) # for HF model

sample = {"src": "I love cake.", "mt": "Ich liebe Kekse."} # if src modality text
sample = {"src_audio": "cake.wav", "mt": "Ich liebe Kekse."} # if src modality speech
score = model.predict(samples=[sample], gpus=1, num_workers=1, batch_size=1).scores

```


## Misc

If you use this work, please cite:
```bibtex
@misc{speechcomet26,
  author={Maike Züfle, Vilém Zouhar},
  url={https://github.com/zouharvi/speechCOMET},
  title={SpeechCOMET: audio-source, text-target translation quality estimation},
  year={2025}
}
```