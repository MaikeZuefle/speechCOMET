# -*- coding: utf-8 -*-

r"""
SONAR SpeechToEmbeddingModel
"""
from typing import Dict

import torch
from speechcomet.encoders.base import Encoder


class SONAREncoder(Encoder):
    """SONAR encoder.

    Args:
        pretrained_model (str): Pretrained model from hugging face.
    """

    def __init__(
        self,
        pretrained_model: str,
        text_out_dim: int, 
        device: str,
    ) -> None:
        super().__init__()
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
        self.model = SpeechToEmbeddingModelPipeline(pretrained_model, device=torch.device(device))
        self.sr=16000
        self.out_dim = text_out_dim

        sonar_dim = self.model.model.encoder_pooler.projection_out.output_dim
        self.need_project = self.out_dim != sonar_dim


        if self.need_project:
            self.projection = torch.nn.Linear(sonar_dim, text_out_dim).to(device)

    lora_enabled = False  # SONAR does not support LoRA

    def prepare_sample(self, audios):
        from datasets.features._torchcodec import AudioDecoder
        import torchaudio
        waveforms = []

        for item in audios:
            if type(item).__name__ == "AudioDecoder": # HF encoded audio
                # Use torchaudio via _hf_encoded to avoid torchcodec segfaults
                import io
                encoded = getattr(item, "_hf_encoded", None)
                if encoded is not None and encoded.get("bytes"):
                    waveform, sr = torchaudio.load(io.BytesIO(encoded["bytes"]))
                elif encoded is not None and encoded.get("path"):
                    waveform, sr = torchaudio.load(encoded["path"])
                else:
                    samples = item.get_all_samples()  # fallback
                    waveform = samples.data
                    sr = samples.sample_rate
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
            else:
                waveform, sr = torchaudio.load(str(item))

            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
                waveform = resampler(waveform)
            waveforms.append(waveform)
        return {"waveforms": waveforms}

    def forward(
        self, inputs: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        sonar_emb = self.model.predict(inputs).clone() #.detach().requires_grad_(False) # torch.Size([batch_size, 1024])
        if self.need_project:
            sonar_emb = self.projection(sonar_emb)
        return sonar_emb
    
    def freeze_embeddings(self) -> None:
        """Frezees the embedding layer."""
        for param  in self.model.model.parameters():
            param.requires_grad = False
        if self.need_project:
            for param in self.projection.parameters():
                param.requires_grad = True

    @property
    def output_units(self) -> int:
        """Output dim"""
        return self.out_dim

    @property
    def max_positions(self) -> int:
        """Max number of tokens the encoder handles."""
        encoder = self.model.model.encoder
        max_seq_len = encoder.layers[0].self_attn.sdpa.pos_encoding.max_seq_len
        return max_seq_len

    @property
    def num_layers(self) -> int:
        """Number of model layers available."""
        encoder = self.model.model.encoder
        return  len(encoder.layers)

    @property
    def size_separator(self) -> int:
        """Size of the seperator.
        E.g: For BERT is just 1 ([SEP]) but models such as XLM-R use 2 (</s></s>).

        Returns:
            int: Number of tokens used between two segments.
        """
        return 0

    @property
    def uses_token_type_ids(self) -> bool:
        """Whether or not the model uses token type ids to differentiate sentences.

        Returns:
            bool: True for models that use token_type_ids such as BERT.
        """
        return False


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> Encoder:
        raise NotImplementedError
    
    def layerwise_lr(self, lr: float, decay: float):
        """Calculates the learning rate for each layer by applying a small decay.

        Args:
            lr (float): Learning rate for the highest encoder layer.
            decay (float): decay percentage for the lower layers.

        Returns:
            list: List of model parameters for all layers and the corresponding lr.
        """
        raise NotImplementedError
        # Last layer keeps LR
        opt_parameters = [
            {
                "params": self.model.encoder.layer[-1].parameters(),
                "lr": lr,
            }
        ]
        # Decay at each layer.
        for i in range(2, self.num_layers):
            opt_parameters.append(
                {
                    "params": self.model.encoder.layer[-i].parameters(),
                    "lr": lr * decay ** (i - 1),
                }
            )
        # Embedding Layer
        opt_parameters.append(
            {
                "params": self.model.embeddings.parameters(),
                "lr": lr * decay ** (self.num_layers),
            }
        )
        return opt_parameters