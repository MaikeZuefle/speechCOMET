# -*- coding: utf-8 -*-

r"""
Whisper Speech Encoder
"""
from typing import Dict, List

import torch
from speechcomet.encoders.base import Encoder


class WhisperEncoder(Encoder):
    """Whisper encoder that pools the encoder's sequence output into a fixed-size embedding.

    Args:
        pretrained_model (str): HuggingFace model name, e.g. 'openai/whisper-small'.
        text_out_dim (int): Target embedding dimension (to match the text encoder).
        device (str): Device to place the model on.
        pool (str): Pooling strategy over the encoder sequence: 'avg', 'max', or 'attn'.
    """

    def __init__(
        self,
        pretrained_model: str,
        text_out_dim: int,
        device: str,
        pool: str = "avg",
    ) -> None:
        super().__init__()
        from transformers import WhisperModel, WhisperProcessor

        full_model = WhisperModel.from_pretrained(pretrained_model)
        self.model = full_model.encoder.to(device)
        self.processor = WhisperProcessor.from_pretrained(pretrained_model)
        self.sr = 16000
        self.out_dim = text_out_dim
        self.pool_strategy = pool

        whisper_dim = self.model.config.d_model
        self.need_project = self.out_dim != whisper_dim
        if self.need_project:
            self.projection = torch.nn.Linear(whisper_dim, text_out_dim).to(device)

        if self.pool_strategy == "attn":
            self.attn_pool = torch.nn.Linear(whisper_dim, 1).to(device)

        self.lora_enabled = False

    def apply_lora(self, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        """Replace the Whisper encoder with a LoRA-wrapped version.
        Freezes all base weights; only the small LoRA adapter matrices are trainable.

        Args:
            r: LoRA rank.
            alpha: LoRA scaling factor.
            dropout: Dropout on the LoRA layers.
        """
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.lora_enabled = True

    def prepare_sample(self, audios: list) -> Dict[str, list]:
        """Load and resample audio. Returns the same dict format as SONAREncoder
        so the rest of the pipeline is unchanged."""
        import torchaudio

        waveforms = []
        for item in audios:
            if type(item).__name__ == "AudioDecoder":  # HF encoded audio
                samples = item.get_all_samples()
                waveform = samples.data  # (channels, samples)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                sr = samples.sample_rate
            elif isinstance(item, dict) and "array" in item:
                import numpy as np
                waveform = torch.from_numpy(np.array(item["array"], dtype="float32")).unsqueeze(0)
                sr = int(item["sampling_rate"])
            else:
                waveform, sr = torchaudio.load(str(item))

            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
                waveform = resampler(waveform)
            waveforms.append(waveform)
        return {"waveforms": waveforms}

    def forward(self, inputs: list, **kwargs) -> torch.Tensor:
        """Run waveforms through Whisper encoder and pool to a fixed-size embedding.

        Args:
            inputs: list of waveform tensors, each shape (1, T) or (T,).

        Returns:
            torch.Tensor: shape (batch_size, out_dim).
        """
        device = next(self.model.parameters()).device

        # WhisperProcessor expects 1-D numpy arrays at 16 kHz
        waveforms_np = []
        for w in inputs:
            w = w.squeeze(0)  # (T,)
            waveforms_np.append(w.cpu().float().numpy())

        features = self.processor(
            waveforms_np,
            sampling_rate=self.sr,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).input_features.to(device)  # (batch, 80, 3000)

        encoder_out = self.model(features)
        hidden = encoder_out.last_hidden_state  # (batch, seq_len, d_model)

        if self.pool_strategy == "avg":
            emb = hidden.mean(dim=1)
        elif self.pool_strategy == "max":
            emb = hidden.max(dim=1).values
        elif self.pool_strategy == "attn":
            # (batch, seq_len, 1) -> softmax over frames -> weighted sum
            scores = torch.softmax(self.attn_pool(hidden), dim=1)  # (batch, seq_len, 1)
            emb = (scores * hidden).sum(dim=1)                     # (batch, d_model)
        else:
            raise ValueError(f"Unknown pool strategy for WhisperEncoder: '{self.pool_strategy}'. Choose 'avg', 'max', or 'attn'.")

        if self.need_project:
            emb = self.projection(emb)
        return emb

    def freeze_embeddings(self) -> None:
        """Freeze the Whisper encoder weights; keep projection and attn_pool trainable."""
        for param in self.model.parameters():
            param.requires_grad = False
        if self.need_project:
            for param in self.projection.parameters():
                param.requires_grad = True
        if self.pool_strategy == "attn":
            for param in self.attn_pool.parameters():
                param.requires_grad = True

    @property
    def output_units(self) -> int:
        return self.out_dim

    @property
    def max_positions(self) -> int:
        return self.model.config.max_source_positions

    @property
    def num_layers(self) -> int:
        return self.model.config.encoder_layers

    @property
    def size_separator(self) -> int:
        return 0

    @property
    def uses_token_type_ids(self) -> bool:
        return False

    @classmethod
    def from_pretrained(cls, pretrained_model, load_pretrained_weights=True, local_files_only=False):
        raise NotImplementedError

    def layerwise_lr(self, lr: float, decay: float) -> List[dict]:
        raise NotImplementedError
