# -*- coding: utf-8 -*-

r"""
Fine-tunable SONAR SpeechToEmbeddingModel encoder.

Partial layer freezing: the first ``freeze_ratio`` fraction of transformer
layers is frozen (early layers encode low-level acoustic/phonetic features
that are not speech-quality-relevant), while the remaining top layers and
the encoder pooler are left trainable.

Recommended values based on layer-wise probing literature (24 ConformerBlocks total):
    freeze_layers=4    – freeze bottom 4 layers (~17%)
    freeze_layers=12   – freeze bottom half
    freeze_layers=18   – freeze bottom three-quarters
"""
from typing import Dict, List

import torch

from speechcomet.encoders.sonar import SONAREncoder


class SONARFTEncoder(SONAREncoder):
    """Partially fine-tunable SONAR speech encoder.

    Args:
        pretrained_model (str): SONAR model name (e.g. 'sonar_speech_encoder_eng').
        text_out_dim (int): Dimensionality of the output embedding.
        device (str): Device string passed to SpeechToEmbeddingModelPipeline.
        freeze_layers (int): Number of encoder layers to freeze from the bottom.
            E.g. 4 freezes layers 0-3, leaving layers 4-23 and the pooler trainable.
            Defaults to 4.
    """

    def __init__(
        self,
        pretrained_model: str,
        text_out_dim: int,
        device: str,
        freeze_layers: int = 4,
    ) -> None:
        super().__init__(pretrained_model, text_out_dim, device)
        self.freeze_layers = freeze_layers

    # SONARFTEncoder supports gradient-based fine-tuning; no LoRA needed.
    lora_enabled = False

    # ------------------------------------------------------------------
    # Layer helpers
    # ------------------------------------------------------------------

    def _encoder_layers(self):
        return self.model.model.encoder.layers

    def _n_freeze(self) -> int:
        return self.freeze_layers

    # ------------------------------------------------------------------
    # Freezing / unfreezing
    # ------------------------------------------------------------------

    def freeze_embeddings(self) -> None:
        """Freeze the bottom ``freeze_ratio`` of encoder layers.

        Everything else (upper layers, encoder pooler, optional projection)
        is kept trainable.
        """
        # Freeze the entire speech model first.
        for param in self.model.model.parameters():
            param.requires_grad = False

        layers = self._encoder_layers()
        n_freeze = self._n_freeze()

        # Unfreeze upper transformer layers.
        for layer in layers[n_freeze:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze the encoder pooler.
        for param in self.model.model.encoder_pooler.parameters():
            param.requires_grad = True

        # Unfreeze the projection head if present.
        if self.need_project:
            for param in self.projection.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    # Layerwise learning-rate schedule
    # ------------------------------------------------------------------

    def layerwise_lr(self, lr: float, decay: float) -> List[dict]:
        """Per-layer learning rates with exponential decay toward the bottom.

        Only the trainable (unfrozen) layers are included.  The top layer
        gets ``lr``, each lower layer gets ``lr * decay^i``.

        Args:
            lr (float): Learning rate for the topmost encoder layer.
            decay (float): Multiplicative decay per layer step downward.

        Returns:
            List[dict]: Parameter groups suitable for an optimizer.
        """
        layers = self._encoder_layers()
        n_freeze = self._n_freeze()
        trainable_layers = layers[n_freeze:]  # bottom-to-top order in list

        opt_parameters: List[dict] = []

        # Iterate from top to bottom among the unfrozen layers.
        for i, layer in enumerate(reversed(trainable_layers)):
            opt_parameters.append(
                {
                    "params": list(layer.parameters()),
                    "lr": lr * decay ** i,
                }
            )

        # Encoder pooler at the learning rate of the topmost layer.
        opt_parameters.append(
            {
                "params": list(self.model.model.encoder_pooler.parameters()),
                "lr": lr,
            }
        )

        # Note: the projection head (if any) is added by the model's
        # configure_optimizers separately, so we do not include it here.

        return opt_parameters

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs, **kwargs) -> Dict[str, torch.Tensor]:
        """Run the SONAR encoder forward with gradient flow through unfrozen layers.

        ``predict()`` is decorated with ``@torch.inference_mode()`` which cannot
        be overridden by ``torch.enable_grad()``.  We therefore replicate the
        pipeline manually: fbank extraction runs in no_grad (it touches no
        learned parameters), then we call the encoder model directly so that
        autograd tracks activations through the unfrozen layers.
        """
        from fairseq2.data.data_pipeline import Collater
        from sonar.inference_pipelines.utils import extract_sequence_batch

        pipeline = self.model  # SpeechToEmbeddingModelPipeline

        # Preprocessing: audio → fbank (no learned parameters, no grad needed)
        with torch.no_grad():
            fbank_dicts = [
                pipeline.convert_to_fbank(pipeline._decode_audio(w))
                for w in inputs
            ]
            collated = Collater(pad_value=0, pad_to_multiple=2)(fbank_dicts)
            seqs, batch_layout = extract_sequence_batch(collated["fbank"], pipeline.device)

        # Encoder forward — gradients flow through unfrozen ConformerBlocks + pooler
        output = pipeline.model(seqs, batch_layout)
        sonar_emb = output.sentence_embeddings
        if self.need_project:
            sonar_emb = self.projection(sonar_emb)
        return sonar_emb
