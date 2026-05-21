"""SpeechRegressionJoint — Option B speech QE model.

The audio embedding (from SONAR) is injected as a virtual token into the
XLM-R sequence so that XLM-R's self-attention operates jointly over both
MT tokens and the audio representation:

    [CLS] mt_token_1 ... mt_token_n [SEP] | audio_emb_as_virtual_token

This is the closest speech analogue of COMETKiwi's unified_metric, where
src and MT are encoded together in one transformer forward pass.

The estimator input is output_units (not ×4), matching COMETKiwi.
Only input_modality: audio is supported (audio is the source).
"""
from typing import Dict, List, Tuple, Union

import torch
from speechcomet.modules import FeedForward
from speechcomet.models.regression.speech import SpeechRegression
from speechcomet.models.utils import Prediction, Target


class SpeechRegressionJoint(SpeechRegression):
    """SpeechRegression variant with joint audio-text encoding.

    SONAR audio embedding is appended as a virtual token to the MT token
    embedding sequence before the XLM-R transformer layers run, enabling
    full cross-attention between audio and every MT token.

    Only supports input_modality: audio.
    Estimator in_dim = encoder.output_units (not × 4).
    """

    def __init__(self, *args, class_identifier=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.class_identifier = "speech_joint_metric"
        if self.input_modality != "audio":
            raise ValueError(
                "SpeechRegressionJoint only supports input_modality: audio"
            )
        # Replace estimator: in_dim = output_units (not × 4)
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
            out_dim=1,
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # 1. Audio embedding from SONAR (B, output_units)
        waveforms = kwargs["src_waveforms"]
        audio_emb = self.get_audio_embedding(waveforms)  # (B, H)

        # 2. MT token embeddings from XLM-R's embedding layer (B, seq_len, H)
        mt_input_ids   = kwargs["mt_input_ids"]
        mt_attn_mask   = kwargs["mt_attention_mask"]
        mt_token_embeds = self.encoder.model.embeddings(mt_input_ids)  # (B, seq_len, H)

        # 3. Append audio as virtual token: [mt_embeds | audio_emb]
        B = mt_input_ids.shape[0]
        audio_token = audio_emb.unsqueeze(1)  # (B, 1, H)
        joint_embeds = torch.cat([mt_token_embeds, audio_token], dim=1)  # (B, seq_len+1, H)

        # 4. Extend attention mask to include the virtual token
        audio_mask  = torch.ones(B, 1, device=mt_attn_mask.device, dtype=mt_attn_mask.dtype)
        joint_mask  = torch.cat([mt_attn_mask, audio_mask], dim=1)  # (B, seq_len+1)

        # 5. Run through XLM-R transformer with inputs_embeds
        outputs = self.encoder.model(
            inputs_embeds=joint_embeds,
            attention_mask=joint_mask,
            output_hidden_states=True,
        )
        all_layers = outputs.hidden_states  # tuple of (B, seq_len+1, H)

        # 6. Apply layerwise attention (if layer: mix)
        if self.layerwise_attention:
            embeddings = self.layerwise_attention(list(all_layers), joint_mask)
        else:
            embeddings = all_layers[-1]

        # 7. Pool (over the joint sequence including the virtual audio token)
        if self.hparams.pool == "avg":
            # Masked average: ignore padding tokens (mask=0), include virtual audio token
            mask_expanded = joint_mask.unsqueeze(-1).float()
            sentemb = (embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            # cls (default): CLS token at position 0
            sentemb = embeddings[:, 0, :]

        return Prediction(score=self.estimator(sentemb).view(-1))
