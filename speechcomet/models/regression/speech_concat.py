"""SpeechRegressionConcat — like SpeechRegression but uses the unified-metric
style of encoding: src and MT are concatenated into one sequence [CLS] mt [SEP] src [SEP]
and the CLS token is used as the sentence embedding (in_dim = output_units).

This makes the text encoder compatible with COMETKiwi (Unbabel/wmt22-cometkiwi-da)
checkpoints, which use the same unified encoding strategy.

For audio and audiotext modalities the audio encoder is unchanged; the audio
embedding is fused with the CLS text embedding via fuse_emb_strategy.
"""
from typing import Dict, List, Tuple, Union

import torch
from speechcomet.modules import FeedForward

from speechcomet.models.regression.speech import SpeechRegression
from speechcomet.models.utils import Prediction, Target


class SpeechRegressionConcat(SpeechRegression):
    """SpeechRegression variant using unified (CLS-based) encoding.

    Changes vs SpeechRegression:
    - estimator in_dim = encoder.output_units  (not × 4)
    - text/audiotext: encodes [CLS] mt [SEP] src [SEP] jointly; uses CLS token
    - audio: unchanged (SONAR emb fused with mt text emb)
    """

    def __init__(self, *args, class_identifier=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.class_identifier = "speech_concat_metric"
        # Replace the estimator with correct in_dim
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
            out_dim=1,
        )
        # fusion_layernorm is only built by parent for audiotext; create it for audio too
        if self.input_modality == "audio" and not hasattr(self, "fusion_layernorm"):
            self.fusion_layernorm = torch.nn.LayerNorm(self.encoder.output_units)

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "train"
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        inputs = {k: [dic[k] for dic in sample] for k in sample[0] if k != "score"}

        if self.input_modality == "text":
            # Encode [CLS] mt [SEP] src [SEP] as one sequence
            mt_enc  = self.encoder.prepare_sample(inputs["mt"])
            src_enc = self.encoder.prepare_sample(inputs["src"])
            joint, _, _ = self.encoder.concat_sequences([mt_enc, src_enc])
            model_inputs = {"joint_input_ids": joint["input_ids"],
                            "joint_attention_mask": joint["attention_mask"]}

        elif self.input_modality == "audio":
            # Audio source + text MT encoded separately, fused after
            model_inputs = {
                f"src_{k}": v
                for k, v in self.encoder_model_audio.prepare_sample(inputs["src_audio"]).items()
            }
            mt_enc = self.encoder.prepare_sample(inputs["mt"])
            model_inputs.update({"mt_" + k: v for k, v in mt_enc.items()})

        elif self.input_modality == "audiotext":
            # Audio source + [CLS] mt [SEP] src [SEP] jointly encoded
            model_inputs = {
                f"src_{k}": v
                for k, v in self.encoder_model_audio.prepare_sample(inputs["src_audio"]).items()
            }
            mt_enc  = self.encoder.prepare_sample(inputs["mt"])
            src_enc = self.encoder.prepare_sample(inputs["src"])
            joint, _, _ = self.encoder.concat_sequences([mt_enc, src_enc])
            model_inputs.update({"joint_input_ids": joint["input_ids"],
                                  "joint_attention_mask": joint["attention_mask"]})
        else:
            raise NotImplementedError(f"Unsupported modality: {self.input_modality}")

        if stage == "predict":
            return model_inputs

        scores = [float(s["score"]) for s in sample]
        targets = Target(score=torch.tensor(scores, dtype=torch.float))
        if "system" in inputs:
            targets["system"] = inputs["system"]
        return model_inputs, targets

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        if self.input_modality == "text":
            joint_ids  = kwargs["joint_input_ids"]
            joint_mask = kwargs["joint_attention_mask"]
            sentemb = self.get_sentence_embedding(joint_ids, joint_mask)

        elif self.input_modality == "audio":
            waveforms = kwargs["src_waveforms"]
            audio_emb = self.get_audio_embedding(waveforms)
            mt_emb = self.get_sentence_embedding(kwargs["mt_input_ids"],
                                                  kwargs["mt_attention_mask"])
            if self.fuse_emb_strategy == "sum":
                sentemb = self.fusion_layernorm(audio_emb + mt_emb)
            elif self.fuse_emb_strategy == "avg":
                sentemb = (audio_emb + mt_emb) / 2
            else:
                raise NotImplementedError("audio-only mode supports sum/avg fuse_emb_strategy")

        elif self.input_modality == "audiotext":
            waveforms = kwargs["src_waveforms"]
            audio_emb = self.get_audio_embedding(waveforms)
            joint_ids  = kwargs["joint_input_ids"]
            joint_mask = kwargs["joint_attention_mask"]
            text_cls = self.get_sentence_embedding(joint_ids, joint_mask)

            if self.fuse_emb_strategy == "sum":
                sentemb = self.fusion_layernorm(text_cls + audio_emb)
            elif self.fuse_emb_strategy == "avg":
                sentemb = (text_cls + audio_emb) / 2
            elif self.fuse_emb_strategy == "concat":
                sentemb = self.fusion_proj(torch.cat([text_cls, audio_emb], dim=-1))
        else:
            raise NotImplementedError(f"Unsupported modality: {self.input_modality}")

        return Prediction(score=self.estimator(sentemb).view(-1))
