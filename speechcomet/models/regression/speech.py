# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
SpeechRegression
========================
    Speech Regression Metric that learns to predict a quality assessment by
    looking at audio source and translation.
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from transformers.optimization import Adafactor, get_constant_schedule_with_warmup

from speechcomet.models.regression.regression_metric import RegressionMetric
from speechcomet.models.utils import Prediction, Target
from speechcomet.modules import FeedForward
from speechcomet.encoders.sonar import SONAREncoder
from speechcomet.encoders.whisper import WhisperEncoder

import random
SEED = 42

class SpeechRegression(RegressionMetric):
    """SpeechRegression:

    Args:
        nr_frozen_epochs (Union[float, int]): Number of epochs (% of epoch) that the
            encoder is frozen. Defaults to 0.9.
        keep_embeddings_frozen (bool): Keeps the encoder frozen during training. Defaults
            to True.
        optimizer (str): Optimizer used during training. Defaults to 'AdamW'.
        warmup_steps (int): Warmup steps for LR scheduler.
        encoder_learning_rate (float): Learning rate used to fine-tune the encoder model.
            Defaults to 3.0e-06.
        learning_rate (float): Learning rate used to fine-tune the top layers. Defaults
            to 3.0e-05.
        layerwise_decay (float): Learning rate % decay from top-to-bottom encoder layers.
            Defaults to 0.95.
        encoder_model (str): Encoder model to be used. Defaults to 'XLM-RoBERTa'.
        pretrained_model (str): Pretrained model from Hugging Face. Defaults to
            'microsoft/infoxlm-large'.
        pool (str): Type of sentence level pooling (options: 'max', 'cls', 'avg').
            Defaults to 'avg'
        layer (Union[str, int]): Encoder layer to be used for regression ('mix'
            for pooling info from all layers). Defaults to 'mix'.
        layer_transformation (str): Transformation applied when pooling info from all
            layers (options: 'softmax', 'sparsemax'). Defaults to 'sparsemax'.
        layer_norm (bool): Apply layer normalization. Defaults to 'False'.
        loss (str): Loss function to be used. Defaults to 'mse'.
        dropout (float): Dropout used in the top-layers. Defaults to 0.1.
        batch_size (int): Batch size used during training. Defaults to 4.
        train_data (Optional[List[str]]): List of paths to training data. Each file is
            loaded consecutively for each epoch. Defaults to None.
        validation_data (Optional[List[str]]): List of paths to validation data.
            Validation results are averaged across validation set. Defaults to None.
        hidden_sizes (List[int]): Hidden sizes for the Feed Forward regression.
        activations (str): Feed Forward activation function.
        final_activation (str): Feed Forward final activation.
        local_files_only (bool): Whether or not to only look at local files.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_trg_embeddings_frozen: bool = True,
        keep_src_embeddings_frozen: bool = True,
        keep_embeddings_frozen: Optional[bool] = None,
        optimizer: str = "AdamW",
        warmup_steps: int = 0,
        encoder_learning_rate: float = 1e-06,
        learning_rate: float = 1.5e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        encoder_model_audio: str = "sonar_speech_encoder_eng",
        pretrained_model: str = "xlm-roberta-large",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        layer_transformation: str = "softmax",
        layer_norm: bool = True,
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: List[str] = [],
        validation_data: List[str] = [],
        hidden_sizes: List[int] = [2048, 1024],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        input_modality: str = "audio", # audio, audiotext
        fuse_emb_strategy: str = "avg",
        pool_audio: str = "avg",  # pooling for sequence-based audio encoders (e.g. Whisper)
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
        num_workers: Optional[int] = None,
    ) -> None:
        if keep_embeddings_frozen is not None:
            # Backward compatibility: old checkpoints saved keep_embeddings_frozen
            keep_trg_embeddings_frozen = keep_embeddings_frozen
            keep_src_embeddings_frozen = keep_embeddings_frozen

        super(RegressionMetric, self).__init__(
            nr_frozen_epochs=nr_frozen_epochs,
            keep_embeddings_frozen=keep_trg_embeddings_frozen,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            encoder_learning_rate=encoder_learning_rate,
            learning_rate=learning_rate,
            layerwise_decay=layerwise_decay,
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            pool=pool,
            layer=layer,
            layer_transformation=layer_transformation,
            layer_norm=layer_norm,
            loss=loss,
            dropout=dropout,
            batch_size=batch_size,
            train_data=train_data,
            validation_data=validation_data,
            class_identifier="speech_regression_metric",
            load_pretrained_weights=load_pretrained_weights,
            local_files_only=local_files_only,
        )
        self.save_hyperparameters()
        self.input_modality = input_modality
        self.fuse_emb_strategy = fuse_emb_strategy

        if self.input_modality in ["audio", "audiotext"]:
            # somehow self.device is cpu here, so set manually
            if encoder_model_audio.startswith("sonar_"):
                self.encoder_model_audio = SONAREncoder(
                    encoder_model_audio, text_out_dim=self.encoder.output_units, device="cuda"
                )
            else:
                self.encoder_model_audio = WhisperEncoder(
                    encoder_model_audio, text_out_dim=self.encoder.output_units, device="cuda", pool=pool_audio
                )

            if keep_src_embeddings_frozen:
                self.encoder_model_audio.freeze_embeddings()
            elif isinstance(self.encoder_model_audio, SONAREncoder):
                raise NotImplementedError(
                    "Fine-tuning the SONAR encoder is not supported. "
                    "Set keep_src_embeddings_frozen: True."
                )
            else:
                # Whisper: apply LoRA — only small adapter weights are trained
                self.encoder_model_audio.apply_lora(
                    r=lora_r, alpha=lora_alpha, dropout=lora_dropout
                )

        if self.input_modality == "audiotext":
            fusion_in_dim = self.encoder.output_units * 2
            fusion_out_dim = self.encoder.output_units
            if self.fuse_emb_strategy == "concat":
                self.fusion_proj = torch.nn.Linear(fusion_in_dim, fusion_out_dim)
            elif self.fuse_emb_strategy == "sum":
                self.fusion_layernorm = torch.nn.LayerNorm(fusion_out_dim)

        self.estimator = FeedForward(
            in_dim=self.encoder.output_units *4,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
            out_dim=1,
        )
       

    def requires_references(self) -> bool:
        return False

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Pytorch Lightning method to configure optimizers and schedulers."""
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        top_layers_parameters = [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate}
        ]
        if self.input_modality in ["audio", "audiotext"]:
            if self.encoder_model_audio.need_project:
                top_layers_parameters.append(
                    {"params": self.encoder_model_audio.projection.parameters(), "lr": self.hparams.learning_rate}
                )
            if hasattr(self.encoder_model_audio, "attn_pool"):
                top_layers_parameters.append(
                    {"params": self.encoder_model_audio.attn_pool.parameters(), "lr": self.hparams.learning_rate}
                )
            if self.encoder_model_audio.lora_enabled:
                lora_params = [p for p in self.encoder_model_audio.model.parameters() if p.requires_grad]
                top_layers_parameters.append(
                    {"params": lora_params, "lr": self.hparams.encoder_learning_rate}
                )
        if self.input_modality == "audiotext":
            fusion_params = []
            if self.fuse_emb_strategy == "concat":
                fusion_params = list(self.fusion_proj.parameters())
            elif self.fuse_emb_strategy == "sum":
                fusion_params = list(self.fusion_layernorm.parameters())
            if fusion_params:
                top_layers_parameters.append(
                    {"params": fusion_params, "lr": self.hparams.learning_rate}
                )
        if self.layerwise_attention:
            layerwise_attn_params = [
                {
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]
            params = layer_parameters + top_layers_parameters + layerwise_attn_params
        else:
            params = layer_parameters + top_layers_parameters

        if self.hparams.optimizer == "Adafactor":
            optimizer = Adafactor(
                params,
                lr=self.hparams.learning_rate,
                relative_step=False,
                scale_parameter=False,
            )
        else:
            optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)

        if self.hparams.warmup_steps < 2:
            return [optimizer], []

        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
        )
        return [optimizer], [scheduler]

    def enable_context(self):
        if self.pool == "avg":
            self.use_context = True

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "train"
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """This method will be called by dataloaders to prepared data to input to the
        model.

        Args:
            sample (List[dict]): Batch of train/val/test samples.
            stage (str): model stage (options: 'fit', 'validate', 'test', or
                'predict'). Defaults to 'fit'.

        Returns:
            Model inputs and depending on the 'stage' training labels/targets.
        """
        inputs = {k: [dic[k] for dic in sample] for k in sample[0] if k != "score"}
        
        src_inputs=  {"src_input_ids": None, "src_attention_mask": None}

        if self.input_modality in ["text", "audiotext"]:
            src_inputs = {f"src_{k}": v for k, v in self.encoder.prepare_sample(inputs["src"]).items()}

        if self.input_modality in ["audio", "audiotext"]:
            src_inputs.update({f"src_{k}": v for k, v in self.encoder_model_audio.prepare_sample(inputs["src_audio"]).items()})


        mt_inputs = self.encoder.prepare_sample(inputs["mt"])

       
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        model_inputs = {**src_inputs, **mt_inputs}

        if stage == "predict":
            return model_inputs
        
        scores = [float(s["score"]) for s in sample]
        targets = Target(score=torch.tensor(scores, dtype=torch.float))

        if "system" in inputs:
            targets["system"] = inputs["system"]

        return model_inputs, targets

    def forward(
        self,
        src_input_ids: torch.tensor,
        src_attention_mask: torch.tensor,
        mt_input_ids: torch.tensor,
        mt_attention_mask: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """SpeechRegression model forward method.

        Args:
            src_input_ids [torch.tensor]: input ids from source sentences.
            src_attention_mask [torch.tensor]: Attention mask from source sentences.
            mt_input_ids [torch.tensor]: input ids from MT.
            mt_attention_mask [torch.tensor]: Attention mask from MT.

        Return:
            Prediction object with translation scores.
        """
        if self.input_modality == "text":
            src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        elif self.input_modality == "audio":
            waveforms = kwargs.pop("src_waveforms")
            src_sentemb = self.get_audio_embedding(waveforms)
        elif self.input_modality == "audiotext":
            text_emb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
            waveforms = kwargs.pop("src_waveforms")
            audio_emb = self.get_audio_embedding(waveforms)

            if self.fuse_emb_strategy == "concat":
                src_sentemb = self.fusion_proj(torch.cat([text_emb, audio_emb], dim=-1))
            elif self.fuse_emb_strategy == "sum":
                src_sentemb = self.fusion_layernorm(text_emb + audio_emb)
            elif self.fuse_emb_strategy == "avg":
                src_sentemb = (text_emb + audio_emb) / 2

        else:
            raise NotImplementedError("Only audio, audiotext and text are supported.")
        mt_sentemb = self.get_sentence_embedding(mt_input_ids, mt_attention_mask)
        diff_src = torch.abs(mt_sentemb - src_sentemb)
        prod_src = mt_sentemb * src_sentemb

        embedded_sequences = torch.cat(
            (mt_sentemb, src_sentemb, prod_src, diff_src), dim=1
        )

        return Prediction(score=self.estimator(embedded_sequences).view(-1))


    def _load_hf_dataset(self, path: str):
        if hasattr(self, '_hf_dataset_cache') and self._hf_dataset_cache[0] == path:
            return self._hf_dataset_cache[1]

        from datasets import load_dataset
        from collections import Counter
        rng = random.Random(SEED)

        hf_dataset = load_dataset(path)

        def prepare(ds):
            ds = ds.rename_columns({
                "src_text": "src",
                "tgt_text": "mt",
                "audio": "src_audio",
            })
            ds = ds.remove_columns(
                [c for c in ds.column_names if c not in {"src", "mt", "score", "src_audio"}]
            )
            # Filter samples that would exceed SONAR's 4096 frame limit
            # SONAR uses fbank (10ms hop = 160 samples at 16kHz) + fbank_stride=2 → 320 samples/frame
            # Use sf.info() (header-only, no torchcodec) to avoid leaking resources over ~500k calls
            max_duration = 4096 * 320 / 16000  # 81.92 seconds
            before = len(ds)
            def _duration_ok(x):
                import io, soundfile as sf
                audio = x["src_audio"]
                enc = getattr(audio, "_hf_encoded", None)
                try:
                    if enc and enc.get("bytes"):
                        return sf.info(io.BytesIO(enc["bytes"])).duration <= max_duration
                    elif enc and enc.get("path"):
                        return sf.info(enc["path"]).duration <= max_duration
                except Exception:
                    pass
                # fallback: torchcodec path
                return len(audio["array"]) / audio["sampling_rate"] <= max_duration
            ds = ds.filter(_duration_ok, load_from_cache_file=False)
            if before - len(ds) > 0:
                print(f"Filtered {before - len(ds)} samples exceeding SONAR max audio length (>{max_duration:.1f}s)")
            return ds

        if "validation" in hf_dataset:
            train_dataset = prepare(hf_dataset["train"])
            dev_dataset   = prepare(hf_dataset["validation"])
        else:
            dataset = prepare(hf_dataset["train"])
            TARGET = int(len(dataset) * 0.05)
            src_items = list(Counter(dataset["src"]).items())
            rng.shuffle(src_items)
            dev_srcs, count = set(), 0
            for src, size in src_items:
                if count + size > TARGET:
                    continue
                dev_srcs.add(src)
                count += size
                if count >= TARGET:
                    break
            train_indices = [i for i, src in enumerate(dataset["src"]) if src not in dev_srcs]
            dev_indices   = [i for i, src in enumerate(dataset["src"]) if src in dev_srcs]
            train_dataset = dataset.select(train_indices)
            dev_dataset   = dataset.select(dev_indices)

        self._hf_dataset_cache = (path, (train_dataset, dev_dataset))
        return self._hf_dataset_cache[1]


    def read_training_data(self, path: str) -> List[dict]:
        """Method that reads the training data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            df = df[["src", "mt", "score", "src_audio"]]
            df = df.dropna(subset=["src", "mt"])
            df = df[(df["src"].astype(str).str.strip() != "") & (df["mt"].astype(str).str.strip() != "")]
            df["src"] = df["src"].astype(str)
            df["mt"] = df["mt"].astype(str)
            df["src_audio"] = df["src_audio"].astype(str)
            df["score"] = df["score"].astype("float16")
            return df.to_dict("records")
        else:
            train_dataset, _ = self._load_hf_dataset(path)
            return train_dataset


    def read_validation_data(self, path: str) -> List[dict]:
        """Method that reads the validation data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            df = df[["src", "mt", "score", "src_audio"]]
            df = df.dropna(subset=["src", "mt"])
            df = df[(df["src"].astype(str).str.strip() != "") & (df["mt"].astype(str).str.strip() != "")]
            df["score"] = df["score"].astype("float16")
            df["src"] = df["src"].astype(str)
            df["mt"] = df["mt"].astype(str)
            df["src_audio"] = df["src_audio"].astype(str)
            return df.to_dict("records")
        else:
            _, dev_dataset = self._load_hf_dataset(path)
            return dev_dataset
