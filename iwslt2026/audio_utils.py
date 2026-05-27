"""
Audio loading utilities for the IWSLT 2026 test set.

The test set has two audio sources:
  - ACL domain: audio already segmented and stored in HF as an AudioDecoder object
  - Other domains: audio is None; segments must be extracted from long WAV/MP4 files
    using start_timestamp / end_timestamp and a local base directory.
"""
import os
import subprocess
import tempfile


def _resolve_audio_path(audio_path, audio_base_dir):
    """Map an HF audio_path to its local path.

    HF paths for non-ACL domains have a prefix like:
        OFFLINE_TRACK1-CALLCENTER__audios/2026/apptek_call_center/test2026/audio/file.wav
    Locally the files are stored without this prefix:
        <audio_base_dir>/apptek_call_center/test2026/audio/file.wav

    We strip the first two path components (e.g. OFFLINE_TRACK1-.../2026/) to get
    the relative path that exists under audio_base_dir.
    """
    parts = audio_path.replace("\\", "/").split("/")
    relative = "/".join(parts[2:])  # strip OFFLINE_TRACK1-.../2026/
    return os.path.join(audio_base_dir, relative)


def get_src_audio(entry, audio_base_dir, tmp_dir):
    """Return a WAV file path for a test set entry.

    Args:
        entry:          a row from the HF test dataset
        audio_base_dir: base directory for non-ACL audio files
        tmp_dir:        directory to write temporary WAV segments into

    Returns:
        str: path to a WAV file containing the audio segment
    """
    if entry["audio"] is not None:
        # ACL: decode audio and save to temp WAV so all entries are
        # consistently string paths (avoids mixed-type issues in model batching)
        import numpy as np
        import soundfile as sf
        audio = entry["audio"]
        if isinstance(audio, dict):
            # HF already-decoded dict: {"array": np.ndarray, "sampling_rate": int}
            audio_data = np.array(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
        else:
            # Lazy AudioDecoder — call get_all_samples()
            decoded = audio.get_all_samples()
            audio_data = decoded.data.squeeze(0).numpy()  # [samples] or [channels, samples]
            sr = int(decoded.sample_rate)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=0)
        doc_id = entry.get("doc_id", "unknown").replace("/", "_")
        out_path = os.path.join(tmp_dir, f"acl_{doc_id}.wav")
        sf.write(out_path, audio_data.astype(np.float32), sr)
        return out_path

    full_path = _resolve_audio_path(entry["audio_path"], audio_base_dir)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Audio file not found: {full_path}\n"
            f"  (audio_base_dir={audio_base_dir}, audio_path={entry['audio_path']})"
        )

    start = entry["start_timestamp"]
    end = entry["end_timestamp"]
    doc_id = entry.get("doc_id", "unknown").replace("/", "_")
    tgt_lang = entry.get("tgt_lang", "xx")
    out_path = os.path.join(tmp_dir, f"{doc_id}_{tgt_lang}_{start:.3f}_{end:.3f}.wav")

    if not os.path.exists(out_path):
        _extract_segment_ffmpeg(full_path, start, end, out_path)

    return out_path


def _extract_segment_ffmpeg(input_path, start, end, out_path, sr=16000):
    """Extract an audio segment to a mono 16 kHz WAV file using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", input_path,
            "-ar", str(sr),
            "-ac", "1",
            "-f", "wav",
            out_path,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {input_path} [{start}-{end}]:\n"
            + result.stderr.decode()
        )
