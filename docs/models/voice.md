# Voice Module: Technical Report

This document provides a technical overview of the AI models used in the underthesea voice (text-to-speech) module.

## Overview

The voice module implements a neural text-to-speech (TTS) system for Vietnamese. It is based on [VietTTS](https://github.com/ntt123/vietTTS) by NTT123 and uses a two-stage architecture:

1. **Text-to-Mel**: Converts text/phonemes to mel-spectrogram
2. **Mel-to-Wave (Vocoder)**: Converts mel-spectrogram to audio waveform

```
Text → [Text Normalization] → [Duration Model] → [Acoustic Model] → Mel → [HiFi-GAN] → Audio
```

## Installation

```bash
pip install "underthesea[voice]"
underthesea download-model VIET_TTS_V0_4_1
```

## Model Architecture

### 1. Duration Model

The Duration Model predicts the duration (in seconds) for each phoneme in the input sequence.

**Architecture:**

| Component | Description |
|-----------|-------------|
| Token Encoder | Embedding + 3× Conv1D + Bidirectional LSTM |
| Projection | Linear → GELU → Linear → Softplus |

**Parameters:**

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 256 |
| LSTM Dimension | 256 |
| Dropout Rate | 0.5 |

**Input:** Phoneme sequence with lengths
**Output:** Duration for each phoneme (in seconds)

### 2. Acoustic Model

The Acoustic Model generates mel-spectrograms from phonemes and their predicted durations.

**Architecture:**

| Component | Description |
|-----------|-------------|
| Token Encoder | Embedding + 3× Conv1D + Bidirectional LSTM |
| Upsampling | Gaussian attention-based upsampling |
| PreNet | 2× Linear (256 dim) with dropout |
| Decoder | 2× LSTM with skip connections |
| PostNet | 5× Conv1D with batch normalization |
| Projection | Linear to mel dimension |

**Parameters:**

| Parameter | Value |
|-----------|-------|
| Encoder Dimension | 256 |
| Decoder Dimension | 512 |
| PostNet Dimension | 512 |
| Mel Dimension | 80 |

**Key Features:**

- **Gaussian Upsampling**: Uses soft attention to upsample encoder outputs to match target frame length
- **Autoregressive Decoder**: Generates mel frames sequentially with teacher forcing during training
- **Zoneout Regularization**: Applies zoneout to LSTM states during training for better generalization
- **PostNet Refinement**: Residual convolutional network refines the predicted mel-spectrogram

### 3. HiFi-GAN Vocoder

The vocoder converts mel-spectrograms to raw audio waveforms using the HiFi-GAN architecture.

**Architecture:**

| Component | Description |
|-----------|-------------|
| Conv Pre | Conv1D (7 kernel) |
| Upsampling | Multiple Conv1DTranspose layers |
| Multi-Receptive Field Fusion (MRF) | ResBlocks with varying kernel sizes and dilations |
| Conv Post | Conv1D (7 kernel) + Tanh |

**Key Features:**

- **Multi-Scale Upsampling**: Progressive upsampling from mel frame rate to audio sample rate
- **Multi-Receptive Field Fusion**: Combines outputs from residual blocks with different receptive fields
- **Leaky ReLU Activation**: Uses leaky ReLU with slope 0.1 throughout

**ResBlock Types:**

- **ResBlock1**: 3 dilated convolutions (dilation: 1, 3, 5) with residual connections
- **ResBlock2**: 2 dilated convolutions (dilation: 1, 3) with residual connections

## Audio Configuration

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16,000 Hz |
| FFT Size | 1,024 |
| Mel Channels | 80 |
| Frequency Range | 0 - 8,000 Hz |

## Text Processing Pipeline

### 1. Text Normalization

The input text is normalized before synthesis:

```python
# Normalization steps:
1. Unicode NFKC normalization
2. Lowercase conversion
3. Punctuation → silence markers
4. Multiple spaces → single space
```

### 2. Phoneme Conversion

Text is converted to phonemes using a lexicon lookup:

- Vietnamese characters are mapped to phoneme sequences
- Special tokens: `sil` (silence), `sp` (short pause), ` ` (word boundary)
- Unknown words are processed character-by-character

## Model Files

The `VIET_TTS_V0_4_1` model package includes:

| File | Description |
|------|-------------|
| `lexicon.txt` | Word-to-phoneme mapping |
| `duration_latest_ckpt.pickle` | Duration model weights |
| `acoustic_latest_ckpt.pickle` | Acoustic model weights |
| `hk_hifi.pickle` | HiFi-GAN vocoder weights |
| `config.json` | HiFi-GAN configuration |

## Framework Dependencies

The voice module uses JAX ecosystem:

| Library | Purpose |
|---------|---------|
| JAX | Numerical computation and automatic differentiation |
| JAXlib | JAX backend (CPU/GPU/TPU support) |
| dm-haiku | Neural network library for JAX |
| Optax | Gradient processing and optimization |

## Usage Example

```python
from underthesea.pipeline.say import say

# Basic usage
say("Xin chào Việt Nam")  # Creates sound.wav

# Custom output file
say("Hà Nội là thủ đô", outfile="output.wav")

# With playback
say("Đây là một ví dụ", play=True)
```

## Performance Considerations

- **First Call Latency**: Model loading on first call may take several seconds
- **JAX Compilation**: JIT compilation occurs on first inference, subsequent calls are faster
- **Text Length**: Maximum recommended text length is 500 characters
- **Memory Usage**: GPU memory usage depends on input text length

## Limitations

- **Single Speaker**: Current model supports only one voice
- **Vietnamese Only**: Designed specifically for Vietnamese language
- **Prosody**: Limited control over prosody and emotion
- **Real-time**: Not optimized for real-time streaming

## References

- [VietTTS](https://github.com/ntt123/vietTTS) - Original implementation by NTT123
- [HiFi-GAN](https://arxiv.org/abs/2010.05646) - Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis
- [Non-Attentive Tacotron](https://arxiv.org/abs/2010.04301) - Robust and Controllable Neural TTS Synthesis
- [dm-haiku](https://github.com/deepmind/dm-haiku) - JAX neural network library by DeepMind
