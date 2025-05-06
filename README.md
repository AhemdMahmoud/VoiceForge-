# 🎙️ Advanced Text-to-Speech with Transformer Models 🤖

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A comprehensive toolkit for fine-tuning and generating high-quality speech using state-of-the-art transformer-based TTS models like SpeechT5, Bark, and VITS.

## 🔍 Project Overview

This project provides tools and examples for working with modern text-to-speech (TTS) technologies using transformer architectures. It enables you to:

- 🔊 Generate high-quality speech from text in multiple languages
- 🛠️ Fine-tune pre-trained TTS models on custom datasets
- 🗣️ Use speaker embeddings to control voice characteristics
- 🌍 Support for multiple languages (English, Dutch, German)

### Technology Stack:
- 🐍 Python
- 🤗 Hugging Face Transformers
- 🔊 SpeechT5, Bark, and VITS models
- 📊 PyTorch & TensorFlow
- 🧠 SpeechBrain (for speaker embeddings)

## 📥 Installation Guide

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup Environment

```bash
# Create a virtual environment
python -m venv tts_env
source tts_env/bin/activate  # On Windows: tts_env\Scripts\activate

# Install required packages
pip install transformers datasets soundfile speechbrain accelerate torch
```

### Additional Dependencies

For visualization and audio processing:
```bash
pip install matplotlib ipython
```

## 💻 Usage Examples

### 1. Basic Text-to-Speech with SpeechT5

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
from IPython.display import Audio

# Load models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Prepare input text
inputs = processor(text="Hello, welcome to my project!", return_tensors="pt")

# Load speaker embeddings (required for SpeechT5)
# You can use the CMU-Arctic xvectors dataset or create your own
from datasets import load_dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[16]["xvector"]).unsqueeze(0)

# Generate speech
speech = model.generate_speech(inputs["input_ids"], 
                              speaker_embeddings=speaker_embeddings, 
                              vocoder=vocoder)

# Play the generated audio
Audio(speech.numpy(), rate=16000)
```

### 2. Using Bark for Expressive Speech

```python
from transformers import BarkProcessor, BarkModel
from IPython.display import Audio

# Load models
processor = BarkProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

# Prepare input with special tokens for expressions
inputs = processor(
    text="♪ This is an example with singing and [laughter] expressiveness!", 
    voice_preset="v2/en_speaker_3"
)

# Generate speech
speech_output = model.generate(**inputs).cpu()

# Play the generated audio
Audio(speech_output.numpy(), rate=24000)
```

### 3. Multi-language TTS with VITS Model

```python
from transformers import VitsModel, VitsTokenizer
import torch
from IPython.display import Audio

# Load German TTS model
model = VitsModel.from_pretrained("facebook/mms-tts-deu")
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-deu")

# German text example
text_example = "Guten Tag! Wie geht es Ihnen heute?"

# Process and generate speech
inputs = tokenizer(text_example, return_tensors="pt")
with torch.no_grad():
    outputs = model(inputs["input_ids"])

speech = outputs["waveform"]

# Play the generated audio
Audio(speech, rate=16000)
```

## 🚀 Fine-tuning Guide

You can fine-tune the SpeechT5 model on custom datasets. The project includes an example for fine-tuning on the Dutch language using the VoxPopuli dataset:

```python
# Fine-tuning process overview
# 1. Load and prepare dataset
datasets = load_dataset("facebook/voxpopuli", "nl", split="train")
datasets = datasets.cast_column("audio", Audio(sampling_rate=16000))

# 2. Clean text data and filter speakers
# 3. Generate speaker embeddings
# 4. Prepare and process data for training
# 5. Define data collator
# 6. Set up training arguments
# 7. Train the model

# For complete code, see the fine_tuning_speecht5.py file
```

## 🏗️ Project Structure

```
advanced-tts/
├── 📄 fine_tuning_speecht5.py     # Script for fine-tuning SpeechT5 model
├── 📄 pre_trained_models_for_text_to_speech.py  # Examples of using pre-trained models
├── 🗂️ models/                     # Directory for saved models
│   └── 📄 README.md               # Information about models
├── 📄 README.md                   # This file
└── 📄 requirements.txt            # Dependencies
```

## 📚 API Documentation

### SpeechT5 API

#### `SpeechT5Processor`
Processes text inputs for the SpeechT5 model.

**Methods:**
- `__call__(text, return_tensors)`: Converts text to input tensors.
  - Parameters:
    - `text`: String input text
    - `return_tensors`: Format of return tensors (typically "pt" for PyTorch)
  - Returns: Dictionary with input_ids

#### `SpeechT5ForTextToSpeech`
The main model for SpeechT5 text-to-speech generation.

**Methods:**
- `generate_speech(input_ids, speaker_embeddings, vocoder)`: Generates speech output
  - Parameters:
    - `input_ids`: Tokenized input text
    - `speaker_embeddings`: Tensor of speaker embeddings
    - `vocoder`: HiFi-GAN vocoder model
  - Returns: Tensor containing audio waveform

### Bark API

#### `BarkProcessor`
Processes text inputs for the Bark model.

**Methods:**
- `__call__(text, voice_preset)`: Prepares inputs for the model
  - Parameters:
    - `text`: String input text (can include special tokens)
    - `voice_preset`: Voice style preset id
  - Returns: Dictionary of model inputs

#### `BarkModel`
Generative text-to-speech model with expressiveness support.

**Methods:**
- `generate(**inputs)`: Generates speech from processed inputs
  - Parameters:
    - `inputs`: Dictionary from the processor
  - Returns: Tensor containing audio waveform

### VITS API

#### `VitsTokenizer`
Tokenizes text for VITS model processing.

**Methods:**
- `__call__(text, return_tensors)`: Tokenizes input text
  - Parameters:
    - `text`: String input text
    - `return_tensors`: Format of return tensors
  - Returns: Dictionary with input_ids

#### `VitsModel`
End-to-end text-to-speech model.

**Methods:**
- `__call__(input_ids)`: Generates speech from tokenized text
  - Parameters:
    - `input_ids`: Tokenized input text
  - Returns: Dictionary containing "waveform" key with audio data

## ⚙️ Configuration

### Model Configurations

Each model has configurable parameters that can be adjusted:

#### SpeechT5 Configuration
```python
# Set inference parameters
model.config.use_cache = False  # Important for training

# Training configuration
args = Seq2SeqTrainingArguments(
    output_dir="speecht5_finetuned_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    # Other parameters...
)
```

#### Bark Configuration
```python
# Voice preset options
# Change voice_preset to select different speakers:
# - "v2/en_speaker_0" through "v2/en_speaker_9": English speakers
# - "v2/de_speaker_0" through "v2/de_speaker_9": German speakers
# etc.

inputs = processor(text="Text to synthesize", voice_preset="v2/en_speaker_3")
```

## 🧪 Testing

To verify the models are working correctly:

```bash
# Run test script
python -c "
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
from IPython.display import Audio

processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')
vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

inputs = processor(text='This is a test of the TTS system.', return_tensors='pt')

# Load a sample speaker embedding (you'll need to provide this)
# This is just a placeholder:
speaker_embeddings = torch.randn(1, 512)

speech = model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=vocoder)
print('Speech generated successfully!')
"
```

## 🚢 Deployment

### Local Deployment

For local inference server:

```bash
# Install additional requirements
pip install fastapi uvicorn

# Create a simple API server (sample code)
# See documentation for complete implementation
```

### Cloud Deployment

The models can be deployed to:
- 🌐 Hugging Face Spaces
- ☁️ Google Colab
- 🖥️ AWS SageMaker

For Hugging Face deployment:
```bash
# Push your fine-tuned model to Hub
trainer.push_to_hub()

# Users can then load it with:
model = SpeechT5ForTextToSpeech.from_pretrained("your-username/model-name")
```

## 🤝 Contributing Guidelines

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions and classes
- Maintain test coverage

## ⚖️ License Information

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- Microsoft Research for the SpeechT5 model
- Suno.ai for the Bark model
- Facebook AI Research for the VITS model
- Hugging Face for the Transformers library
- SpeechBrain team for the speaker embedding tools
- The VoxPopuli dataset creators

## ❓ FAQ Section

### General Questions

**Q: Do I need a GPU for this project?**  
A: While not strictly required, a GPU significantly speeds up both inference and especially fine-tuning. For real-time TTS, a GPU is recommended.

**Q: Which model should I use?**  
A: 
- SpeechT5: Best for high-quality, natural-sounding speech with specific voice characteristics
- Bark: Best for expressive speech with emotions, singing, and special effects
- VITS: Good for multilingual support and efficiency

**Q: How do I create my own speaker embeddings?**  
A: You can use the SpeechBrain toolkit to extract x-vectors from your audio samples:

```python
import torch
from speechbrain.pretrained import EncoderClassifier

# Load the speaker embedding model
spk_model = EncoderClassifier.from_hparams(
    "speechbrain/spkrec-xvect-voxceleb", 
    run_opts={"device": "cuda"}
)

# Extract speaker embeddings from an audio file
with torch.no_grad():
    speaker_embeddings = spk_model.encode_batch(wavform)
    speaker_embeddings = torch.nn.functional.normalize(
        speaker_embeddings, dim=2
    ).squeeze().cpu().numpy()
```

### Troubleshooting

**Q: I'm getting CUDA out of memory errors during fine-tuning. What should I do?**  
A: Try these solutions:
- Reduce batch size (`per_device_train_batch_size`)
- Enable gradient checkpointing (`gradient_checkpointing=True`)
- Increase gradient accumulation steps
- Filter your dataset to shorter examples

**Q: The model generates unintelligible speech. What's wrong?**  
A: Check:
- If using a fine-tuned model, ensure it was trained on enough data
- Try different speaker embeddings
- For non-English text, ensure you're using a model that supports that language

## 📝 Changelog

### v1.0.0 (Initial Release)
- Added support for SpeechT5, Bark, and VITS models
- Included fine-tuning script for SpeechT5
- Added comprehensive examples for all models

## 📬 Contact Information

For questions and feedback:
- 🌐 GitHub: [Linked In](https://www.linkedin.com/in/ahmed-mahmoud-80356b220/)

---

Made with ❤️ by Hamedo
