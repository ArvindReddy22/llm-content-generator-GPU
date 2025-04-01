# GPU-Accelerated AI Content Generator

An AI-powered content generation application that leverages GPU acceleration for faster blog creation using the Llama 2 language model.

## Features

- 🚀 GPU-accelerated content generation
- 🎤 Speech-to-text input capability using Vosk
- 📊 Interactive visualizations of content metrics
- ⚙️ Adaptable configuration for different GPU capabilities
- 💻 Low VRAM mode for 4GB GPUs (GTX 1650, RTX 3050, etc.)
- 📝 Multiple blog styles: Technical, Professional, Casual, Academic, Creative

## Requirements

- Python 3.8+
- PyTorch with CUDA support (for GPU acceleration)
- Streamlit
- Vosk speech recognition engine
- LangChain
- CTransformers
- Pandas and Plotly for visualizations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gpu-llm-content-generator.git
cd gpu-llm-content-generator

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Download the Llama 2 model
# You'll need to download the Llama 2 model files from Hugging Face or other sources
# Place them in the project directory

# Download the Vosk model
# Download vosk-model-small-en-us-0.15 and extract it to the project directory
```

## Usage

```bash
streamlit run optimized-gpu-llm.py
```

Navigate to the provided URL (usually http://localhost:8501) in your web browser.

## Model Files

You'll need to download these model files and place them in the project directory:

1. Llama 2 model files:
   - `llama-2-7b-chat.ggmlv3.q4_K_M.bin` (smaller, for low VRAM)
   - `llama-2-7b-chat.ggmlv3.q8_0.bin` (higher quality)

2. Vosk speech recognition model:
   - `vosk-model-small-en-us-0.15` directory

## License

[MIT License](LICENSE)
