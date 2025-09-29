# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

dots.ocr is a multilingual document layout parsing system using a 1.7B-parameter Vision-Language Model. It achieves state-of-the-art results on OmniDocBench with a unified model architecture for both layout detection and content recognition.

## Core Architecture

The system uses a single VLM model served via vLLM, accessed through an OpenAI-compatible API. Different parsing tasks (layout detection, OCR, grounding) are controlled by prompts rather than separate models.

Key modules:
- `dots_ocr/parser.py`: Main DotsOCRParser class orchestrating document parsing
- `dots_ocr/model/inference.py`: vLLM inference via OpenAI client
- `dots_ocr/utils/prompts.py`: Task-specific prompt templates
- `dots_ocr/utils/layout_utils.py`: Layout post-processing and visualization
- `dots_ocr/utils/format_transformer.py`: Output format conversion (JSON to Markdown/HTML)

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n dots_ocr python=3.12
conda activate dots_ocr

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install package
pip install -e .
```

### Model Deployment
```bash
# Download model weights (optional: uses huggingface_hub or modelscope)
python3 tools/download_model.py

# Launch vLLM server
bash demo/launch_model_vllm.sh

# Set inference server endpoint
export DOTS_OCR_INFERENCE_SERVER="http://localhost:8000/v1"
```

### Running Document Parsing
```bash
# Parse image with full layout + OCR
python3 dots_ocr/parser.py demo/demo_image1.jpg

# Parse PDF with multiple threads
python3 dots_ocr/parser.py demo/demo_pdf1.pdf --num_thread 64

# Layout detection only
python3 dots_ocr/parser.py demo/demo_image1.jpg --prompt prompt_layout_only_en

# OCR only (text extraction)
python3 dots_ocr/parser.py demo/demo_image1.jpg --prompt prompt_ocr

# Grounded OCR (extract text from specific bbox)
python3 dots_ocr/parser.py demo/demo_image1.jpg --prompt prompt_grounding_ocr --bbox 163 241 1536 705
```

### Demo Applications
```bash
# Gradio web interface
python3 demo/demo_gradio.py

# Streamlit interface
streamlit run demo/demo_streamlit.py

# Simple CLI demo
python3 demo/demo_vllm.py
```

## Important Implementation Notes

1. **vLLM Registration**: The model requires patching vLLM's entrypoint to import custom modeling code (see `launch_model_vllm.sh`)

2. **Inference Server**: The system expects a vLLM server running with the DotsOCR model. Set `DOTS_OCR_INFERENCE_SERVER` environment variable to the server endpoint.

3. **Prompt Modes**: Available in `dots_ocr/utils/prompts.py`:
   - `prompt_layout_all_en`: Full layout detection + text extraction
   - `prompt_layout_only_en`: Layout detection without text
   - `prompt_ocr`: Text extraction only
   - `prompt_grounding_ocr`: Text from specific bounding box

4. **Output Formats**: Parser generates:
   - `.json`: Structured layout data with bboxes
   - `.md`: Markdown-formatted text content
   - `.jpg`: Visualization with bounding boxes
   - `_nohf.md`: Markdown without headers/footers (for benchmarks)

5. **Image Processing**: The system handles smart resizing with configurable MIN_PIXELS and MAX_PIXELS constraints to optimize model performance.

6. **PDF Processing**: Uses multiprocessing for efficient page-by-page processing with configurable thread count.

## Testing and Validation

No formal test suite exists. Manual testing can be done using demo scripts and sample files in `demo/` directory.

For benchmark evaluation, refer to `tools/eval_omnidocbench.md` for instructions on running OmniDocBench metrics.

## Common Issues and Solutions

1. **CUDA Memory**: Adjust `--gpu-memory-utilization` in launch script if OOM occurs
2. **Thread Count**: For large PDFs, increase `--num_thread` for faster processing
3. **Model Download**: If download fails, try `--type modelscope` option for alternative source
4. **vLLM Version**: Requires vLLM v0.9.1 specifically due to custom model integration

## Code Style Guidelines

- The codebase follows Python conventions but lacks formal linting
- File organization: utilities in `utils/`, models in `model/`, demos in `demo/`
- No pre-commit hooks or formatters are configured
- When modifying, maintain consistency with existing code patterns