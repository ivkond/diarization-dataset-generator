# Project Summary

## Overall Goal
Generate synthetic multi-speaker audio datasets for speaker diarization tasks, specifically for Russian language from the FBK-MT/Speech-MASSIVE-test dataset with realistic features like overlaps, simultaneous speech, and background noise.

## Key Knowledge
- **Technology Stack**: Python 3.10+, Hugging Face Datasets, Pydantic for config validation, NumPy for audio processing, PyArrow for Parquet support
- **Project Structure**: Modular with src/audio, src/config, src/dataset, src/patterns, src/storage, src/track directories
- **Configuration**: YAML-based with Pydantic validation, supports both local Parquet output and direct HuggingFace Hub upload
- **Output Format**: Parquet files compatible with Hugging Face Datasets, containing audio waveforms and structured metadata
- **Key Features**: Multi-speaker conversations (2-4 speakers), overlaps, simultaneous speech, background noise, difficulty levels
- **Parallel Processing**: Uses ProcessPoolExecutor for efficient track generation with worker initialization
- **Build Commands**: `uv sync` for dependencies, `python -m src.main` for execution

## Recent Actions
- [DONE] Analyzed the complete project structure including all source modules
- [DONE] Examined configuration files (config.yaml, pyproject.toml) and main entry points
- [DONE] Reviewed core modules: track generation, audio processing, dataset handling, pattern selection
- [DONE] Created comprehensive QWEN.md file with project overview, architecture, and conventions
- [DONE] Documented all key components and their functions

## Current Plan
- [DONE] Project analysis and documentation complete
- [TODO] Ready for future interactions based on the comprehensive context provided in QWEN.md

---

## Summary Metadata
**Update time**: 2025-11-18T09:09:22.016Z 
