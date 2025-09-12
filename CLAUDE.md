# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Reflex-based web application for analyzing and downloading YouTube audio with BPM detection and metronome overlay functionality. The app allows users to either provide a YouTube URL or upload their own audio files for analysis.

## Commands

### Development
```bash
# Activate virtual environment and run the app
.\env.ps1

# Or manually:
cd env/Scripts
.\activate.ps1
cd ../..
reflex run
```

### Installation
```bash
# Install all required dependencies
pip install -r requirements.txt

# Or install individually with specific versions:
pip install reflex==0.8.10 yt-dlp==2025.9.5 librosa==0.11.0 numpy==2.2.6 pygame==2.6.1 pydub==0.25.1
```

## Architecture

The application consists of a single-page Reflex app with the following key components:

### Core Module: `descargas_youtube/descargas_youtube.py`
- **State Class**: Manages all application state including:
  - Audio analysis (BPM detection using librosa)
  - YouTube video downloading (using yt-dlp)
  - Audio playback with metronome (using pygame)
  - File upload handling for custom audio files
  - Tempo variations (slow/normal/fast modes)
  
### Key Features Implementation
- **BPM Analysis**: Uses librosa's beat tracking to detect tempo, with automatic calculation of half and double BPM values
- **Metronome Integration**: Overlays metronome clicks on audio using pydub, with adjustable volume
- **File Handling**: Manages temporary files for audio processing and cleanup
- **Progress Tracking**: Implements progress bars for download and processing operations
- **Playback System**: Uses pygame for audio preview with synchronized metronome

### State Management Pattern
The app uses Reflex's reactive state management where:
- All UI interactions trigger state methods (e.g., `get_info_and_analyze`, `play_preview`)
- State changes automatically update the UI through reactive bindings
- Async methods handle long-running operations like downloading and audio analysis

### File Structure
- `descargas_youtube/`: Main application module
- `assets/`: Static assets directory
- `uploaded_files/`: Directory for user-uploaded audio files
- `env.ps1`: PowerShell script for environment activation and app startup

## Important Notes

- The app now uses the latest versions of all dependencies (Reflex 0.8.10, librosa 0.11.0, etc.)
- Configuration includes disabled sitemap plugin to avoid warnings
- Audio files are processed in temporary directories and cleaned up after use
- Downloads are saved to the user's Downloads folder by default
- The app supports multiple audio formats for upload: mp3, wav, flac, aac, ogg, m4a
- Only minor pygame pkg_resources deprecation warning remains (library-level, doesn't affect functionality)