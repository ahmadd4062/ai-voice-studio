# 🎙️ AI Voice Studio

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-24.0.0-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [License](#license)

---

## 📖 Overview

**AI Voice Studio** is a full-stack voice cloning and podcast generation platform. It allows users to:

- Clone any voice using F5-TTS
- Generate multi-character podcasts
- Support Hindi/Urdu with perfect pronunciation
- Run the entire app with one Docker command

This project was built as part of an internship at Zenvyro Labs.

---

## ✨ Features

### 🎭 Voice Cloning
- Upload any voice clip (8 seconds)
- F5-TTS clones the voice
- Save voices to library

### 🎙️ Multi-Voice Podcast
- Write scripts with multiple characters
- Each character speaks with their cloned voice
- Seamless audio stitching with pauses

### 🌏 Hindi / Urdu Support
- Auto-convert Roman Hindi → Devanagari script
- Perfect pronunciation using Microsoft Neural Voices
- Two voices: Madhur (Male) & Swara (Female)

### ✂️ Audio Editor
- Trim, cut, or replace audio segments
- Generate new segments with cloned voices

### 🧠 Voice Training Studio
- Upload long audio (10+ min)
- Noise filter (removes background static)
- Silence cutter (removes dead air)
- Normalize to 16kHz mono
- Chunk into 10-second training segments

### 🐳 Docker Support
- One-command deployment
- Volume support for saved voices
- Works on any system

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.10, Flask |
| **Voice Cloning** | F5-TTS |
| **Text-to-Speech** | Edge-TTS (Microsoft Neural Voices) |
| **Frontend** | Gradio |
| **Audio Processing** | PyDub, SoundFile, Librosa |
| **Speech Recognition** | Whisper |
| **Containerization** | Docker, Docker Compose |

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- FFmpeg
- Docker (optional)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/voice-studio-ai.git
cd voice-studio-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py