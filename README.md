# PM Nam Converter - Vintage Edition

A comprehensive desktop suite for Neural Amp Modeler (.nam) rig management, featuring a vintage Fender Tweed UI.

## The Suite
- **The Butcher:** Applies digital gain multipliers and metadata tags (up to +14dB) to fix volume loss on hardware units like the Sonicake Pocketmaster.
- **The Baker Engine:** Simulates professional re-amping. Convolves a DI model with a Cabinet IR and retrains them into a single, hardware-ready Full Rig using PyTorch.
- **The Postman:** A local Wi-Fi Flask server that lets you wirelessly transfer rigs to and from your mobile device.
- **Global Tone Scanner:** Search, demo, and download rigs directly from the Tone3000 API.

## Setup
1. Ensure you have an active `.env` file containing `TONE3000_API_KEY=your_key_here`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python main.py`
