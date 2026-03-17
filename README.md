# PM Nam Converter

A modern Python desktop application to apply a digital gain multiplier to Neural Amp Modeler (.nam) files. This tool helps address volume loss when using .nam rigs on hardware units by boosting the weights within the JSON model.

## Features
- **Modern Dark UI:** Built with `customtkinter` for a sleek, amplifier-inspired look.
- **Gain Selection:** Quickly apply +0dB, +3dB, or +6dB gain offsets.
- **Efficient Processing:** High-speed weight multiplication with small output file sizes (no indentation).
- **Safe & Responsive:** Uses background threading to ensure the UI never freezes during processing.
- **Robust Error Handling:** Validates file structure and JSON integrity before processing.

## Tech Stack
- **Python 3**
- **CustomTkinter** (UI Framework)
- **JSON & Math** (Core Processing)
- **Threading** (Asynchronous Execution)

## How to Use
1. Run the application: `python main.py`
2. Click **Select .nam File** and choose your model.
3. Select your desired gain offset (+0dB, +3dB, or +6dB).
4. Click **Forge Rig**.
5. Your new rig will be saved in the same directory as the original file with a gain suffix (e.g., `my_rig_+3dB.nam`).

## Installation
Ensure you have the required dependencies:
```bash
pip install customtkinter
```
