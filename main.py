import customtkinter as ctk
import os
import json
import threading
import requests
import webbrowser
import math
import scipy.io.wavfile as wav
import numpy as np
import gc
import multiprocessing
from scipy.signal import fftconvolve
import socket
from flask import Flask, request, send_from_directory, render_template_string
from werkzeug.serving import make_server
import qrcode
from PIL import Image, ImageTk
from tkinter import filedialog
from dotenv import load_dotenv
import soundfile as sf
import sys
import re

# Load environment variables
load_dotenv()

def create_tiled_background(window_width, window_height, pattern_path):
    """Creates a tiled image to fill the given window dimensions."""
    try:
        pattern = Image.open(pattern_path)
        pattern_width, pattern_height = pattern.size
        
        # Calculate how many full tiles are needed to cover the width and height
        tiles_x = (window_width // pattern_width) + 1
        tiles_y = (window_height // pattern_height) + 1
        
        # Create a new, transparent image to hold the tiled result
        tiled_image = Image.new('RGBA', (window_width, window_height))
        
        # Paste the repeated pattern tiles perfectly, starting from the top-left
        for i in range(tiles_x):
            for j in range(tiles_y):
                tiled_image.paste(pattern, (i * pattern_width, j * pattern_height))
                
        # Convert to a CTkImage for high-DPI scaling support
        return ctk.CTkImage(light_image=tiled_image, dark_image=tiled_image, size=(window_width, window_height))
        
    except Exception as e:
        print(f"DEBUG: Error creating tiled background: {e}")
        return None

# Appearance & Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Constants (Relative to script directory for reliability)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "downloads"))
BAKED_DIR = os.path.normpath(os.path.join(DOWNLOADS_DIR, "Baked_Rigs"))
RIGS_DIR = os.path.normpath(os.path.join(DOWNLOADS_DIR, "Full_Rigs"))
DI_DIR = os.path.normpath(os.path.join(DOWNLOADS_DIR, "DI_Amps"))
IR_DIR = os.path.normpath(os.path.join(DOWNLOADS_DIR, "Cabinet_IRs"))
for d in [DOWNLOADS_DIR, BAKED_DIR, RIGS_DIR, DI_DIR, IR_DIR]:
    os.makedirs(d, exist_ok=True)

# API Configuration
TONE3000_API_KEY = os.getenv("TONE3000_API_KEY")
API_BASE_URL = "https://www.tone3000.com/api/v1"
API_AUTH_URL = f"{API_BASE_URL}/auth/session"
API_SEARCH_URL = f"{API_BASE_URL}/tones/search"

# Baker Constants
ASSETS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "assets"))
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR, exist_ok=True)
INPUT_WAV_PATH = os.path.join(ASSETS_DIR, "input.wav")
INPUT_WAV_URL = "https://raw.githubusercontent.com/sdatkinson/neural-amp-modeler/main/input.wav"
TWEED_PATTERN_PATH = os.path.join(ASSETS_DIR, "tweed_pattern_seamless.png")

# Tweed Theme Constants
TWEED_BG = "#D1A95F"       # Warm vintage tan
OXBLOOD_PANEL = "#3B1C1A"  # Dark reddish-brown faceplate
PANEL_TEXT = "#F4EBD9"     # Cream/Vintage white for text on Oxblood
TWEED_TEXT = "#2A1A10"     # Dark brown for text on Tweed

class AmpKnob(ctk.CTkCanvas):
    def __init__(self, master, command=None, **kwargs):
        # Default bg to Oxblood to match the panel
        bg_color = kwargs.get('bg', OXBLOOD_PANEL)
        super().__init__(master, width=80, height=80, bg=bg_color, highlightthickness=0)
        self.command = command
        # Positions: 0dB, +3dB, +6dB, +9dB, +12dB, +14dB
        self.positions = [180, 144, 108, 72, 36, 0] # Angles in degrees
        self.current_pos_index = 0 
        
        self.bind("<B1-Motion>", self.turn_knob)
        self.bind("<Button-1>", self.turn_knob)
        self.draw_knob()

    def draw_knob(self):
        self.delete("all")
        # Draw the shadow/base
        self.create_oval(10, 10, 70, 70, fill="#1A1A1A", outline="#000000", width=2)
        
        # Calculate indicator line end point
        angle_rad = math.radians(self.positions[self.current_pos_index])
        x_center, y_center = 40, 40
        radius = 25
        x_end = x_center + radius * math.cos(angle_rad)
        y_end = y_center - radius * math.sin(angle_rad) 
        
        # Draw vintage white indicator line
        self.create_line(x_center, y_center, x_end, y_end, fill="#F4EBD9", width=4, capstyle="round")
        
        # Small center cap
        self.create_oval(35, 35, 45, 45, fill="#333333", outline="#111111")
        
    def turn_knob(self, event):
        x_center, y_center = 40, 40
        dx = event.x - x_center
        dy = y_center - event.y # Invert Y
        
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
            
        # Snap to nearest position
        closest_index = min(range(len(self.positions)), key=lambda i: abs(self.positions[i] - angle))
        
        if self.current_pos_index != closest_index:
            self.current_pos_index = closest_index
            self.draw_knob()
            
            # Pass the corresponding dB value back to the app
            db_values = [0, 3, 6, 9, 12, 14]
            if self.command:
                self.command(db_values[self.current_pos_index])

class VacuumTube(ctk.CTkCanvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, width=40, height=85, bg=kwargs.get('bg', '#3B1C1A'), highlightthickness=0)
        self.is_on = False
        self.is_animating = False
        self.glow_state = 0.0  # Range 0.0 to 1.0
        self.glow_dir = 1
        self.draw_tube()

    def draw_tube(self):
        self.delete("all")
        # 1. Base Pins
        for x in [14, 20, 26]:
            self.create_line(x, 70, x, 80, fill="#888888", width=2)
        
        # 2. Black Plastic Base
        self.create_rectangle(10, 65, 30, 72, fill="#1A1A1A", outline="#000000")
        
        # 3. Internal Plates (Dark grey structure)
        self.create_rectangle(14, 25, 26, 55, fill="#2A2A2A", outline="#111111")
        
        # 4. Heater Filament / Glow
        if self.is_on:
            # Colors array from dark red to bright hot yellow
            colors = ["#4A0000", "#8B0000", "#CC3300", "#FF6600", "#FFCC00"]
            idx = int(self.glow_state * 4)
            glow_color = colors[idx]
            core_color = "#FFFFDD" if idx > 2 else glow_color
            
            # Draw the glowing heater
            self.create_rectangle(16, 30, 24, 50, fill=glow_color, outline="")
            self.create_line(20, 25, 20, 55, fill=core_color, width=2)
        else:
            # Cold filament
            self.create_line(20, 25, 20, 55, fill="#111111", width=2)
            
        # 5. Glass Envelope (drawn last so it overlaps)
        self.create_polygon(12, 65,  10, 45,  10, 20,  20, 8,  30, 20,  30, 45,  28, 65, 
                            fill="", outline="#AACCFF", smooth=True, width=1)
        # Simple glass glare/reflection
        self.create_line(13, 22, 13, 40, fill="#FFFFFF", width=1)

    def start_glow(self):
        self.is_animating = True
        self.is_on = True
        self.animate_glow()

    def stop_glow(self):
        self.is_animating = False
        self.is_on = False
        self.glow_state = 0.0
        self.draw_tube()

    def animate_glow(self):
        if not self.is_animating:
            return
            
        # Throb the glow state up and down smoothly
        self.glow_state += 0.05 * self.glow_dir
        if self.glow_state >= 1.0:
            self.glow_state = 1.0
            self.glow_dir = -1
        elif self.glow_state <= 0.3: # Don't go entirely dark when under load
            self.glow_state = 0.3
            self.glow_dir = 1
            
        self.draw_tube()
        self.after(50, self.animate_glow) # 50ms = smooth 20fps animation

class ProgressInterceptor:
    """Intercepts stdout/stderr to parse PyTorch Lightning epoch progress."""
    def __init__(self, original_stream, queue):
        self.original_stream = original_stream
        self.queue = queue
        # Regex to match the 'Epoch 12/99' console output
        self.epoch_pattern = re.compile(r'Epoch (\d+)/(\d+)')

    def write(self, text):
        # Always pass the text to the real terminal so logs still appear
        self.original_stream.write(text)
        
        # Check for epoch progress in this chunk of text
        match = self.epoch_pattern.search(text)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            if total > 0:
                self.queue.put(("progress", current / total))
    
    def flush(self):
        self.original_stream.flush()

def bake_worker(di_path, ir_path, input_wav_path, output_dir, model_name, queue, epochs_to_run):
    """
    Isolated process for the Baker Engine pipeline.
    Handles Phase 4.1 to 4.5.
    """
    try:
        import torch
        import traceback
        from nam.models._from_nam import init_from_nam
        from nam.train.core import train

        # Safely load audio natively as normalized float32
        input_audio, rate = sf.read(input_wav_path, dtype='float32')
        if len(input_audio.shape) > 1:
            input_audio = input_audio[:, 0] # Mono conversion
        
        # Phase 4.2: Inference (The DI Amp)
        queue.put(("status", "Phase 4.2: Running DI Model Inference..."))
        # Load the NAM JSON dictionary first
        with open(di_path, "r", encoding="utf-8") as f:
            di_config = json.load(f)
            
        di_model = init_from_nam(di_config)
        input_tensor = torch.from_numpy(input_audio).view(1, -1)
        
        with torch.no_grad():
            amp_output_audio = di_model(input_tensor)
            # Handle possible sequence outputs
            if isinstance(amp_output_audio, (list, tuple)):
                amp_output_audio = amp_output_audio[0]
            amp_output_audio = amp_output_audio.numpy().flatten()

        # Phase 4.3: DSP Convolution (The Cabinet IR)
        queue.put(("status", "Phase 4.3: Convolving Cabinet IR..."))
        ir_audio, ir_rate = sf.read(ir_path, dtype='float32')
        if len(ir_audio.shape) > 1:
            ir_audio = ir_audio[:, 0]
            
        # Use mode='full' and slice to preserve the initial trigger spike
        convolved_audio = fftconvolve(amp_output_audio, ir_audio, mode='full')[:len(amp_output_audio)]

        # Prevent mathematical wrapping below the NAM validation threshold
        output_audio = np.clip(convolved_audio, -0.99, 0.99)
        temp_target_path = os.path.join(output_dir, "baker_target_tmp.wav")
        sf.write(temp_target_path, output_audio, rate, subtype='PCM_16')

        # Phase 4.5: The Retraining Loop
        queue.put(("status", "Phase 4.5: Retraining Full Rig Neural Model..."))
        
        # Wiretap the terminal output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = ProgressInterceptor(original_stdout, queue)
        sys.stderr = ProgressInterceptor(original_stderr, queue)
        
        try:
            # The trainer will create {model_name}.nam in output_dir
            train(
                input_wav_path,    # source
                temp_target_path,  # target
                output_dir,        # export dir
                epochs=epochs_to_run,        # dynamic for re-bake
                architecture="standard",
                modelname=model_name,
                silent=False       # MUST be False to generate console output
            )
        finally:
            # ALWAYS restore the original terminal streams
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        # Finalize
        if os.path.exists(temp_target_path):
            os.remove(temp_target_path)
            
        queue.put(("success", f"Baker Session Complete! Saved to {output_dir}"))

    except Exception as e:
        err_type = type(e).__name__
        tb = traceback.format_exc()
        print(f"DEBUG: Baker Worker Exception: {tb}")
        queue.put(("error", f"{err_type}: {str(e)}"))
    finally:
        # Phase 4 Memory Management
        if 'input_audio' in locals(): del input_audio
        if 'amp_output_audio' in locals(): del amp_output_audio
        if 'convolved_audio' in locals(): del convolved_audio
        if 'final_target_audio' in locals(): del final_target_audio
        gc.collect()

class PMNamConverter(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PM Nam Converter - Vintage Edition")
        self.geometry("900x900")
        self.configure(fg_color=TWEED_BG)
        self.resizable(True, True)
        self.minsize(600, 650)

        # Variables
        self.selected_file_path = ctk.StringVar(value="No file selected")
        self.gain_db = ctk.IntVar(value=0)
        self.search_query = ctk.StringVar()
        self.access_token = None

        # Baker Variables
        self.baker_di_path = ctk.StringVar(value="No DI model selected")
        self.baker_ir_path = ctk.StringVar(value="No IR cabinet selected")
        self.baker_di_display = ctk.StringVar(value="No DI model selected")
        self.baker_ir_display = ctk.StringVar(value="No Cabinet IR selected")
        self.baker_epochs = ctk.IntVar(value=15) # Default to Quick for CPU sanity
        
        # Search Filters
        self.gear_filter = ctk.StringVar(value="Full Rig")

        # UI Components
        self.setup_ui()
        
        # Audio Engine Setup
        self.after(500, self.ensure_assets)
        
        # Bind resize for tiled background
        self.bind("<Configure>", self.update_background)

    def get_local_ip(self):
        """Utility to find the local IPv4 address on the Wi-Fi network."""
        try:
            # Create a dummy socket to find the primary interface IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def update_background(self, event=None):
        """Dynamically updates the tiled background pattern on window resize."""
        # Use current window dimensions, or fallback to default size
        width = self.winfo_width() if self.winfo_width() > 1 else 1000
        height = self.winfo_height() if self.winfo_height() > 1 else 700
        
        # Generate the seamless tiled CTkImage
        self.bg_image_tiled = create_tiled_background(width, height, TWEED_PATTERN_PATH)
        
        if self.bg_image_tiled:
            # Update the background label with the new tiled pattern
            self.bg_label.configure(image=self.bg_image_tiled)

    def setup_ui(self):
        # Initial empty label for background, will be tiled on first resize/call
        self.bg_label = ctk.CTkLabel(self, text="")
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Header
        self.header_label = ctk.CTkLabel(
            self, 
            text="Butcher | The Baker | The Postman", 
            font=ctk.CTkFont(family="Century Schoolbook", size=26, weight="bold"),
            text_color=TWEED_TEXT,
            fg_color="transparent" # Explicit transparency fix
        )
        self.header_label.pack(pady=(25, 15))

        # Main Layout (Sidebar + Content)
        self.main_layout = ctk.CTkFrame(self, fg_color="transparent")
        self.main_layout.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # --- Sidebar: Tone3000 Search ---
        self.sidebar = ctk.CTkFrame(self.main_layout, width=380, corner_radius=12, fg_color="transparent", border_width=3, border_color="#8B6D3B")
        self.sidebar.pack(side="left", fill="both", padx=(0, 15))
        self.sidebar.pack_propagate(False)

        self.search_header = ctk.CTkLabel(self.sidebar, text="GLOBAL TONE SCANNER", text_color=TWEED_TEXT, font=ctk.CTkFont(size=18, weight="bold"))
        self.search_header.pack(pady=(20, 10))

        # Search Controls
        self.search_input_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.search_input_frame.pack(padx=20, pady=5, fill="x")

        self.search_entry = ctk.CTkEntry(
            self.search_input_frame, textvariable=self.search_query, 
            placeholder_text="Find your sound...",
            fg_color="#F4EBD9", text_color="#2A1A10", border_color="#8B6D3B"
        )
        self.search_entry.pack(fill="x", pady=(0, 10))
        self.search_entry.bind("<Return>", lambda event: self.start_search_thread())

        self.filter_frame = ctk.CTkFrame(self.search_input_frame, fg_color="transparent")
        self.filter_frame.pack(fill="x", pady=5)
        
        self.gear_menu = ctk.CTkOptionMenu(
            self.filter_frame, 
            values=["Full Rig", "DI / Amp", "IR (Cab)"],
            variable=self.gear_filter,
            fg_color="#3B1C1A", button_color="#5A2E2A", text_color="#F4EBD9"
        )
        self.gear_menu.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.search_button = ctk.CTkButton(
            self.filter_frame, text="SCAN", width=90,
            command=self.start_search_thread,
            fg_color="#3B1C1A", hover_color="#5A2E2A", text_color="#F4EBD9"
        )
        self.search_button.pack(side="right")

        # Results Area
        self.results_frame = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent", scrollbar_button_color="#8B6D3B")
        self.results_frame.pack(padx=12, pady=15, fill="both", expand=True)
        
        self.results_header_row = ctk.CTkFrame(self.results_frame, fg_color="#B38D4F", height=35)
        self.results_header_row.pack(fill="x", pady=(0, 5))
        self.results_header_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.results_header_row, text="DATABASE RESULTS", text_color="#F4EBD9", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, padx=15, sticky="w")

        # --- Content Area: Processing Tabs ---
        self.content_area = ctk.CTkFrame(self.main_layout, fg_color="transparent")
        self.content_area.pack(side="right", fill="both", expand=True)

        self.tabview = ctk.CTkTabview(
            self.content_area, corner_radius=12,
            fg_color="transparent", segmented_button_fg_color="#B38D4F",
            segmented_button_selected_color="#3B1C1A",
            segmented_button_selected_hover_color="#5A2E2A",
            text_color="#F4EBD9"
        )
        self.tabview.pack(fill="both", expand=True)

        self.tabview.add("The Butcher")
        self.tabview.add("The Baker Engine")
        self.tabview.add("The Postman")

        # --- Tab 1: The Butcher ---
        self.converter_tab = self.tabview.tab("The Butcher")
        
        self.processor_frame = ctk.CTkFrame(self.converter_tab, corner_radius=15, fg_color=OXBLOOD_PANEL, border_width=4, border_color="#222222")
        self.processor_frame.pack(padx=30, pady=30, fill="both", expand=True)

        ctk.CTkLabel(self.processor_frame, text="VALVE SCALER PANEL", text_color=PANEL_TEXT, font=ctk.CTkFont(size=20, weight="bold", family="Courier")).pack(pady=(25, 10))
        
        self.file_path_label = ctk.CTkLabel(
            self.processor_frame, textvariable=self.selected_file_path, wraplength=450, 
            font=ctk.CTkFont(size=12, family="Courier"), text_color="#A8A8A8"
        )
        self.file_path_label.pack(pady=10)

        self.select_button = ctk.CTkButton(
            self.processor_frame, text="OPEN RIG FILE", 
            command=self.select_file,
            fg_color="#5A2E2A", hover_color="#7A3E38", text_color=PANEL_TEXT
        )
        self.select_button.pack(pady=5)

        # Gain Knob Section
        self.knob_frame = ctk.CTkFrame(self.processor_frame, fg_color="transparent")
        self.knob_frame.pack(pady=25)
        
        ctk.CTkLabel(self.knob_frame, text="OUTPUT GAIN", text_color=PANEL_TEXT, font=ctk.CTkFont(size=14, weight="bold", family="Courier")).pack()
        
        # Specialized Oxblood frame for the knob to look integrated
        self.oxblood_frame = ctk.CTkFrame(self.knob_frame, fg_color=OXBLOOD_PANEL)
        self.oxblood_frame.pack(pady=10)
        
        self.gain_knob = AmpKnob(master=self.oxblood_frame, command=self.update_gain_and_warning)
        self.gain_knob.pack()
        
        labels_frame = ctk.CTkFrame(self.knob_frame, fg_color="transparent")
        labels_frame.pack()
        for val in [0, 3, 6, 9, 12, 14]:
            ctk.CTkLabel(labels_frame, text=f"{val}dB", text_color=PANEL_TEXT, font=ctk.CTkFont(size=10)).pack(side="left", padx=5)

        self.forge_button = ctk.CTkButton(
            self.processor_frame, text="ENGAGE SCALER", 
            font=ctk.CTkFont(size=20, weight="bold", family="Century Schoolbook"), height=60, 
            command=self.start_forging,
            fg_color="#8d1f1f", hover_color="#ba2b2b", text_color=PANEL_TEXT
        )
        self.forge_button.pack(pady=(20, 10), padx=40, fill="x")

        self.high_gain_warning = ctk.CTkLabel(self.processor_frame, text="Caution: High boosts may raise the noise floor or clip hardware.", text_color="#ff4d4d", font=ctk.CTkFont(size=11))
        # Initial pack depends on default value (0), but we'll manage it via callback

        # --- Tab 2: The Baker Engine ---
        self.baker_tab = self.tabview.tab("The Baker Engine")
        self.setup_baker_ui()

        # Shared Status UI
        self.status_container = ctk.CTkFrame(self.content_area, fg_color="transparent")
        self.status_container.pack(fill="x", pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(self.status_container, height=14, fg_color="#3B1C1A", progress_color="#44ff44")
        self.progress_bar.pack(padx=60, pady=(5, 5), fill="x")
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(self.status_container, text="SYSTEM STATUS: IDLE", text_color=TWEED_TEXT, font=ctk.CTkFont(weight="bold"))
        self.status_label.pack(pady=(0, 15))

        # --- Tab 3: The Postman (Wi-Fi Transfer) ---
        self.postman_tab = self.tabview.tab("The Postman")
        self.setup_postman_ui()

    def setup_postman_ui(self):
        # Description
        ctk.CTkLabel(
            self.postman_tab, 
            text="THE POSTMAN", 
            text_color=TWEED_TEXT,
            font=ctk.CTkFont(size=24, weight="bold", family="Century Schoolbook")
        ).pack(pady=(25, 10))
        
        ctk.CTkLabel(
            self.postman_tab, 
            text="Global wireless rig distribution. Scan the code to beam your tones to any device on the network.",
            text_color=TWEED_TEXT,
            wraplength=600, font=ctk.CTkFont(size=14, slant="italic")
        ).pack(pady=10)

        self.postman_frame = ctk.CTkFrame(self.postman_tab, corner_radius=15, fg_color=OXBLOOD_PANEL, border_width=4, border_color="#222222")
        self.postman_frame.pack(padx=60, pady=25, fill="both")

        self.server_status_label = ctk.CTkLabel(self.postman_frame, text="POSTAL STATUS: STATION CLOSED", font=ctk.CTkFont(weight="bold", family="Courier"), text_color="#ff4d4d")
        self.server_status_label.pack(pady=(25, 15))

        self.server_url_label = ctk.CTkLabel(self.postman_frame, text="URL: ---", text_color=PANEL_TEXT, font=ctk.CTkFont(family="Courier", size=15))
        self.server_url_label.pack(pady=10)

        # QR Code Display
        self.qr_label = ctk.CTkLabel(self.postman_frame, text="", image=None)
        self.qr_label.pack(pady=15)

        self.transfer_toggle = ctk.CTkButton(
            self.postman_frame, text="OPEN STATION", 
            height=60, font=ctk.CTkFont(size=18, weight="bold", family="Century Schoolbook"),
            command=self.toggle_http_server,
            fg_color="#5A2E2A", hover_color="#7A3E38", text_color=PANEL_TEXT
        )
        self.transfer_toggle.pack(pady=(15, 35), padx=60, fill="x")

        # Server variables
        self.http_server = None
        self.server_thread = None

    def toggle_http_server(self):
        if self.http_server:
            self.stop_http_server()
        else:
            self.start_http_server()


    def create_flask_app(self):
        app = Flask(__name__)
        
        HTML_TEMPLATE = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Postman - Wireless Rig Transfer</title>
            <style>
                body { font-family: sans-serif; background: #3B1C1A; color: #F4EBD9; padding: 20px; text-align: center; }
                .card { background: #D1A95F; color: #2A1A10; padding: 20px; border-radius: 12px; margin-bottom: 20px; max-width: 500px; margin-left: auto; margin-right: auto; }
                input[type="file"] { margin: 10px 0; width: 100%; box-sizing: border-box; }
                .btn { background: #3B1C1A; color: white; padding: 12px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin-top: 10px; border: none; cursor: pointer; }
                ul { list-style: none; padding: 0; }
                li { margin: 10px 0; background: #E5C78F; padding: 15px; border-radius: 8px; border: 1px solid #8B6D3B; display: flex; justify-content: space-between; align-items: center; }
                a { color: #3B1C1A; font-weight: bold; text-decoration: none; }
                .success { color: #228B22; font-weight: bold; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>THE POSTMAN</h1>
            <div class="card">
                <h3>Upload Rig / IR</h3>
                <p style="font-size: 0.9em;">Files will be available for 'The Baker'.</p>
                {% if msg %} <div class="success">{{ msg }}</div> {% endif %}
                <form method="post" action="/upload" enctype="multipart/form-data">
                    <select name="category" style="width: 100%; margin-bottom: 10px; padding: 8px; border-radius: 4px; background: #F4EBD9; color: #2A1A10; border: 1px solid #8B6D3B;">
                        <option value="Full_Rigs">Full Rig (.nam)</option>
                        <option value="DI_Amps">DI Amp Model (.nam)</option>
                        <option value="Cabinet_IRs">Cabinet IR (.wav)</option>
                    </select>
                    <input type="file" name="file"><br>
                    <button type="submit" class="btn" style="width: 100%;">UPLOAD TO LIBRARY</button>
                </form>
            </div>
            <div class="card">
                <h3>Your Library</h3>
                {% for category, file_list in categorized_files.items() %}
                    <h4 style="text-align: left; color: #5A2E2A; margin-bottom: 5px;">{{ category }}</h4>
                    <ul>
                        {% for file in file_list %}
                            <li>
                                <span style="word-break: break-all; text-align: left; padding-right: 10px; font-size: 0.9em;">{{ file }}</span>
                                <div style="display: flex; gap: 5px;">
                                    <a href="/download/{{ category }}/{{ file }}" class="btn" style="padding: 8px 15px;">GET</a>
                                    <form action="/delete/{{ category }}/{{ file }}" method="POST" style="margin: 0; padding: 0;">
                                        <button type="submit" class="btn" style="padding: 8px 12px; background: #8d1f1f;" onclick="return confirm('Delete {{ file }}?')">X</button>
                                    </form>
                                </div>
                            </li>
                        {% else %}
                            <li style="color: #888; font-style: italic; justify-content: center;">Empty</li>
                        {% endfor %}
                    </ul>
                {% endfor %}
            </div>
        </body>
        </html>
        """

        @app.route('/')
        def index():
            msg = request.args.get('msg')
            categorized_files = {
                "Full_Rigs": [f for f in os.listdir(RIGS_DIR) if os.path.isfile(os.path.join(RIGS_DIR, f)) and f.endswith('.nam')],
                "DI_Amps": [f for f in os.listdir(DI_DIR) if os.path.isfile(os.path.join(DI_DIR, f)) and f.endswith('.nam')],
                "Cabinet_IRs": [f for f in os.listdir(IR_DIR) if os.path.isfile(os.path.join(IR_DIR, f)) and f.endswith('.wav')],
                "Baked_Rigs": [f for f in os.listdir(BAKED_DIR) if f.endswith('.nam')] if os.path.exists(BAKED_DIR) else []
            }
            return render_template_string(HTML_TEMPLATE, categorized_files=categorized_files, msg=msg)

        @app.route('/upload', methods=['POST'])
        def upload_file():
            if 'file' not in request.files:
                return "No file part", 400
            file = request.files['file']
            if file.filename == '':
                return "No selected file", 400
            
            category = request.form.get('category', 'Full_Rigs')
            folder_map = {
                "Full_Rigs": RIGS_DIR,
                "DI_Amps": DI_DIR,
                "Cabinet_IRs": IR_DIR
            }
            target_dir = folder_map.get(category, RIGS_DIR)
            
            # File extension validation
            ext = file.filename.lower()
            if category in ["Full_Rigs", "DI_Amps"] and not ext.endswith('.nam'):
                return "<html><script>window.location.href='/?msg=Error: Must be a .nam file';</script></html>"
            if category == "Cabinet_IRs" and not ext.endswith('.wav'):
                return "<html><script>window.location.href='/?msg=Error: Must be a .wav file';</script></html>"

            filename = "".join([c for c in file.filename if c.isalnum() or c in ('.', '_', '-')])
            save_path = os.path.join(target_dir, filename)
            file.save(save_path)
            return f"<html><script>window.location.href='/?msg=Successfully Uploaded {filename}';</script></html>"

        @app.route('/download/<folder>/<filename>')
        def download_file(folder, filename):
            # Strict mapping to prevent directory traversal
            folder_map = {
                "Full_Rigs": RIGS_DIR,
                "DI_Amps": DI_DIR,
                "Cabinet_IRs": IR_DIR,
                "Baked_Rigs": BAKED_DIR
            }
            
            if folder not in folder_map:
                return "Invalid directory", 403
                
            target_dir = folder_map[folder]
            
            # Strict extension check based on folder
            if folder in ["Full_Rigs", "DI_Amps", "Baked_Rigs"] and not filename.endswith('.nam'):
                return "Unauthorized file type", 403
            if folder == "Cabinet_IRs" and not filename.endswith('.wav'):
                return "Unauthorized file type", 403
                
            return send_from_directory(target_dir, filename)

        @app.route('/delete/<folder>/<filename>', methods=['POST'])
        def delete_file(folder, filename):
            folder_map = {
                "Full_Rigs": RIGS_DIR,
                "DI_Amps": DI_DIR,
                "Cabinet_IRs": IR_DIR,
                "Baked_Rigs": BAKED_DIR
            }
            if folder not in folder_map:
                return "Invalid directory", 403
                
            target_dir = folder_map[folder]
            file_path = os.path.join(target_dir, filename)
            
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    return f"<html><script>window.location.href='/?msg=Deleted {filename}';</script></html>"
                else:
                    return f"<html><script>window.location.href='/?msg=File not found';</script></html>"
            except Exception as e:
                return f"<html><script>window.location.href='/?msg=Error deleting file';</script></html>"


        return app

    def start_http_server(self):
        try:
            local_ip = self.get_local_ip()
            port = 8000
            
            app = self.create_flask_app()
            self.http_server = make_server("", port, app, threaded=True)
            
            def serve():
                self.http_server.serve_forever()

            self.server_thread = threading.Thread(target=serve, daemon=True)
            self.server_thread.start()

            self.server_status_label.configure(text="POSTAL STATUS: Online & Listening", text_color="#44ff44")
            server_url = f"http://{local_ip}:{port}"
            self.server_url_label.configure(text=server_url)
            
            # Generate QR Code
            qr = qrcode.QRCode(version=1, box_size=5, border=2)
            qr.add_data(server_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to CTkImage for High-DPI scaling
            self.qr_img = ctk.CTkImage(light_image=img, dark_image=img, size=(180, 180))
            self.qr_label.configure(image=self.qr_img, text="")
            
            self.transfer_toggle.configure(text="STOP WI-FI TRANSFER", fg_color="#8d1f1f", hover_color="#ba2b2b")
            self.update_status(f"Postman active at {server_url}", "#44ff44")

        except Exception as e:
            self.update_status(f"Server Error: {str(e)}", "#ff4d4d")

    def stop_http_server(self):
        if self.http_server:
            # shutdown() is part of the Werkzeug server object
            threading.Thread(target=self.http_server.shutdown).start()
            self.http_server = None
            self.server_thread = None
            
            self.server_status_label.configure(text="POSTAL STATUS: Offline", text_color="#ff4d4d")
            self.server_url_label.configure(text="URL: ---")
            self.qr_label.configure(image=None, text="")
            self.qr_img = None
            self.transfer_toggle.configure(text="START WI-FI TRANSFER", fg_color="#1f538d", hover_color="#2b71ba")
            self.update_status("Postman station closed.", "#888888")

    def setup_baker_ui(self):
        # Description (Fixed transparency and wraplength)
        self.baker_desc = ctk.CTkLabel(
            self.baker_tab, 
            text="Simulate professional re-amping. Process a DI NAM model through a Cabinet IR and bake a new Full Rig.",
            wraplength=550, font=ctk.CTkFont(size=13, slant="italic"), fg_color="transparent", text_color=TWEED_TEXT
        )
        self.baker_desc.pack(pady=(10, 15))

        # Selection Frame (Oxblood Faceplate)
        self.selection_frame = ctk.CTkFrame(self.baker_tab, corner_radius=15, fg_color=OXBLOOD_PANEL, border_width=4, border_color="#222222")
        self.selection_frame.pack(padx=20, pady=5, fill="both", expand=True)

        # Step 1: DI Section
        di_frame = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        di_frame.pack(fill="x", padx=30, pady=(20, 10))
        ctk.CTkLabel(di_frame, text="Step 1: Select DI Model (.nam)", font=ctk.CTkFont(weight="bold", size=15), text_color=PANEL_TEXT).pack(anchor="w")
        
        di_inner = ctk.CTkFrame(di_frame, fg_color="transparent")
        di_inner.pack(fill="x", pady=(5, 0))
        ctk.CTkButton(di_inner, text="Choose DI", width=120, command=self.choose_baker_di, fg_color="#5A2E2A", hover_color="#7A3E38", text_color=PANEL_TEXT).pack(side="left")
        ctk.CTkLabel(di_inner, textvariable=self.baker_di_display, font=ctk.CTkFont(size=13, family="Courier"), text_color="#A8A8A8", anchor="w").pack(side="left", padx=15, fill="x", expand=True)

        # Step 2: IR Section
        ir_frame = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        ir_frame.pack(fill="x", padx=30, pady=10)
        ctk.CTkLabel(ir_frame, text="Step 2: Select Cabinet IR (.wav)", font=ctk.CTkFont(weight="bold", size=15), text_color=PANEL_TEXT).pack(anchor="w")
        
        ir_inner = ctk.CTkFrame(ir_frame, fg_color="transparent")
        ir_inner.pack(fill="x", pady=(5, 0))
        ctk.CTkButton(ir_inner, text="Choose IR", width=120, command=self.choose_baker_ir, fg_color="#5A2E2A", hover_color="#7A3E38", text_color=PANEL_TEXT).pack(side="left")
        ctk.CTkLabel(ir_inner, textvariable=self.baker_ir_display, font=ctk.CTkFont(size=13, family="Courier"), text_color="#A8A8A8", anchor="w").pack(side="left", padx=15, fill="x", expand=True)

        # Quality Selection
        quality_frame = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        quality_frame.pack(fill="x", padx=30, pady=10)
        ctk.CTkLabel(quality_frame, text="Step 3: Bake Quality", font=ctk.CTkFont(weight="bold", size=15), text_color=PANEL_TEXT).pack(anchor="w")
        
        quality_inner = ctk.CTkFrame(quality_frame, fg_color="transparent")
        quality_inner.pack(anchor="w", pady=(5, 0))
        
        ctk.CTkRadioButton(quality_inner, text="Quick Toast (15 Epochs)", variable=self.baker_epochs, value=15, 
                           fg_color="#8d1f1f", hover_color="#ba2b2b", text_color=PANEL_TEXT).pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(quality_inner, text="Full Roast (99 Epochs)", variable=self.baker_epochs, value=99, 
                           fg_color="#8d1f1f", hover_color="#ba2b2b", text_color=PANEL_TEXT).pack(side="left")

        # Action Area (Button + Tube)
        action_frame = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        action_frame.pack(fill="x", padx=30, pady=(10, 30))
        
        self.baker_tube = VacuumTube(action_frame, bg=OXBLOOD_PANEL)
        self.baker_tube.pack(side="left", padx=(0, 20))
        
        self.bake_button = ctk.CTkButton(
            action_frame, text="BAKE NEW FULL RIG", 
            font=ctk.CTkFont(size=20, weight="bold", family="Century Schoolbook"), 
            height=65, fg_color="#8d1f1f", hover_color="#ba2b2b", text_color=PANEL_TEXT,
            command=self.start_baking
        )
        self.bake_button.pack(side="left", fill="x", expand=True)

    # --- Assets Management ---
    def ensure_assets(self):
        if os.path.exists(INPUT_WAV_PATH):
            return

        print("DEBUG: input.wav missing. Attempting silent download...")
        self.update_status("Downloading training assets...", "#ffcc00")
        
        def download_task():
            try:
                r = requests.get(INPUT_WAV_URL, stream=True, timeout=30)
                r.raise_for_status()
                with open(INPUT_WAV_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                self.after(0, lambda: self.update_status("Assets ready.", "#44ff44"))
            except Exception as e:
                print(f"DEBUG: Asset download failed - {str(e)}")
                self.after(0, lambda: self.update_status("Missing input.wav. Please add to /assets folder.", "#ff4d4d"))

        threading.Thread(target=download_task, daemon=True).start()

    # --- Baker UI Logic ---
    def choose_baker_di(self):
        path = filedialog.askopenfilename(
            initialdir=DI_DIR,
            title="Select DI Model (.nam)",
            filetypes=[("NAM Files", "*.nam")]
        )
        if path:
            self.baker_di_path.set(path)
            if hasattr(self, 'baker_di_display'):
                self.baker_di_display.set(os.path.basename(path))

    def choose_baker_ir(self):
        path = filedialog.askopenfilename(
            initialdir=IR_DIR,
            title="Select Cabinet IR (.wav)",
            filetypes=[("Wav Files", "*.wav")]
        )
        if path:
            self.baker_ir_path.set(path)
            if hasattr(self, 'baker_ir_display'):
                self.baker_ir_display.set(os.path.basename(path))

    def start_baking(self):
        di = self.baker_di_path.get()
        ir = self.baker_ir_path.get()
        
        if "No" in di or "No" in ir:
            self.update_status("Error: Select DI and IR first.", "#ff4d4d")
            return
            
        if not os.path.exists(INPUT_WAV_PATH):
            self.update_status("Error: input.wav missing from assets.", "#ff4d4d")
            return

        self.update_status("Baking... This will take a few minutes.", "#ffcc00")
        self.bake_button.configure(state="disabled", text="BAKING...")
        self.baker_tube.start_glow()
        
        # Prepare output dir
        base_name = os.path.basename(di).replace(".nam", "")
        ir_name = os.path.basename(ir).replace(".wav", "")
        model_name = f"{base_name}_{ir_name}_Baked"
        output_dir = BAKED_DIR

        # Multiprocessing Queue for status updates
        self.baker_queue = multiprocessing.Queue()
        
        # Start the Baker Process
        self.baker_process = multiprocessing.Process(
            target=bake_worker, 
            args=(di, ir, INPUT_WAV_PATH, output_dir, model_name, self.baker_queue, self.baker_epochs.get())
        )
        self.baker_process.start()
        
        # Start polling for status
        self.poll_baker_queue()

    def poll_baker_queue(self):
        try:
            while True:
                msg_type, content = self.baker_queue.get_nowait()
                if msg_type == "status":
                    self.update_status(content, "#ffcc00")
                elif msg_type == "progress":
                    # Smoothly update the Tkinter progress bar
                    self.progress_bar.set(content)
                elif msg_type == "success":
                    self.update_status(content, "#44ff44")
                    self.bake_button.configure(state="normal", text="BAKE NEW FULL RIG")
                    self.baker_tube.stop_glow()
                    return
                elif msg_type == "error":
                    self.update_status(f"Baker Error: {content}", "#ff4d4d")
                    self.bake_button.configure(state="normal", text="BAKE NEW FULL RIG")
                    self.baker_tube.stop_glow()
                    return
        except:
            pass
        
        # Keep polling if process is alive
        if hasattr(self, "baker_process") and self.baker_process.is_alive():
            self.after(200, self.poll_baker_queue)
        else:
            self.bake_button.configure(state="normal")

    # --- Tone3000 Search Logic ---
    def start_search_thread(self):
        query = self.search_query.get()
        print(f"DEBUG: Search Button Clicked. Query: '{query}'")
        
        if not query.strip():
            self.update_status("Error: Enter search keywords.", "#ff4d4d")
            return

        self.search_button.configure(state="disabled")
        self.update_status("Authenticating...", "#ffcc00")
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            if widget != self.results_header_row:
                widget.destroy()

        threading.Thread(target=self.search_tone3000_task, args=(query,), daemon=True).start()

    def search_tone3000_task(self, query):
        try:
            # Phase 1: Cached Authentication
            if not self.access_token:
                self.after(0, lambda: self.update_status("Authenticating with Tone3000...", "#ffcc00"))
                
                auth_payload = {"api_key": TONE3000_API_KEY}
                auth_response = requests.post(API_AUTH_URL, json=auth_payload, timeout=10)
                
                if auth_response.status_code != 200:
                    self.after(0, lambda: self.update_status("Authentication Failed: Check API Key", "#ff4d4d"))
                    return
                
                self.access_token = auth_response.json().get("access_token")
                if not self.access_token:
                    self.after(0, lambda: self.update_status("Auth Failed: No token received", "#ff4d4d"))
                    return

            # Phase 2: GET Search Request
            selected_type = self.gear_filter.get()
            self.after(0, lambda: self.update_status(f"Searching {selected_type}s...", "#ffcc00"))
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # Step 1: UI Dropdown Mapping to API query parameters
            # Map: "Full Rig" -> gear=full-rig, "DI / Amp" -> gear=amp_pedal, "IR (Cab)" -> gear=ir
            api_gear_map = {
                "Full Rig": "full-rig",
                "DI / Amp": "amp_pedal",
                "IR (Cab)": "ir"
            }
            
            search_params = {
                "query": query,
                "gear": api_gear_map.get(selected_type, "full-rig"),
                "page_size": 100  # Ensure we retrieve a large batch of results
            }

            response = requests.get(API_SEARCH_URL, params=search_params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Extract raw results from various possible JSON structures
            raw_results = []
            if isinstance(data, list):
                raw_results = data
            elif isinstance(data, dict):
                raw_results = data.get("data", data.get("results", data.get("tones", data.get("items", []))))
            
            # Debug Step: Output raw JSON of first item
            if raw_results:
                print(f"DEBUG RAW TONE FULL: {json.dumps(raw_results[0], indent=2)}")
            else:
                print("DEBUG: raw_results is EMPTY")

            # Step 2: Simplified Metadata Filtering
            filtered_results = []
            for tone in raw_results:
                if not isinstance(tone, dict): continue
                platform_val = str(tone.get("platform", "")).lower()
                gear_val = str(tone.get("gear", "")).lower()
                
                if selected_type in ["Full Rig", "DI / Amp"]:
                    if platform_val == "nam" or any(x in gear_val for x in ["amp", "pedal", "full-rig"]):
                        filtered_results.append(tone)
                elif selected_type == "IR (Cab)":
                    if platform_val in ["ir", "wav"] or "ir" in gear_val:
                        filtered_results.append(tone)

            print(f"DEBUG: Processed {len(filtered_results)} valid filtered results.")
            self.after(0, lambda: self.display_search_results(filtered_results))
            
        except requests.exceptions.RequestException as e:
            msg = "Error: Connection failed."
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 401:
                    self.access_token = None
                    msg = "Authentication Failed: Check API Key"
            self.after(0, lambda: self.update_status(msg, "#ff4d4d"))
        except Exception as e:
            self.after(0, lambda: self.update_status(f"Parse Error: {str(e)[:40]}", "#ff4d4d"))
        finally:
            self.after(0, lambda: self.search_button.configure(state="normal"))

    def display_search_results(self, results):
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Re-add header
        header_row = ctk.CTkFrame(self.results_frame, fg_color="#B38D4F", height=35)
        header_row.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(header_row, text="DATABASE RESULTS", text_color="#F4EBD9", font=ctk.CTkFont(size=11, weight="bold")).pack(padx=15, side="left")

        if not results:
            self.update_status("No results found.", "#ff4d4d")
            return

        self.update_status(f"Found {len(results)} rigs.", "#888888")
        
        # Limit to top 25 results for UI performance
        for item in results[:25]:
            if not isinstance(item, dict): continue
            
            # Main result container
            card = ctk.CTkFrame(
                self.results_frame, 
                fg_color="#F4EBD9", 
                corner_radius=10, 
                border_width=2, 
                border_color="#8B6D3B"
            )
            card.pack(fill="x", pady=6, padx=8)
            
            info_frame = ctk.CTkFrame(card, fg_color="transparent")
            info_frame.pack(fill="x", padx=12, pady=(10, 5))
            
            name = item.get("title", item.get("name", "Unknown"))
            gear_type = item.get("gear", item.get("type", "N/A")).capitalize()
            dls = item.get("downloads_count", item.get("downloads", 0))
            tone_id = item.get("id")
            
            ctk.CTkLabel(info_frame, text=name, text_color="#2A1A10", font=ctk.CTkFont(weight="bold", size=13), anchor="w").pack(side="left")
            ctk.CTkLabel(info_frame, text=f"{gear_type} | {dls} DLs", font=ctk.CTkFont(size=10), text_color="#5A2E2A").pack(side="right")
            
            # Web URL Label (Using ID-based pattern for reliability)
            reliable_url = f"https://www.tone3000.com/tones/{tone_id}" if tone_id else item.get("url", "#")
            url_label = ctk.CTkLabel(
                card, text=f"ID: {tone_id} | {reliable_url}", 
                text_color="#1f538d", cursor="hand2",
                font=ctk.CTkFont(size=9, underline=True),
                anchor="w"
            )
            url_label.pack(fill="x", padx=12, pady=(0, 5))
            url_label.bind("<Button-1>", lambda e, url=reliable_url: webbrowser.open(url))
            
            if tone_id:
                btn_frame = ctk.CTkFrame(card, fg_color="transparent")
                btn_frame.pack(fill="x", padx=12, pady=(0, 10))
                
                # Conversion Button (The Butcher)
                ctk.CTkButton(
                    btn_frame, text="SCALER", width=85, height=26, font=ctk.CTkFont(size=10, weight="bold"),
                    command=lambda tid=tone_id, n=name: self.start_download_thread(tid, n, "butcher"),
                    fg_color="#3B1C1A", hover_color="#5A2E2A", text_color="#F4EBD9"
                ).pack(side="left", padx=(0, 5))
                
                # Baker DI Button
                ctk.CTkButton(
                    btn_frame, text="BAKE DI", width=85, height=26, font=ctk.CTkFont(size=10, weight="bold"),
                    command=lambda tid=tone_id, n=name: self.start_download_thread(tid, n, "baker_di"),
                    fg_color="#5A2E2A", hover_color="#7A3E38", text_color="#F4EBD9"
                ).pack(side="left", padx=5)

                # Baker IR Button
                ctk.CTkButton(
                    btn_frame, text="BAKE IR", width=85, height=26, font=ctk.CTkFont(size=10, weight="bold"),
                    command=lambda tid=tone_id, n=name: self.start_download_thread(tid, n, "baker_ir"),
                    fg_color="#8B6D3B", hover_color="#A88B59", text_color="#F4EBD9"
                ).pack(side="left", padx=5)

                # Web Browser Demo Button
                ctk.CTkButton(
                    btn_frame, text="DEMO", width=60, height=26, font=ctk.CTkFont(size=10, weight="bold"),
                    command=lambda tid=tone_id, url=reliable_url: self.open_in_browser(tid, url),
                    fg_color="#1f538d", hover_color="#2b71ba", text_color="#F4EBD9"
                ).pack(side="left", padx=5)

    def open_in_browser(self, tone_id, direct_url=None):
        """Opens the Tone3000 web player for the selected rig."""
        try:
            # Prioritize the ID-based URL as the API 'url' slug often 404s
            target_url = f"https://www.tone3000.com/tones/{tone_id}"
            
            if not tone_id:
                if direct_url:
                    target_url = direct_url
                else:
                    self.update_status("Error: Link not found.", "#ff4d4d")
                    return

            webbrowser.open(target_url)
            self.update_status("Opening web demo...", "#44ff44")
        except Exception as e:
            self.update_status(f"Error: Could not open browser.", "#ff4d4d")

    # --- Download Logic ---
    def start_download_thread(self, tone_id, name, target):
        print(f"DEBUG: Starting download for {name} (Target: {target})")
        self.update_status(f"Fetching models: {name}", "#ffcc00")
        self.progress_bar.set(0)
        threading.Thread(target=self.download_task, args=(tone_id, name, target), daemon=True).start()

    def download_task(self, tone_id, name, target):
        try:
            # Step 1: Authentication
            if not self.access_token:
                auth_response = requests.post(API_AUTH_URL, json={"api_key": TONE3000_API_KEY}, timeout=10)
                if auth_response.status_code == 200:
                    self.access_token = auth_response.json().get("access_token")
                else:
                    self.after(0, lambda: self.update_status("Auth Failed", "#ff4d4d"))
                    return

            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            # Step 2: Get Model List for the Tone
            models_url = f"{API_BASE_URL}/models"
            params = {"tone_id": tone_id}
            
            models_response = requests.get(models_url, params=params, headers=headers, timeout=15)
            models_response.raise_for_status()
            models_data = models_response.json()
            
            # Handle results
            models = []
            if isinstance(models_data, dict):
                models = models_data.get("data", [])
            elif isinstance(models_data, list):
                models = models_data

            if not models:
                self.after(0, lambda: self.update_status("Error: No models available.", "#ff4d4d"))
                return

            # Step 3: Select the best model (prefer 'standard' size)
            # Security: Ensure we are dealing with dicts
            selected_model = None
            for m in models:
                if isinstance(m, dict):
                    if m.get("size") == "standard" or selected_model is None:
                        selected_model = m
                        if m.get("size") == "standard": break
            
            if not selected_model:
                self.after(0, lambda: self.update_status("Error: model discovery failed.", "#ff4d4d"))
                return

            model_url = selected_model.get("model_url")
            if not model_url:
                self.after(0, lambda: self.update_status("Error: Corrupt model data.", "#ff4d4d"))
                return

            self.after(0, lambda: self.update_status(f"Downloading {name}...", "#ffcc00"))

            # Step 4: Actual File Download
            response = requests.get(model_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()

            # Dynamic Extension Parsing (Fix for Baker extension bug)
            url_lower = model_url.lower().split('?')[0]
            ext = ".nam"
            if url_lower.endswith(".wav"):
                ext = ".wav"
            elif ".wav" in url_lower:
                ext = ".wav"
                
            safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '.', '_')]).rstrip()
            filename = f"{safe_name}{ext}"
            if target == "butcher":
                target_dir = RIGS_DIR
            elif target == "baker_di":
                target_dir = DI_DIR
            elif target == "baker_ir":
                target_dir = IR_DIR
            else:
                target_dir = DOWNLOADS_DIR
                
            save_path = os.path.join(target_dir, filename)

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = downloaded_size / total_size
                            self.after(0, lambda p=progress: self.progress_bar.set(p))

            self.after(0, lambda: self.progress_bar.set(1.0))
            self.after(0, lambda: self.finish_download(save_path, target))
            
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self.update_status(f"Download Error: {err_msg}", "#ff4d4d"))
            self.after(0, lambda: self.progress_bar.set(0))

    def finish_download(self, file_path, target):
        if target == "butcher":
            self.selected_file_path.set(file_path)
            self.tabview.set("The Butcher")
        elif target == "baker_di":
            self.baker_di_path.set(file_path)
            self.tabview.set("The Baker Engine")
        elif target == "baker_ir":
            self.baker_ir_path.set(file_path)
            self.tabview.set("The Baker Engine")
            
        self.update_status(f"Loaded {target}: {os.path.basename(file_path)}", "#44ff44")


    def update_gain_and_warning(self, value):
        self.gain_db.set(value)
        if value >= 12:
            self.high_gain_warning.pack(pady=(0, 10))
        else:
            self.high_gain_warning.pack_forget()

    # --- Core Processor Logic ---
    def select_file(self):
        file_path = filedialog.askopenfilename(
            initialdir=RIGS_DIR,
            title="Select .nam File",
            filetypes=[("Neural Amp Modeler Files", "*.nam"), ("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            self.selected_file_path.set(file_path)
            self.update_status("File loaded.", "#888888")

    def update_status(self, text, color="#888888"):
        self.status_label.configure(text=text, text_color=color)

    def start_forging(self):
        file_path = self.selected_file_path.get()
        if file_path == "No file selected" or not os.path.exists(file_path):
            self.update_status("Error: Please select/download a file first.", "#ff4d4d")
            return

        gain = self.gain_db.get()
        self.update_status(f"Forging... Applying +{gain}dB", "#ffcc00")
        self.forge_button.configure(state="disabled")
        self.select_button.configure(state="disabled")
        self.progress_bar.set(0)

        threading.Thread(target=self.forge_rig_task, args=(file_path, gain), daemon=True).start()

    def forge_rig_task(self, file_path, gain_db):
        try:
            multiplier = math.pow(10, gain_db / 20.0)
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Safety check: Ensure data is a dictionary
            if not isinstance(data, dict):
                self.after(0, lambda: self.finish_forging("Error: Invalid NAM file format.", "#ff4d4d"))
                return

            if "weights" not in data:
                self.after(0, lambda: self.finish_forging("Error: Missing 'weights' key in NAM file.", "#ff4d4d"))
                return

            self.after(0, lambda: self.progress_bar.set(0.1))

            weights = data["weights"]
            if not isinstance(weights, list):
                self.after(0, lambda: self.finish_forging("Error: Invalid weights data in NAM file.", "#ff4d4d"))
                return

            total_weights = len(weights)
            modified_weights = []
            
            # Process in chunks to show progress
            chunk_size = max(1, total_weights // 100)
            for i in range(0, total_weights, chunk_size):
                chunk = weights[i:i + chunk_size]
                modified_weights.extend([float(w) * multiplier for w in chunk])
                progress = 0.1 + (0.7 * (len(modified_weights) / total_weights))
                self.after(0, lambda p=progress: self.progress_bar.set(p))
            
            data["weights"] = modified_weights
            self.after(0, lambda: self.progress_bar.set(0.85))

            # Metadata Injection
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["output_level_dbu"] = gain_db

            base_name, extension = os.path.splitext(file_path)
            new_file_path = f"{base_name}_+{gain_db}dB{extension}"

            with open(new_file_path, 'w') as f:
                json.dump(data, f, separators=(',', ':'))

            self.after(0, lambda: self.progress_bar.set(1.0))
            success_msg = f"Success! Forge complete: {os.path.basename(new_file_path)}"
            self.after(0, lambda: self.finish_forging(success_msg, "#44ff44"))

        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self.finish_forging(f"Forge Error: {err_msg}", "#ff4d4d"))

    def finish_forging(self, message, color):
        self.update_status(message, color)
        self.forge_button.configure(state="normal")
        self.select_button.configure(state="normal")

if __name__ == "__main__":
    app = PMNamConverter()
    app.mainloop()
