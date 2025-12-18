import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import queue
import time
import PySpin
import re

# PaletteUtil import with error handling
try:
    from palette_util import PaletteUtil
    PALETTE_AVAILABLE = True
except Exception as e:
    print(f"Warning: PaletteUtil not available: {e}")
    PALETTE_AVAILABLE = False
    class PaletteUtil:  # Dummy class
        def __init__(self): pass
        def apply_color_palette(self, *args, **kwargs): return None

class FLIR_Demo_App:
    def __init__(self, root):
        self.root = root
        self.root.title("FLIR PySpin Learning Dashboard")
        self.root.geometry("1400x950")
        
        # 상태 변수
        self.cam = None # ir_camera 객체 (사용 안함, PySpin 직접 사용)
        self.image_queue = queue.Queue(maxsize=1) 
        self.is_streaming = False
        self.palette_util = PaletteUtil()
        self.flir_info = {'Root': {}} # 더미 정보
        self.system = None
        self.cam_list = None
        self.spin_cam = None 
        self.nodemap = None
        self.sNodemap = None # Stream Nodemap
        self.current_ir_mode = "Radiometric"  # Track IR mode for temp conversion
        self.display_w = 0
        self.display_h = 0
        
        # 레이아웃 구성
        self.setup_ui()
        
    def setup_ui(self):
        # --- Theme Colors ---
        self.colors = {
            "bg_main": "#2E2E2E",      # Root background
            "bg_panel": "#3C3C3C",     # Panel background
            "fg_text": "#FFFFFF",      # Default text
            "accent": "#007ACC",       # Accent (Blue)
            "code_bg": "#1E1E1E",      # Code editor bg
            "code_fg": "#D4D4D4",      # Code text
            "log_bg": "#1E1E1E",       # Log bg
            "log_fg": "#00FF00",       # Log text (Green)
            "header_bg": "#505050",    # Header bg
            "header_fg": "#FFFFFF"     # Header text
        }
        
        self.root.configure(bg=self.colors["bg_main"])
        
        # --- Style Configuration ---
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("TFrame", background=self.colors["bg_main"])
        style.configure("TLabelframe", background=self.colors["bg_main"], foreground=self.colors["fg_text"])
        style.configure("TLabelframe.Label", background=self.colors["bg_main"], foreground=self.colors["fg_text"], font=("Segoe UI", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 9), background="#555555", foreground="white", borderwidth=1)
        style.map("TButton", background=[("active", self.colors["accent"])])
        style.configure("TRadiobutton", background=self.colors["bg_panel"], foreground=self.colors["fg_text"], font=("Segoe UI", 9))
        style.map("TRadiobutton", background=[("active", self.colors["bg_panel"])])

        # --- Main Layout Structure ---
        main_frame = tk.Frame(self.root, bg=self.colors["bg_main"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. Main Horizontal Split (Left vs Right)
        main_paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, bg=self.colors["bg_main"], sashwidth=6, sashrelief=tk.FLAT)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # [Left Column] Control Panel
        ctrl_frame_container = ttk.LabelFrame(main_paned, text="Control Panel")
        main_paned.add(ctrl_frame_container, width=320, minsize=300, stretch="never")
        
        canvas = tk.Canvas(ctrl_frame_container, bg=self.colors["bg_panel"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(ctrl_frame_container, orient="vertical", command=canvas.yview)
        menu_scroll_frame = tk.Frame(canvas, bg=self.colors["bg_panel"])
        
        menu_scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=menu_scroll_frame, anchor="nw")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        scrollbar.pack(side="right", fill="y")
        
        self.populate_menu(menu_scroll_frame)

        # [Right Area] Vertical Split (Top: Cam+Log / Bottom: Code)
        right_area_paned = tk.PanedWindow(main_paned, orient=tk.VERTICAL, bg=self.colors["bg_main"], sashwidth=6, sashrelief=tk.FLAT)
        main_paned.add(right_area_paned, stretch="always")

        # -> [Right-Top] Horizontal Split (Camera | Log)
        top_h_paned = tk.PanedWindow(right_area_paned, orient=tk.HORIZONTAL, bg=self.colors["bg_main"], sashwidth=6, sashrelief=tk.FLAT)
        right_area_paned.add(top_h_paned, stretch="always", height=450)
        
        # ---> Camera View
        viewer_frame = ttk.LabelFrame(top_h_paned, text="Live Camera View")
        top_h_paned.add(viewer_frame, stretch="always", minsize=400)

        # Stats Label Overlay (Manual Place inside frame)
        self.stats_label = tk.Label(viewer_frame, text="Max: -- Min: -- Avg: --", 
                                    bg=self.colors["bg_main"], fg="#00FF00", font=("Consolas", 10, "bold"))
        self.stats_label.place(x=120, y=0) 
        
        self.image_label = tk.Label(
            viewer_frame, 
            text="No Signal\nPlease Connect Camera", 
            bg="black", fg="#555", 
            font=("Segoe UI", 16)
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=(20, 5)) # Add top padding for stats space
        self.image_label.bind("<Motion>", self.callback_mouse_move)

        # Tooltip Label (Floating Helper)
        self.tooltip = tk.Label(self.root, text="", bg="#ffffe0", fg="black", relief="solid", borderwidth=1, font=("Segoe UI", 9))
        
        # ---> System Log
        log_frame = ttk.LabelFrame(top_h_paned, text="System Log")
        top_h_paned.add(log_frame, stretch="always", minsize=300)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, 
            bg=self.colors["log_bg"], fg=self.colors["log_fg"], 
            font=("Consolas", 10), state='disabled',
            borderwidth=0, relief="flat"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # -> [Right-Bottom] Code Snippet
        code_frame = ttk.LabelFrame(right_area_paned, text="PySpin Code Snippet")
        right_area_paned.add(code_frame, stretch="always", height=350)
        
        self.code_text = scrolledtext.ScrolledText(
            code_frame, 
            bg=self.colors["code_bg"], fg=self.colors["code_fg"], 
            font=("Consolas", 12, "bold"), insertbackground="white", 
            borderwidth=0, relief="flat"
        )
        self.code_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.setup_text_tags()

    def populate_menu(self, parent):
        # 1. Device Discovery & Init
        self.create_header(parent, "1. Device Discovery & Init")
        self.create_btn_row(parent, [("1-1. Scan Cameras", self.cmd_scan_cameras)])
        
        lbl_cam = tk.Label(parent, text="Select Camera:", bg=self.colors["bg_panel"], fg="white", anchor="w")
        lbl_cam.pack(fill="x", padx=15, pady=(5, 0))
        self.combo_cameras = ttk.Combobox(parent, state="readonly")
        self.combo_cameras.pack(fill="x", padx=15, pady=2)
        
        self.create_btn_row(parent, [("1-2. Initialize Camera", self.cmd_init_camera)])
        
        # 2. Configuration
        self.create_header(parent, "2. Configuration")
        
        lbl_fmt = tk.Label(parent, text="Pixel Format:", bg=self.colors["bg_panel"], fg="white", anchor="w")
        lbl_fmt.pack(fill="x", padx=15, pady=(5, 0))
        self.combo_pixel = ttk.Combobox(parent, values=["Mono8", "Mono16"], state="readonly")
        self.combo_pixel.current(1)
        self.combo_pixel.pack(fill="x", padx=15, pady=2)
        
        lbl_ir = tk.Label(parent, text="IR Format:", bg=self.colors["bg_panel"], fg="white", anchor="w")
        lbl_ir.pack(fill="x", padx=15, pady=(5, 0))
        self.combo_ir = ttk.Combobox(parent, values=["Radiometric", "TemperatureLinear10mK", "TemperatureLinear100mK"], state="readonly")
        self.combo_ir.current(0) # Default Radiometric
        self.combo_ir.pack(fill="x", padx=15, pady=2)
        
        self.create_btn_row(parent, [("2-1. Apply Formats", self.cmd_apply_config)])
        
        # 3. Node Control Examples
        self.create_header(parent, "3. Node Control Examples")
        self.create_btn_row(parent, [("String (Model)", self.cmd_node_string), ("Integer (Width)", self.cmd_node_integer)])
        self.create_btn_row(parent, [("Float (FrameRate)", self.cmd_node_float), ("Enum (PixelFmt)", self.cmd_node_enum)])
        self.create_btn_row(parent, [("Bool (QueryCase)", self.cmd_node_bool), ("Set FPS 30Hz", self.cmd_node_command)])
        self.create_btn_row(parent, [("Cmd (NUC)", self.cmd_node_nuc), ("Cmd (AutoFocus)", self.cmd_node_autofocus)])
        
        # 4. StreamingSequence
        self.create_header(parent, "4. Streaming & Processing")
        self.create_btn_row(parent, [("Begin Acquisition", self.cmd_begin_acquisition), ("End Acquisition", self.cmd_end_acquisition)])
        self.create_btn_row(parent, [("Start Streaming", self.cmd_start_stream), ("Stop Streaming", self.cmd_stop_stream)])
        self.create_btn_row(parent, [("Snapshot", self.cmd_snapshot_process), ("Normalize", self.cmd_normalize_process)])

        # 5. Visualization Mode (Unified Palette)
        self.create_header(parent, "5. Palette Visualization")
        mode_frame = tk.Frame(parent, bg=self.colors["bg_panel"])
        mode_frame.pack(fill="x", padx=10, pady=5)
        
        lbl_pal = tk.Label(mode_frame, text="Palette:", bg=self.colors["bg_panel"], fg="white")
        lbl_pal.pack(side="left", padx=(0, 5))
        
        self.combo_palette = ttk.Combobox(mode_frame, values=["Gray", "Iron", "Rainbow", "Redgray"], state="readonly", width=15)
        self.combo_palette.current(0) # Default Gray
        self.combo_palette.pack(side="left", fill="x", expand=True)
        self.combo_palette.bind("<<ComboboxSelected>>", self.on_palette_change)

        # 6. System Management
        self.create_header(parent, "6. System Management")
        self.create_btn_row(parent, [("Disconnect", self.cmd_disconnect)])

    def setup_text_tags(self):
        self.code_text.tag_config("keyword", foreground="#C586C0") 
        self.code_text.tag_config("def_cls", foreground="#DCDCAA") 
        self.code_text.tag_config("string", foreground="#CE9178") 
        self.code_text.tag_config("comment", foreground="#6A9955") 
        self.code_text.tag_config("number", foreground="#B5CEA8") 

    # --- UI Helper Methods ---
    def create_header(self, parent, text):
        frame = tk.Frame(parent, bg=self.colors["header_bg"])
        frame.pack(fill="x", pady=(15, 5), padx=2)
        tk.Frame(frame, bg=self.colors["accent"], width=4).pack(side="left", fill="y")
        lbl = tk.Label(frame, text=f" {text}", font=("Segoe UI", 10, "bold"), fg=self.colors["header_fg"], bg=self.colors["header_bg"], anchor="w", pady=6)
        lbl.pack(side="left", fill="x", expand=True)
        
    def create_btn_row(self, parent, buttons):
        frame = tk.Frame(parent, bg=self.colors["bg_panel"])
        frame.pack(fill="x", padx=10, pady=4)

        # Configure columns to have equal weight for uniform width
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        for i, (text, cmd) in enumerate(buttons):
            btn = ttk.Button(frame, text=text, command=cmd)
            btn.grid(row=0, column=i, padx=4, sticky="ew")

    def log(self, msg):
        self.log_text.config(state='normal')
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_text.insert(tk.END, f"{timestamp}{msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        
    def show_code(self, code_str):
        self.code_text.delete(1.0, tk.END)
        self.code_text.insert(tk.END, code_str)
        self.highlight_code()

    def highlight_code(self):
        content = self.code_text.get("1.0", tk.END)
        for tag in ["keyword", "def_cls", "string", "comment", "number"]:
            self.code_text.tag_remove(tag, "1.0", tk.END)
        keywords = r"\b(import|from|class|def|return|if|else|try|except|while|for|in|True|False|None)\b"
        for match in re.finditer(keywords, content):
            self.code_text.tag_add("keyword", f"1.0+{match.start()}c", f"1.0+{match.end()}c")
        for match in re.finditer(r"#.*", content):
            self.code_text.tag_add("comment", f"1.0+{match.start()}c", f"1.0+{match.end()}c")
        for match in re.finditer(r"['\"].*?['\"]", content):
            self.code_text.tag_add("string", f"1.0+{match.start()}c", f"1.0+{match.end()}c")

    # --- Command Handlers ---
    
    def on_palette_change(self, event):
        pal_name = self.combo_palette.get()
        code = f"""import numpy as np
import cv2
from palette_util import PaletteUtil

# [Function Explanation] apply_color_palette
# This logic is implemented inside PaletteUtil.
def apply_color_palette_demo(image, palette_lut):
    # 1. Normalize (Min-Max -> 0~255)
    min_v, max_v = np.min(image), np.max(image)
    if max_v == min_v:
        norm_img = np.zeros_like(image, dtype=np.uint8)
    else:
        norm_img = ((image - min_v) / (max_v - min_v) * 255).astype(np.uint8)

    # 2. Convert to 3-Channel (BGR) for LUT
    src_bgr = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)

    # 3. Apply LUT per Channel
    # palette_lut shape: (256, 3) -> [B, G, R]
    b = cv2.LUT(src_bgr[:,:,0], palette_lut[:,0])
    g = cv2.LUT(src_bgr[:,:,1], palette_lut[:,1])
    r = cv2.LUT(src_bgr[:,:,2], palette_lut[:,2])

    return cv2.merge([b, g, r])

# --- Usage ---
pu = PaletteUtil() # Loads CSVs (Iron, Rainbow...)
if "{pal_name}" == "Gray":
    # Simple Grayscale
    norm = cv2.normalize(raw_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    result = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
else:
    # Get LUT for {pal_name}
    lut = pu.palettes["{pal_name}".upper()] 
    result = apply_color_palette_demo(raw_data, lut)

cv2.imshow("Palette Result", result)
"""
        self.show_code(code)
        self.log(f"Selected Palette: {pal_name}")
    
    # 32-bit Integer (0xAABBCCDD) -> IP String ("AA.BB.CC.DD")
    # GevDeviceIPAddress는 32비트 정수 형태로 IP를 반환합니다.
    # 사람이 읽는 IP 주소(Network Byte Order)는 Big Endian 방식이므로
    # 최상위 비트(MSB, 24~31비트)부터 8비트씩 잘라서 순서대로 연결합니다.
    def int_to_ip(self, ip_int):
        return ".".join([
                        str((ip_int >> 24) & 0xFF)
                        , str((ip_int >> 16) & 0xFF)
                        , str((ip_int >> 8) & 0xFF)
                        , str(ip_int & 0xFF)
                        ])

    def cmd_scan_cameras(self):
        """1. 카메라 스캔 및 리스트업"""
        code = """# System 인스턴스 생성 및 카메라 리스트 조회
self.system = PySpin.System.GetInstance()
self.cam_list = self.system.GetCameras()
n_cams = self.cam_list.GetSize().

# 32-bit Integer (0xAABBCCDD) -> IP String ("AA.BB.CC.DD")
# GevDeviceIPAddress는 32비트 정수 형태로 IP를 반환합니다.
# 사람이 읽는 IP 주소(Network Byte Order)는 Big Endian 방식이므로
# 최상위 비트(MSB, 24~31비트)부터 8비트씩 잘라서 순서대로 연결합니다
def int_to_ip(ip_int):
    return ".".join([str((ip_int >> 24) & 0xFF), str((ip_int >> 16) & 0xFF),
                     str((ip_int >> 8) & 0xFF), str(ip_int & 0xFF)])

for cam in self.cam_list:
    # Get TLDevice NodeMap
    nodemap_tl = cam.GetTLDeviceNodeMap()
    
    # Access Model Name, Serial & IP
    node_model = PySpin.CStringPtr(nodemap_tl.GetNode("DeviceModelName"))
    node_serial = PySpin.CStringPtr(nodemap_tl.GetNode("DeviceSerialNumber"))
    node_ip = PySpin.CIntegerPtr(nodemap_tl.GetNode("GevDeviceIPAddress"))
    
    model = node_model.GetValue()
    serial = node_serial.GetValue()
    ip = int_to_ip(node_ip.GetValue())
    
    print(f"{model} ({serial}) - IP: {ip}")
"""
        self.show_code(code)
        self.log("Scanning cameras...")
        
        try:
            self.system = PySpin.System.GetInstance()
            self.cam_list = self.system.GetCameras()
            num_cams = self.cam_list.GetSize()
            
            if num_cams == 0:
                self.log("No cameras detected.")
                self.combo_cameras['values'] = []
                return
            
            self.log(f"Cameras found: {num_cams}")
            
            cam_names = []
            for i in range(num_cams):
                cam = self.cam_list.GetByIndex(i)
                try:
                    nodemap_tl = cam.GetTLDeviceNodeMap()
                    node_model = PySpin.CStringPtr(nodemap_tl.GetNode("DeviceModelName"))
                    node_serial = PySpin.CStringPtr(nodemap_tl.GetNode("DeviceSerialNumber"))
                    
                    # IP Address Retrieval
                    ip_str = "N/A"
                    node_ip = nodemap_tl.GetNode("GevDeviceIPAddress")
                    if PySpin.IsAvailable(node_ip) and PySpin.IsReadable(node_ip):
                        ip_val = PySpin.CIntegerPtr(node_ip).GetValue()
                        ip_str = self.int_to_ip(ip_val)

                    name_str = node_model.GetValue() if PySpin.IsAvailable(node_model) else "Unknown"
                    serial_str = node_serial.GetValue() if PySpin.IsAvailable(node_serial) else "Unknown"
                    
                    label = f"[{i}] {name_str} ({serial_str}) IP:{ip_str}"
                    self.log(f"  Found: {label}")
                    cam_names.append(label)
                except Exception as e:
                    cam_names.append(f"[{i}] Info Error: {e}")
            
            self.combo_cameras['values'] = cam_names
            if cam_names:
                self.combo_cameras.current(0)
                
        except Exception as e:
            self.log(f"Scan Error: {e}")

    def cmd_init_camera(self):
        """1-2. 선택된 카메라 Init Only"""
        idx = self.combo_cameras.current()
        if idx < 0:
            self.log("Please select a camera first.")
            return
            
        code = f"""# 1. Select Camera
self.cam = self.cam_list.GetByIndex({idx})

# 2. Init
self.cam.Init()

# 3. Get NodeMaps
self.nodemap = self.cam.GetNodeMap()
self.sNodemap = self.cam.GetTLStreamNodeMap() # Stream Control
"""
        self.show_code(code)
        self.log(f"Initializing Camera [{idx}]...")
        
        try:
            self.spin_cam = self.cam_list.GetByIndex(idx)
            self.spin_cam.Init()
            self.nodemap = self.spin_cam.GetNodeMap()
            self.sNodemap = self.spin_cam.GetTLStreamNodeMap()
            self.image_queue = queue.Queue(maxsize=1)
            
            self.log("Camera Initialized. Ready for Configuration.")
            
        except Exception as e:
            self.log(f"Init Failed: {e}")

    def cmd_apply_config(self):
        """2-1. Configuration 적용 (Buffer -> Pixel -> IR)"""
        if not hasattr(self, 'nodemap') or self.nodemap is None:
            self.log("Initialize Camera first!")
            return

        pixel_fmt = self.combo_pixel.get()
        ir_fmt = self.combo_ir.get()

        code = f"""# 1. Set Stream Buffer to 'NewestOnly'
node_buffer = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
entry_newest = node_buffer.GetEntryByName('NewestOnly')
node_buffer.SetIntValue(entry_newest.GetValue())

# 2. Set Pixel Format ({pixel_fmt})
node_pix = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
entry_pix = node_pix.GetEntryByName('{pixel_fmt}')
node_pix.SetIntValue(entry_pix.GetValue())

# 3. Set IR Format ({ir_fmt})
node_ir = PySpin.CEnumerationPtr(nodemap.GetNode('IRFormat'))
entry_ir = node_ir.GetEntryByName('{ir_fmt}')
node_ir.SetIntValue(entry_ir.GetValue())
"""
        self.show_code(code)
        self.log(f"Applying Config: NewestOnly, {pixel_fmt}, {ir_fmt}...")
        
        try:
            # 1. Stream Buffer Handling Mode (NewestOnly)
            if hasattr(self, 'sNodemap') and self.sNodemap:
                try:
                    node_buffer = PySpin.CEnumerationPtr(self.sNodemap.GetNode('StreamBufferHandlingMode'))
                    if PySpin.IsAvailable(node_buffer) and PySpin.IsWritable(node_buffer):
                        entry_newest = node_buffer.GetEntryByName('NewestOnly')
                        if PySpin.IsAvailable(entry_newest) and PySpin.IsReadable(entry_newest):
                            node_buffer.SetIntValue(entry_newest.GetValue())
                            self.log("Buffer Mode set to NewestOnly")
                except Exception as e:
                    self.log(f"Buffer Mode Error: {e}")
            else:
                 self.log("Stream Nodemap unavailable.")

            # 2. Pixel Format
            try:
                node_pix_fmt = PySpin.CEnumerationPtr(self.nodemap.GetNode('PixelFormat'))
                if PySpin.IsAvailable(node_pix_fmt) and PySpin.IsWritable(node_pix_fmt):
                    entry = node_pix_fmt.GetEntryByName(pixel_fmt)
                    if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                        node_pix_fmt.SetIntValue(entry.GetValue())
                        self.log(f"PixelFormat set to {pixel_fmt}")
            except Exception as e:
                self.log(f"PixelFormat Error: {e}")

            # 3. IR Format
            try:
                node_ir_fmt = PySpin.CEnumerationPtr(self.nodemap.GetNode('IRFormat'))
                if PySpin.IsAvailable(node_ir_fmt) and PySpin.IsWritable(node_ir_fmt):
                    entry = node_ir_fmt.GetEntryByName(ir_fmt)
                    if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                        node_ir_fmt.SetIntValue(entry.GetValue())
                        self.log(f"IRFormat set to {ir_fmt}")
            except Exception as e:
                self.log(f"IRFormat Error: {e}")
                
            self.log("Configuration Applied.")
            
        except Exception as e:
            self.log(f"Config Error: {e}")

    # ... (Node Examples: String, Int, Float, Enum, Bool, Command)
    def cmd_node_string(self):
        code = """node_model = PySpin.CStringPtr(self.nodemap.GetNode('DeviceModelName'))
val = node_model.GetValue()
print(f"Model: {val}")

# [String 쓰기 예제] - 만약 쓰기 가능한 노드라면(예: DeviceUserID):
# if PySpin.IsWritable(node_model):
#     node_model.SetValue("MyNewName")
"""
        self.show_code(code)
        self.run_node_val('DeviceModelName', PySpin.CStringPtr, "Model")

    def cmd_node_integer(self):
        code = """node_width = PySpin.CIntegerPtr(self.nodemap.GetNode('Width'))
val = node_width.GetValue()
print(f"Width: {val}")

# [Integer 설정 예제] - 값을 640으로 설정
# if PySpin.IsWritable(node_width):
#     node_width.SetValue(640)
"""
        self.show_code(code)
        self.run_node_val('Width', PySpin.CIntegerPtr, "Width")

    def cmd_node_float(self):
        code = """node_fr = PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate'))
val = node_fr.GetValue()
print(f"FrameRate: {val:.2f} Hz")

# [Float 설정 예제] - 30.0 fps로 설정
# if PySpin.IsWritable(node_fr):
#     node_fr.SetValue(30.0)
"""
        self.show_code(code)
        self.run_node_val('AcquisitionFrameRate', PySpin.CFloatPtr, "Frame Rate")

    def cmd_node_enum(self):
        code = """node_pix = PySpin.CEnumerationPtr(self.nodemap.GetNode('PixelFormat'))
entry = PySpin.CEnumEntryPtr(node_pix.GetCurrentEntry())
val = entry.GetSymbolic()
print(f"PixelFormat: {val}")

# [Enum 설정 예제] - 'Mono8'로 변경
# entry_mono8 = node_pix.GetEntryByName('Mono8')
# if PySpin.IsReadable(entry_mono8):
#     node_pix.SetIntValue(entry_mono8.GetValue())
"""
        self.show_code(code)
        # Custom logic for enum
        if hasattr(self, 'nodemap'):
            try:
                node = PySpin.CEnumerationPtr(self.nodemap.GetNode('PixelFormat'))
                if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
                    entry = PySpin.CEnumEntryPtr(node.GetCurrentEntry())
                    val = entry.GetSymbolic()
                    self.log(f"[Result] PixelFormat: {val}")
            except Exception as e: self.log(f"Error: {e}")

    def cmd_node_bool(self):
        """QueryCase 3 -> QueryCaseEnable"""
        code = """# [Bool 설정 예제] - QueryCase를 3으로 설정 (사전조건)
# Integer 노드지만, 아래 Bool 노드의 동작을 바꾸기 위해 설정
node_case = PySpin.CIntegerPtr(nodemap.GetNode('QueryCase'))
node_case.SetValue(3)

# [Bool 읽기 예제]
node_en = PySpin.CBooleanPtr(nodemap.GetNode('QueryCaseEnable'))
val = node_en.GetValue()
print(f"QueryCaseEnable (Case 3): {val}")

# [Bool 설정 예제] - 직접 True/False 설정 시:
# if PySpin.IsWritable(node_en):
#     node_en.SetValue(True)
"""
        self.show_code(code)
        self.log("QueryCase 3 -> Checking Enable...")
        
        if hasattr(self, 'nodemap'):
            try:
                # 1. Set QueryCase (Integer)
                node_case = PySpin.CIntegerPtr(self.nodemap.GetNode('QueryCase'))
                if PySpin.IsAvailable(node_case) and PySpin.IsWritable(node_case):
                    node_case.SetValue(3)
                    self.log("Set QueryCase to 3")
                else:
                    self.log("QueryCase node not writable/found")

                # 2. Get QueryCaseEnable (Boolean)
                node_en = PySpin.CBooleanPtr(self.nodemap.GetNode('QueryCaseEnable'))
                if PySpin.IsAvailable(node_en) and PySpin.IsReadable(node_en):
                    val = node_en.GetValue()
                    self.log(f"[Result] QueryCaseEnable: {val}")
                else:
                    self.log("QueryCaseEnable node not readable")
                    
            except Exception as e:
                self.log(f"Error: {e}")
        else:
            self.log("Camera not initialized.")

    def cmd_node_command(self):
        code = """# [Command/복합 설정 예제] - FPS 30Hz 고정
# 1. Enable (Bool 설정)
node_en = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
node_en.SetValue(True)

# 2. Set Float (값 설정)
node_fr = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
node_fr.SetValue(30.0)
print("FPS Set to 30.0")
"""
        self.show_code(code)
        self.log("Setting FPS 30Hz...")
        if hasattr(self, 'nodemap'):
            try:
                node_en = PySpin.CBooleanPtr(self.nodemap.GetNode('AcquisitionFrameRateEnable'))
                if PySpin.IsAvailable(node_en) and PySpin.IsWritable(node_en): node_en.SetValue(True)
                node_fr = PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate'))
                if PySpin.IsAvailable(node_fr) and PySpin.IsWritable(node_fr): 
                    node_fr.SetValue(30.0)
                    self.log("FPS set to 30.0 Hz")
            except Exception as e: self.log(f"Error: {e}")

    def cmd_node_nuc(self):
        """Execute NUC (Non-Uniformity Correction)"""
        code = """# [Command 실행 예제] - NUC 수행
# Command 노드는 Execute()로 실행합니다.
node_nuc = PySpin.CCommandPtr(nodemap.GetNode('PerformNuc'))

if PySpin.IsAvailable(node_nuc) and PySpin.IsWritable(node_nuc):
    node_nuc.Execute()
    print("NUC Executed")
"""
        self.show_code(code)
        self.log("Attempting NUC...")
        
        if hasattr(self, 'nodemap'):
            try:
                # node_nuc = PySpin.CCommandPtr(self.nodemap.GetNode('PerformNuc'))
                # if not PySpin.IsAvailable(node_nuc):
                node_nuc = PySpin.CCommandPtr(self.nodemap.GetNode('NUCAction'))
                
                if PySpin.IsAvailable(node_nuc) and PySpin.IsWritable(node_nuc):
                    node_nuc.Execute()
                    self.log("NUC Command Executed.")
                else:
                    self.log("NUC node not found or not writable.")
            except Exception as e:
                self.log(f"NUC Error: {e}")
        else:
            self.log("Camera not initialized.")

    def cmd_node_autofocus(self):
        """Execute AutoFocus"""
        code = """# [Command 실행 예제] - 오토포커스
node_af = PySpin.CCommandPtr(nodemap.GetNode('AutoFocus'))

if PySpin.IsAvailable(node_af) and PySpin.IsWritable(node_af):
    node_af.Execute()
    print("AutoFocus Executed")
"""
        self.show_code(code)
        self.log("Attempting AutoFocus...")
        
        if hasattr(self, 'nodemap'):
            try:
                node_af = PySpin.CCommandPtr(self.nodemap.GetNode('AutoFocus'))
                if PySpin.IsAvailable(node_af) and PySpin.IsWritable(node_af):
                    node_af.Execute()
                    self.log("AutoFocus Command Executed.")
                else:
                    self.log("AutoFocus node not found or not writable.")
            except Exception as e:
                self.log(f"AutoFocus Error: {e}")
        else:
            self.log("Camera not initialized.")

    def run_node_val(self, name, ptr_type, label):
        if hasattr(self, 'nodemap'):
            try:
                node = ptr_type(self.nodemap.GetNode(name))
                if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
                    val = node.GetValue()
                    self.log(f"[Result] {label}: {val}")
                else: self.log(f"{name} not readable")
            except Exception as e: self.log(f"Error reading {name}: {e}")

    def cmd_begin_acquisition(self):
        """4-1. BeginAcquisition 호출"""
        code = """# Puts the camera into a state where it can deliver images.
self.spin_cam.BeginAcquisition()
"""
        self.show_code(code)
        if not hasattr(self, 'spin_cam') or self.spin_cam is None:
            self.log("Camera not initialized.")
            return
        try:
            self.spin_cam.BeginAcquisition()
            self.log("Camera is now in acquisition mode. (Not streaming yet)")
            # Do NOT start streaming here automatically.
            # self.is_streaming = True # Removed
        except Exception as e:
            self.log(f"BeginAcquisition Error: {e}")

    def cmd_end_acquisition(self):
        """4-2. EndAcquisition 호출"""
        code = """# Takes the camera out of acquisition mode.
self.spin_cam.EndAcquisition()
"""
        self.show_code(code)
        if not hasattr(self, 'spin_cam') or self.spin_cam is None:
            self.log("Camera not initialized.")
            return
        try:
            self.spin_cam.EndAcquisition()
            self.log("Camera is no longer in acquisition mode.")
        except Exception as e:
            self.log(f"EndAcquisition Error: {e}")

    def cmd_start_stream(self):
        if not hasattr(self, 'spin_cam') or self.spin_cam is None:
            self.log("Camera not initialized.")
            return
        if self.is_streaming:
            self.log("Streaming is already active.")
            return
            
        try:
            # 1. Update Mode & Params
            self.current_ir_mode = self.get_camera_ir_mode()
            self.log(f"Stream started with Camera Mode: {self.current_ir_mode}")

            if "Radiometric" in self.current_ir_mode:
                self.fetch_radiometric_params()
                self.calculate_radiometric_correction()

            # 2. Generate Param Setup Snippet (User Request: Split)
            # Shows only Parameter Fetching & Setup
            def f(val): return f"{float(val):.4f}"
            
            p_code = ""
            if "Radiometric" in self.current_ir_mode and hasattr(self, 'R'):
                p_code = f"""
# --- Radiometric Parameters (Fetched from Camera) ---
R = {f(self.R)}
B = {f(self.B)}
F = {f(self.F)}
J0 = {f(self.J0)}
J1 = {f(self.J1)}
X = {f(self.X)}
Alpha1 = {f(self.A1)}; Alpha2 = {f(self.A2)}
Beta1 = {f(self.B1)}; Beta2 = {f(self.B2)}

# --- Environmental Params ---
Emissivity = {f(self.Emissivity)}
Dist = {f(self.Dist)}
Humidity = {f(self.Humidity)}
ReflTemp = {f(self.TRefl)}; AtmTemp = {f(self.TAtm)}
ExtOptTemp = {f(self.ExtOpticsTemp)}; ExtOptTrans = {f(self.ExtOpticsTransmission)}

# (Calculation Logic moved to Snapshot Snippet)
"""
            
            code = f"""# Begin Acquisition & Setup
{p_code}
# Start Camera
camera.BeginAcquisition()
"""
            self.show_code(code)

            # 3. Real Action
            # Note: Assuming SpinView/Camera state is ready.
            # Only calling BeginAcquisition if not already streaming at driver level?
            # Safest is just set flag and start thread. The thread calls GetNextImage.
            # But usually we must call BeginAcquisition ONCE.
            if not self.spin_cam.IsStreaming():
                self.spin_cam.BeginAcquisition()
                
            self.is_streaming = True
            self.log("Live stream started.")
            self.stream_thread = threading.Thread(target=self.streaming_task, daemon=True)
            self.stream_thread.start()
            self.update_ui_loop()
            
        except Exception as e: self.log(f"Start Error: {e}")

    def streaming_task(self):
        while self.is_streaming:
            try:
                if self.spin_cam is None: break
                image_result = self.spin_cam.GetNextImage(500)
                if not image_result.IsIncomplete():
                    img_data = image_result.GetNDArray().copy()
                    self.current_raw_frame = img_data # Save for snapshot/norm usage
                    if not self.image_queue.full(): self.image_queue.put(img_data)
                image_result.Release()
            except: continue

    def cmd_stop_stream(self):
        self.show_code("self.is_streaming = False")
        if not self.is_streaming:
            self.log("Streaming is not active.")
            return
        self.is_streaming = False
        # The thread will stop on its own.
        # Wait a moment for the loop to terminate cleanly
        time.sleep(0.1)
        self.log("Live stream stopped.")

    def cmd_snapshot_process(self):
        """4. 스냅샷 찍고 -> Raw 데이터 확인 -> 온도 변환 과정 보여주기"""
        if not hasattr(self, 'spin_cam') or self.spin_cam is None:
            self.log("Camera not initialized!")
            return

        # [Fix] Read Actual Mode from Camera
        ir_mode = self.get_camera_ir_mode()
        self.current_ir_mode = ir_mode # Sync
        
        pixel_fmt = self.combo_pixel.get() # PixelFormat is less critical for math but good for log
        
        self.log(f"Processing Snapshot (Actual Cam Mode: {ir_mode}, Pix: {pixel_fmt})...")
        
        try:
            # 1. Acquire Data
            # Assumes BeginAcquisition has been called.
            img_result = self.spin_cam.GetNextImage(1000)
            if not img_result.IsIncomplete():
                raw_data = img_result.GetNDArray().copy()
                self.current_raw_frame = raw_data
                self.current_ir_mode = ir_mode
            img_result.Release()

            if raw_data is None:
                self.log("Failed to acquire image.")
                return

            # Convert to Celsius right away if possible for logging
            celsius_data = self.raw_to_celsius(raw_data, ir_mode)

            # Stats
            min_val, max_val = np.min(raw_data), np.max(raw_data)
            cy, cx = raw_data.shape[0]//2, raw_data.shape[1]//2
            center_val = raw_data[cy, cx]

            # === Print Raw Data Matrix to Log ===
            # Keep this section for showing raw data as requested.
            self.log("=" * 60)
            self.log(f"Raw Data Shape: {raw_data.shape}")
            self.log(f"Raw Stats: Min={min_val}, Max={max_val}, Center={center_val}")
            self.log("-" * 60)
            self.log("Raw Data Sample (center 5x5):")
            center_region_raw = raw_data[cy-2:cy+3, cx-2:cx+3]
            for row in center_region_raw:
                row_str = "  ".join(f"{val:5d}" for val in row)
                self.log(f"  [{row_str}]")
            self.log("=" * 60)

            # Code Gen & Calc
            code = f"""# --- Step 1. Get Raw Data ({pixel_fmt}) ---
image_result = cam.GetNextImage()
raw_data = image_result.GetNDArray()

# Show sample of raw data (center 5x5 region)
cy, cx = raw_data.shape[0]//2, raw_data.shape[1]//2
print(f"Raw Data Sample (center 5x5):\\n{{raw_data[cy-2:cy+3, cx-2:cx+3]}}")
print(f"Shape: {{raw_data.shape}}") 
print(f"Raw Stats: Min={{raw_data.min()}}, Max={{raw_data.max()}}")
# Actual: Min={min_val}, Max={max_val}, Center={center_val}
"""
            calc_msg = ""
            
            # --- Logic Branching ---
            # --- Logic Branching ---
            if "Radiometric" in ir_mode and hasattr(self, 'R'):
                # Radiometric Conversion Snippet with Korean Comments
                def f(val): return f"{float(val):.4f}"
                
                code += f"""
# [1] 파라미터 준비 (Start Stream 단계에서 로드된 값 사용)
R = {f(self.R)}; B = {f(self.B)}; F = {f(self.F)}
J0 = {f(self.J0)}; J1 = {f(self.J1)}
Emis = {f(self.Emissivity)}; Dist = {f(self.Dist)}; Humidity = {f(self.Humidity)}
AtmTemp = {f(self.TAtm)}; ReflTemp = {f(self.TRefl)}
X = {f(self.X)}; A1={f(self.A1)}; A2={f(self.A2)}; B1={f(self.B1)}; B2={f(self.B2)}

# [2] 보정 계수 계산 (Calculation Formulas)

# 1. 수증기 함량 (Water Vapour Content)
# 대기 온도(AtmTemp)와 상대 습도(Humidity)를 기반으로 계산
t_atm_c = AtmTemp - 273.15
h2o = Humidity * np.exp(1.5587 + 0.06939 * t_atm_c - 0.00027816 * t_atm_c**2 + 0.00000068455 * t_atm_c**3)

# 2. 대기 투과율 (Atmospheric Transmission, Tau)
# 물체와의 거리(Dist)와 수증기량(h2o)에 따른 감쇠율 계산
sqrt_dist = np.sqrt(Dist)
tau = X * np.exp(-sqrt_dist * (A1 + B1 * np.sqrt(h2o))) + \\
      (1 - X) * np.exp(-sqrt_dist * (A2 + B2 * np.sqrt(h2o)))

# 3. 보정 항 (Radiance Terms) & K2
# r1: 반사 온도에 의한 영향
r1 = ((1 - Emis) / Emis) * (R / (np.exp(B / ReflTemp) - F))
# r2: 대기 방출에 의한 영향
r2 = ((1 - tau) / (Emis * tau)) * (R / (np.exp(B / AtmTemp) - F))
# r3: 외부 광학계(Lens/Window)에 의한 영향
ext_opt_trans = {f(self.ExtOpticsTransmission)}
ext_opt_temp = {f(self.ExtOpticsTemp)}
r3 = ((1 - ext_opt_trans) / (Emis * tau * ext_opt_trans)) * (R / (np.exp(B / ext_opt_temp) - F))

K2 = r1 + r2 + r3
print(f"보정 계수 K2: {{K2:.4f}}, 투과율 Tau: {{tau:.4f}}")

# [3] 온도 변환 (Temp Conversion)
# 픽셀값(Raw) -> 복사 에너지(Radiance) -> 온도(Celsius)
raw_val = raw_data
radiance = (raw_val - J0) / J1
obj_rad = (radiance / Emis / tau) - K2
obj_rad = np.maximum(obj_rad, 1e-5) # 로그 오류 방지

temp_k = B / np.log(R / obj_rad + F)
temp_c = temp_k - 273.15
"""
                # Actual Calculation for Logging
                try:
                    # Reuse cached correction factors if available
                    if hasattr(self, 'K2'):
                        term = ((raw_data - self.J0) / self.J1 / self.Emissivity / self.Tau) - self.K2
                        term = np.maximum(term, 1e-9)
                        celsius_img = (self.B / np.log(self.R / term + self.F)) - 273.15
                        
                        ct_temp = celsius_img[cy, cx]
                        calc_msg = f"Center Temp: {ct_temp:.2f} C (Radiometric Corrected)"
                    else:
                        calc_msg = "Radiometric params not fully ready."
                except Exception as e:
                    calc_msg = f"Calc Error: {e}"

            elif "TemperatureLinear10mK" in ir_mode:
                # Linear 10mK Conversion
                code += """
# --- Step 2. Convert Linear 10mK Data ---
# Formula: T(C) = (Digital * 0.01) - 273.15
temp_c = (raw_data * 0.01) - 273.15
"""
                ct_temp = (center_val * 0.01) - 273.15
                calc_msg = f"Center Temp: {ct_temp:.2f} C (Linear 10mK)"

            elif "TemperatureLinear100mK" in ir_mode:
                # Linear 100mK Conversion
                code += """
# --- Step 2. Convert Linear 100mK Data ---
                        # Formula: T(C) = (Digital * 0.1) - 273.15
                        temp_c = (raw_data * 0.1) - 273.15
                        """
                ct_temp = (center_val * 0.1) - 273.15
                calc_msg = f"Center Temp: {ct_temp:.2f} C (Linear 100mK)"
            else:
                # Raw Mono8 or other
                code += "# No Temperature conversion selected (Mono8 or Raw)"
                calc_msg = f"Center Value: {center_val} (Raw)"

            code += f"\n# Final Result: {calc_msg}"

            self.show_code(code)
            self.log(f"Snapshot Processed. {calc_msg}")

            # Safe static update
            self.current_raw_frame = raw_data
            self.current_ir_mode = ir_mode  # Save for tooltip
            
            # Display Image (Static)
            norm_img = cv2.normalize(raw_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
            
            h, w, c = disp_img.shape
            display_h = 450
            display_w = int(display_h * (w/h))
            disp_img = cv2.resize(disp_img, (display_w, display_h))

            img_pil = Image.fromarray(disp_img)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.image_label.config(image=img_tk, text="")
            self.image_label.image = img_tk
            
            # Update Stats Label
            try:
                celsius_data = self.raw_to_celsius(raw_data, ir_mode)
                
                if celsius_data is not None:
                    min_v = np.min(celsius_data)
                    max_v = np.max(celsius_data)
                    avg_v = np.mean(celsius_data)
                    txt_stats = f"Max: {max_v:.1f}C  Min: {min_v:.1f}C  Avg: {avg_v:.1f}C"
                else:
                    # Fallback to raw if conversion fails or not applicable
                    min_v = np.min(raw_data)
                    max_v = np.max(raw_data)
                    avg_v = np.mean(raw_data)
                    txt_stats = f"Max: {max_v}  Min: {min_v}  Avg: {avg_v:.1f} (Raw)"

                self.stats_label.config(text=txt_stats)
            except Exception as e:
                self.log(f"Stats update on snapshot failed: {e}")
        except Exception as e:
            self.log(f"Snapshot processing error: {e}")

    def cmd_disconnect(self):
        code = """# Stop streaming thread
self.is_streaming = False

# Gracefully release camera resources
if self.spin_cam:
    # End acquisition if it's running
    try: self.spin_cam.EndAcquisition()
    except: pass

    # De-initialize the camera
    self.spin_cam.DeInit()
    del self.spin_cam
    self.spin_cam = None

# Clear the camera list
if self.cam_list:
    self.cam_list.Clear()

# Release the system instance
if self.system:
    self.system.ReleaseInstance()
"""
        self.show_code(code)
        self.log("Disconnecting camera and releasing resources...")
        try:
            self.log("Stopping stream...")
            self.is_streaming = False
            time.sleep(0.1)
            if self.spin_cam:
                try: 
                    self.spin_cam.EndAcquisition() # Ensure acquisition is stopped
                    self.log("Acquisition ended.")
                except PySpin.SpinnakerException: pass # Ignore if not acquiring
                self.log("De-initializing camera...")
                self.spin_cam.DeInit()
                del self.spin_cam
                self.spin_cam = None
            if self.cam_list:
                self.log("Clearing camera list...")
                self.cam_list.Clear()
            if self.system:
                self.log("Releasing system instance...")
                self.system.ReleaseInstance()
            self.log("All resources released. Disconnected.")
            self.image_label.config(image='', text="Disconnected")
        except Exception as e:
            self.log(f"Disconnect Error: {e}")

    def fetch_radiometric_params(self):
        """Fetches Radiometric Params using Direct Access (Fast) & Recursive Scan (Fallback)."""
        self.log("Initializing Radiometric Parameters...")
        
        if not hasattr(self, 'nodemap') or self.nodemap is None:
            return

        found_vals = {}
        
        # 1. Helper to safely read Value based on Type
        def read_node_val(node):
            if not PySpin.IsReadable(node): return None
            try:
                nt = node.GetPrincipalInterfaceType()
                if nt == PySpin.intfIFloat:
                    return PySpin.CFloatPtr(node).GetValue()
                elif nt == PySpin.intfIInteger:
                    return float(PySpin.CIntegerPtr(node).GetValue())
                elif nt == PySpin.intfIEnumeration:
                    return float(PySpin.CEnumerationPtr(node).GetIntValue())
                elif nt == PySpin.intfIBoolean:
                    return 1.0 if PySpin.CBooleanPtr(node).GetValue() else 0.0
            except: return None
            return None

        # 2. Direct Fetch (Try standard GenICam names & FLIR specific names)
        # Check the user provided image: Name is "J0", "J1" (Type Int/Float)
        direct_keys = [
            "J0", "J1", "GlobalGain", "GlobalOffset", 
            "R", "B", "F", "X", "alpha1", "alpha2", "beta1", "beta2",
            "ObjectDistance", "ObjectEmissivity", "ReflectedTemperature",
            "RelativeHumidity", "AtmosphericTemperature",
            "ExternalOpticsTemperature", "ExternalOpticsTransmission"
        ]

        for key in direct_keys:
            try:
                node = self.nodemap.GetNode(key)
                val = read_node_val(node)
                if val is not None:
                    found_vals[key] = val
                    # Also map DisplayName just in case (e.g. key="J0", dname="Global Offset")
                    try:
                        dname = node.GetDisplayName().replace(" ", "").strip()
                        if dname not in found_vals: found_vals[dname] = val
                    except: pass
            except: pass

        # 3. Recursive Scan (Fallback for hidden nodes)
        if "R" not in found_vals or "J0" not in found_vals:
            self.log("Direct fetch incomplete, scanning tree...")
            target_set = set(direct_keys)
            
            def recursive_scan(node):
                try:
                    nt = node.GetPrincipalInterfaceType()
                    if nt == PySpin.intfICategory:
                        features = PySpin.CCategoryPtr(node).GetFeatures()
                        for f in features: recursive_scan(f)
                    elif PySpin.IsReadable(node):
                        dname = node.GetDisplayName().replace(" ", "").strip()
                        if dname in target_set:
                            val = read_node_val(node)
                            if val is not None: found_vals[dname] = val
                except: pass

            try:
                recursive_scan(self.nodemap.GetNode("Root"))
            except: pass

        # --- Assign Values ---
        def v(name, default): 
            # Check exact key, then check DisplayName variants
            return found_vals.get(name, found_vals.get(name.replace(" ", ""), default))

        # 1. Constants
        self.R = v("R", 15000.0)
        self.B = v("B", 1450.0)
        self.F = v("F", 1.0)
        self.X = v("X", 1.9)
        self.A1 = v("alpha1", 0.006569)
        self.A2 = v("alpha2", 0.01262)
        self.B1 = v("beta1", -0.002276)
        self.B2 = v("beta2", -0.00667)

        # 2. Env Params
        self.TRefl = v("ReflectedTemperature", 293.15)
        self.TAtm = v("AtmosphericTemperature", self.TRefl)
        self.TAtmC = self.TAtm - 273.15
        
        self.ExtOpticsTemp = v("ExternalOpticsTemperature", 293.15)
        self.ExtOpticsTransmission = v("ExternalOpticsTransmission", 1.0)
        self.Humidity = v("RelativeHumidity", 0.55)
        self.Dist = v("ObjectDistance", 2.0)
        self.Emissivity = v("ObjectEmissivity", 0.95)

        # 3. J0, J1 Logic (Unified, Robust)
        # Try Global first (DisplayName usually), then Name keys
        gain_val = found_vals.get("GlobalGain", found_vals.get("J1")) 
        offset_val = found_vals.get("GlobalOffset", found_vals.get("J0"))
        
        # If still None, map J1 default to 1.0, J0 to 0.0
        self.J1 = gain_val if gain_val is not None else 1.0
        self.J0 = offset_val if offset_val is not None else 0.0
        
        self.log(f"Params Fetched: R={self.R:.0f} B={self.B:.0f} J1={self.J1:.2f} J0={self.J0:.0f}")

    def calculate_radiometric_correction(self):
        """Calculates Tau, K2, H2O based on fetched params."""
        if not hasattr(self, 'R'): return 

        try:
            # H2O
            self.H2O = self.Humidity * np.exp(1.5587 + 0.06939 * self.TAtmC - 0.00027816 * self.TAtmC**2 + 0.00000068455 * self.TAtmC**3)
            
            # Tau
            sqrt_dist = np.sqrt(self.Dist)
            self.Tau = self.X * np.exp(-sqrt_dist * (self.A1 + self.B1 * np.sqrt(self.H2O))) + \
                       (1 - self.X) * np.exp(-sqrt_dist * (self.A2 + self.B2 * np.sqrt(self.H2O)))
                       
            # Radiance Terms
            r1 = ((1 - self.Emissivity) / self.Emissivity) * (self.R / (np.exp(self.B / self.TRefl) - self.F))
            r2 = ((1 - self.Tau) / (self.Emissivity * self.Tau)) * (self.R / (np.exp(self.B / self.TAtm) - self.F))
            r3 = ((1 - self.ExtOpticsTransmission) / (self.Emissivity * self.Tau * self.ExtOpticsTransmission)) * \
                      (self.R / (np.exp(self.B / self.ExtOpticsTemp) - self.F))
            
            self.K2 = r1 + r2 + r3
            
            self.radiometric_ready = True
            
            self.log(f"Radiometric Correction: K2={self.K2:.2f}, Tau={self.Tau:.4f}")

        except Exception as e:
            self.log(f"Error Calculating Factors: {e}")
            self.radiometric_ready = False

    def cmd_normalize_process(self):
        """Normalize Process: Snapshot -> Temp -> PaletteUtil -> ROI Stats"""
        pal_name = self.combo_palette.get().upper()
        code = f"""# Normalize & Palette Process
# 1. Snapshot (Get Raw)
image_result = cam.GetNextImage()
raw_data = image_result.GetNDArray()

# 2. Temperature Conversion
# temp_data = raw_to_celsius(raw_data) ...

# 3. Apply Palette (Using PaletteUtil)
from palette_util import PaletteUtil
pu = PaletteUtil()

if "{pal_name}" == "GRAY":
    norm_img = cv2.normalize(temp_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
else:
    # Apply Custom LUT (Iron, Rainbow, etc.)
    # Returns BGR image using internal normalize + LUT logic
    color_img = pu.apply_color_palette(temp_data, "{pal_name}")

# 4. ROI & Stats (Draw Red Box)
# ...
cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), thickness)
cv2.imshow("Result", color_img)
"""
        self.show_code(code)
        
        if not hasattr(self, 'spin_cam') or self.spin_cam is None:
            self.log("Camera not initialized!")
            return
            
        self.log("Processing Normalize with ROI...")
        
        raw_data = None
        try:
            # 1. Acquire Logic
            if self.is_streaming:
                # If streaming, use the latest frame captured by the thread
                if hasattr(self, 'current_raw_frame') and self.current_raw_frame is not None:
                    raw_data = self.current_raw_frame.copy()
                else:
                    self.log("Streaming but no frame available yet.")
                    return
            else:
                # Not streaming: Manual Capture
                # Check if camera is strictly in Acquisition mode
                # Note: self.spin_cam.IsStreaming() checks if device is streaming
                was_acquiring = self.spin_cam.IsStreaming()
                
                if not was_acquiring:
                    self.spin_cam.BeginAcquisition()
                
                try:
                    img_result = self.spin_cam.GetNextImage(1000)
                    if not img_result.IsIncomplete():
                        raw_data = img_result.GetNDArray().copy()
                    img_result.Release()
                except Exception as ex:
                    self.log(f"Acquire failed: {ex}")
                finally:
                    # If we started it just for this, stop it
                    if not was_acquiring:
                        self.spin_cam.EndAcquisition()
                        
            if raw_data is None:
                self.log("Failed to acquire valid image.")
                return
            
            # 2. Temp Convert
            ir_mode = self.combo_ir.get()
            temp_data = self.raw_to_celsius(raw_data, ir_mode)
            
            if temp_data is None:
                self.log("Temp conversion failed.")
                return
                
            # 3. Apply Palette via PaletteUtil (Robust method)
            # Use 'PaletteUtil' to apply LUTs like IRON, RAINBOW, etc.
            selected_palette = self.combo_palette.get().upper() # PaletteUtil expects uppercase keys
            
            # If Gray or Default, use standard normalization
            if selected_palette == "GRAY" or selected_palette == "DEFAULT":
                min_t, max_t = np.min(temp_data), np.max(temp_data)
                diff = max_t - min_t if max_t != min_t else 1.0
                norm_img = ((temp_data - min_t) / diff * 255).astype(np.uint8)
                color_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
            else:
                # Use PaletteUtil logic (Load LUT -> Apply)
                # Note: PaletteUtil expects raw or normalized? 
                # Let's check PaletteUtil: it calls normalize_image inside.
                # So we pass raw temp_data directly? 
                # Wait, PaletteUtil usually expects raw image data (uint16/float). 
                # Let's pass temp_data (float) and let it normalize.
                
                # Check if PaletteUtil is available
                if not PALETTE_AVAILABLE:
                    self.log("PaletteUtil not loaded. Using Gray.")
                    min_t, max_t = np.min(temp_data), np.max(temp_data)
                    diff = max_t - min_t if max_t != min_t else 1.0
                    norm_img = ((temp_data - min_t) / diff * 255).astype(np.uint8)
                    color_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
                else:
                    color_img = self.palette_util.apply_color_palette(temp_data, selected_palette)
                    if color_img is None:
                        # Fallback
                        min_t, max_t = np.min(temp_data), np.max(temp_data)
                        diff = max_t - min_t if max_t != min_t else 1.0
                        norm_img = ((temp_data - min_t) / diff * 255).astype(np.uint8)
                        color_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
                    else:
                        # PaletteUtil returns BGR, convert to RGB for Tkinter
                        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            
            # 4. ROI Stats & Drawing
            h_img, w_img = temp_data.shape[:2] # Careful with shape

            
            # 4. ROI Stats
            h_img, w_img = temp_data.shape
            
            rw, rh = 80, 80
            rx, ry = (w_img - rw)//2, (h_img - rh)//2
            
            roi_data = temp_data[ry:ry+rh, rx:rx+rw]
            avg_val = np.mean(roi_data)
            roi_max = np.max(roi_data)
            roi_min = np.min(roi_data)
            
            self.log(f"ROI ({rx},{ry},{rw}x{rh}) Stats: Avg={avg_val:.2f}C")
            
            # Update Stats Label with ROI Info
            self.stats_label.config(text=f"ROI Max: {roi_max:.1f}C  Min: {roi_min:.1f}C  Avg: {avg_val:.1f}C")

            # 5. Draw Box & Display
            
            # Dynamic thickness based on resolution (approx 0.5% of width)
            # This ensures visibility even when downscaled
            thickness = max(2, int(w_img * 0.005))
            
            # Ensure int coordinates
            pt1 = (int(rx), int(ry))
            pt2 = (int(rx+rw), int(ry+rh))
            
            # color_img is already RGB (or BGR -> converted above)
            # CAUTION: cv2.rectangle modifies in-place.
            # If color_img came from PaletteUtil (BGR), we converted to RGB.
            # So here we draw Red as (255, 0, 0) because we are in RGB space for PIL.
            cv2.rectangle(color_img, pt1, pt2, (255, 0, 0), thickness) 
            
            # Display on Tkinter Canvas
            
            # Resize Logic (Match Snapshot size: Fixed Height 450)
            h, w, c = color_img.shape
            display_h = 450
            display_w = int(display_h * (w/h))
            
            # Use OpenCV resize for better performance/quality control before PIL
            color_img_resized = cv2.resize(color_img, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
            
            img_pil = Image.fromarray(color_img_resized)
            
            # Remove previous dynamic resizing logic based on label size
            # w_disp = self.image_label.winfo_width() ... (Removed)

            img_tk = ImageTk.PhotoImage(img_pil)
            self.image_label.configure(image=img_tk, text="")
            self.image_label.image = img_tk
            
        except Exception as e:
            self.log(f"Normalize Error: {e}")

    def raw_to_celsius(self, raw_data, ir_mode):
        """Converts raw sensor data (scalar or array) to Celsius based on the IR mode."""
        if raw_data is None:
            return None

        try:
            # Note: raw_data can be a scalar (from tooltip) or numpy array (frame)
            if "TemperatureLinear10mK" in ir_mode:
                return (raw_data * 0.01) - 273.15
                
            elif "TemperatureLinear100mK" in ir_mode:
                return (raw_data * 0.1) - 273.15
                
            elif "Radiometric" in ir_mode:
                if not hasattr(self, 'radiometric_ready') or not self.radiometric_ready:
                    return None 

                # User-requested Formula Implementation
                # image_Radiance = (image_data - self.J0) / self.J1
                # Celsius_data = (self.B / np.log(self.R / ((image_Radiance / self.Emissivity / self.Tau) - self.K2) + self.F)) - 273.15
                
                image_Radiance = (raw_data - self.J0) / self.J1
                
                term = (image_Radiance / self.Emissivity / self.Tau) - self.K2
                
                # Safety for Log: term must be positive
                # In extremely cold spots or invalid data, term might be <= 0
                term = np.maximum(term, 1e-9)
                
                Celsius_data = (self.B / np.log(self.R / term + self.F)) - 273.15
                
                return Celsius_data
            else:
                return None 
        except Exception as e:
            return None

    def update_ui_loop(self):
        if not self.is_streaming: return

        if not self.image_queue.empty():
            raw_img = self.image_queue.get()
            self.current_raw_frame = raw_img 
            
            # Stats Logic (Celsius Conversion)
            txt_stats = ""
            ir_mode = getattr(self, 'current_ir_mode', 'Radiometric')
            
            try:
                celsius_data = self.raw_to_celsius(raw_img, ir_mode)
                
                if celsius_data is not None:
                    min_v = np.min(celsius_data)
                    max_v = np.max(celsius_data)
                    avg_v = np.mean(celsius_data)
                    txt_stats = f"Max: {max_v:.1f}C  Min: {min_v:.1f}C  Avg: {avg_v:.1f}C"
                else:
                    # Fallback to raw if conversion fails or not applicable
                    min_v = np.min(raw_img)
                    max_v = np.max(raw_img)
                    avg_v = np.mean(raw_img)
                    txt_stats = f"Max: {max_v}  Min: {min_v}  Avg: {avg_v:.1f} (Raw)"

                self.stats_label.config(text=txt_stats)
            except: pass

            # Palette Processing
            palette_name = self.combo_palette.get()
            display_img = None
            
            if palette_name == "Gray":
                norm_img = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                display_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
            else:
                # Use PaletteUtil
                name_upper = palette_name.upper()
                h_r, w_r = raw_img.shape
                display_img = self.palette_util.apply_color_palette(raw_img, name_upper, out_size=(w_r, h_r))
                
                if display_img is not None:
                     display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

            if display_img is not None:
                h, w, c = display_img.shape
                display_h = 480
                display_w = int(display_h * (w/h))
                self.display_w = display_w
                self.display_h = display_h
                display_img = cv2.resize(display_img, (display_w, display_h))
                img_pil = Image.fromarray(display_img)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.image_label.config(image=img_tk, text="")
                self.image_label.image = img_tk 

        self.root.after(30, self.update_ui_loop)

    def get_camera_ir_mode(self):
        """Helper to read the actual IRFormat from the camera node."""
        if not hasattr(self, 'nodemap') or self.nodemap is None:
            return "Unknown"
        
        try:
            node_ir = PySpin.CEnumerationPtr(self.nodemap.GetNode('IRFormat'))
            if PySpin.IsAvailable(node_ir) and PySpin.IsReadable(node_ir):
                 curr_entry = node_ir.GetCurrentEntry()
                 if PySpin.IsAvailable(curr_entry) and PySpin.IsReadable(curr_entry):
                     return curr_entry.GetSymbolic()
        except Exception:
            pass
        return self.combo_ir.get() # Fallback to UI if read fails

    def callback_mouse_move(self, event):
        if not hasattr(self, 'current_raw_frame') or self.current_raw_frame is None or not hasattr(self, 'display_w') or self.display_w == 0:
            self.tooltip.place_forget()
            return

        gui_w = self.image_label.winfo_width()
        gui_h = self.image_label.winfo_height()
        
        # Calculate the offset of the image (assuming it's centered)
        offset_x = (gui_w - self.display_w) / 2
        offset_y = (gui_h - self.display_h) / 2
        
        ex, ey = event.x, event.y

        # Check if the mouse is inside the displayed image area
        if (offset_x <= ex < offset_x + self.display_w) and \
           (offset_y <= ey < offset_y + self.display_h):
            
            # Coords relative to the top-left of the image inside the label
            img_x = ex - offset_x
            img_y = ey - offset_y
            
            # Scale to original raw frame dimensions
            orig_h, orig_w = self.current_raw_frame.shape
            raw_x = int(img_x * (orig_w / self.display_w))
            raw_y = int(img_y * (orig_h / self.display_h))

            if 0 <= raw_x < orig_w and 0 <= raw_y < orig_h:
                raw_val = self.current_raw_frame[raw_y, raw_x]
                
                ir_mode = getattr(self, 'current_ir_mode', 'Radiometric')
                
                try:
                    celsius_val = self.raw_to_celsius(raw_val, ir_mode)
                    if celsius_val is not None:
                        temp_text = f"{celsius_val:.1f} C"
                    else:
                        temp_text = f"Raw: {raw_val}"
                except:
                    temp_text = f"Raw: {raw_val}" # Fallback
                
                self.tooltip.config(text=temp_text)
                
                # Position tooltip relative to the event
                root_x = self.image_label.winfo_rootx() + ex + 15
                root_y = self.image_label.winfo_rooty() + ey - 5
                
                self.tooltip.place(x=root_x - self.root.winfo_rootx(), y=root_y - self.root.winfo_rooty())
                return

        # If we are here, the mouse is outside the image, so hide tooltip
        self.tooltip.place_forget()

if __name__ == "__main__":
    root = tk.Tk()
    app = FLIR_Demo_App(root)
    root.mainloop()
