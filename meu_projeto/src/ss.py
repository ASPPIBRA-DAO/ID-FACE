import cv2
import threading
import numpy as np
import time
import math
import os
import logging
from collections import deque

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class CameraThread(threading.Thread):
    MAX_RECONNECTS = 5
    MIN_FILE_SIZE = 1024  # 1KB
    RECONNECT_DELAY = 1  # 1 segundo entre tentativas

    def __init__(self, url, idx, timeout=5, motion_threshold=500, 
                 record_on_motion=True, min_record_time=5, record_directory='recordings'):
        super().__init__()
        self.url = url
        self.idx = idx
        self.frame = None
        self.running = False
        self.cap = None
        self.is_active = False
        self.last_update = 0
        self.timeout = timeout
        self.lock = threading.Lock()
        
        # Parâmetros de detecção de movimento e gravação
        self.motion_threshold = motion_threshold
        self.record_on_motion = record_on_motion
        self.min_record_time = min_record_time
        self.record_directory = record_directory
        
        # Gerenciamento de estado
        self.reconnect_attempts = 0
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.recording = False
        self.video_writer = None
        self.last_motion_time = 0
        self.motion_detected = False
        self.current_filename = None
        self.buffer = deque(maxlen=30)  # Buffer para 1 segundo a 30 FPS

    def run(self):
        self.running = True
        try:
            self.cap = self.initialize_camera()
            if not self.cap.isOpened():
                logging.error(f"Failed to connect to camera {self.idx}")
                return

            with self.lock:
                self.is_active = True
                self.last_update = time.time()
            
            logging.info(f"Connected to camera {self.idx}")

            while self.running:
                start_read = time.time()
                ret, frame = self.cap.read()
                
                if not ret:
                    if (time.time() - start_read) > (self.timeout / 2):
                        self.handle_read_error()
                    continue

                # Adicionar frame ao buffer
                self.buffer.append(frame.copy())

                # Detecção de movimento avançada
                current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fgmask = self.fgbg.apply(current_gray)
                _, thresh = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                motion_detected = any(cv2.contourArea(c) > self.motion_threshold for c in contours)

                with self.lock:
                    self.motion_detected = motion_detected

                # Controle de gravação
                if motion_detected:
                    self.last_motion_time = time.time()
                    if not self.recording and self.record_on_motion:
                        self.start_recording()

                if self.recording:
                    if (time.time() - self.last_motion_time) > self.min_record_time:
                        self.stop_recording()
                    else:
                        self.add_recording_indicator(frame)
                        self.video_writer.write(frame)

                with self.lock:
                    self.frame = frame.copy() if frame is not None else None
                    self.last_update = time.time()

        except Exception as e:
            logging.error(f"Error in camera {self.idx}: {str(e)}")
        
        finally:
            self.stop_recording()
            with self.lock:
                self.is_active = False
            if self.cap:
                self.cap.release()
            self.running = False

    def initialize_camera(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        return cap

    def handle_read_error(self):
        logging.warning(f"Read timeout on camera {self.idx}")
        self.reconnect_attempts += 1
        if self.reconnect_attempts > self.MAX_RECONNECTS:
            logging.error(f"Max reconnection attempts reached for camera {self.idx}")
            self.running = False
            return
        
        logging.info(f"Attempting to reconnect to camera {self.idx} (attempt {self.reconnect_attempts})")
        self.cap.release()
        time.sleep(self.RECONNECT_DELAY)
        self.cap = self.initialize_camera()
        
        if self.cap.isOpened():
            self.reconnect_attempts = 0
            logging.info(f"Successfully reconnected to camera {self.idx}")

    def add_recording_indicator(self, frame):
        cv2.drawMarker(frame, (frame.shape[1]-20, 20), 
                      (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 20, 2)

    def start_recording(self):
        if not self.check_disk_space():
            logging.warning("Insufficient disk space to start recording")
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.current_filename = os.path.join(self.record_directory, f"camera_{self.idx}_{timestamp}.mp4")
        os.makedirs(self.record_directory, exist_ok=True)
        
        if self.frame is not None:
            height, width = self.frame.shape[:2]
        else:
            width, height = 640, 480

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 60:
            fps = 20.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.current_filename, fourcc, fps, (width, height))
        
        if self.video_writer.isOpened():
            self.recording = True
            # Escrever frames do buffer
            while self.buffer:
                self.video_writer.write(self.buffer.popleft())
            logging.info(f"Started recording for camera {self.idx}")
        else:
            logging.error(f"Failed to start recording for camera {self.idx}")
            self.video_writer = None

    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            filename = self.current_filename
            self.video_writer = None
            self.recording = False
            
            if filename and os.path.exists(filename):
                if os.path.getsize(filename) < self.MIN_FILE_SIZE:
                    os.remove(filename)
                    logging.info(f"Removed small recording file: {filename}")
                else:
                    logging.info(f"Recording saved: {filename}")
            self.current_filename = None

    def check_disk_space(self, min_space_gb=1):
        try:
            stat = os.statvfs(self.record_directory)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            return available_gb >= min_space_gb
        except:
            logging.error("Error checking disk space")
            return False

    def check_activity(self):
        with self.lock:
            if self.is_active and (time.time() - self.last_update) > self.timeout:
                logging.warning(f"Camera {self.idx} marked inactive")
                self.is_active = False
            return self.is_active

    def stop(self):
        self.running = False

def create_adaptive_mosaic(frames, target_size=(640, 480)):
    if not frames:
        return None
    
    resized_frames = []
    for f in frames:
        if f is not None and f.size > 0:
            h, w = f.shape[:2]
            ratio = min(target_size[0]/w, target_size[1]/h)
            new_size = (int(w*ratio), int(h*ratio))
            resized = cv2.resize(f, new_size)
            
            # Adicionar bordas pretas
            delta_w = target_size[0] - new_size[0]
            delta_h = target_size[1] - new_size[1]
            resized = cv2.copyMakeBorder(resized, 
                                       top=0,
                                       bottom=delta_h,
                                       left=0,
                                       right=delta_w,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=(0, 0, 0))
            resized_frames.append(resized)
        else:
            black_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            resized_frames.append(black_frame)
    
    num_frames = len(resized_frames)
    cols = max(1, math.isqrt(num_frames))
    rows = math.ceil(num_frames / cols)
    
    mosaic_rows = []
    for i in range(rows):
        row_frames = resized_frames[i*cols : (i+1)*cols]
        while len(row_frames) < cols:
            row_frames.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
        mosaic_rows.append(cv2.hconcat(row_frames))
    
    return cv2.vconcat(mosaic_rows)

# Configuração de câmeras
CAMERA_CONFIG = [
    {"url": "http://192.168.1.7:8080/video", "name": "Entrada"},
    {"url": "http://192.168.1.4:8080/video", "name": "Jardim"},
]

# Inicialização das threads
webcam_threads = []
for idx, config in enumerate(CAMERA_CONFIG):
    thread = CameraThread(
        url=config["url"],
        idx=idx,
        timeout=5,
        motion_threshold=500,
        record_on_motion=True,
        min_record_time=5,
        record_directory='recordings'
    )
    thread.start()
    webcam_threads.append(thread)

# Verificação inicial de conexão
start_time = time.time()
while time.time() - start_time < 10:
    if any(t.is_active for t in webcam_threads):
        break
    time.sleep(0.1)

active_cams = [t for t in webcam_threads if t.is_active]
if not active_cams:
    logging.error("No active cameras. Exiting...")
    exit()

logging.info(f"Active cameras: {len(active_cams)}")

try:
    TARGET_FPS = 30
    last_activity_check = time.time()
    
    while True:
        start_time = time.time()
        
        # Verificação periódica de atividade
        if time.time() - last_activity_check > 1:
            for thread in webcam_threads:
                thread.check_activity()
            active_cams = [t for t in webcam_threads if t.is_active]
            last_activity_check = time.time()
        
        # Coleta de quadros com overlays
        frames = []
        for idx, thread in enumerate(webcam_threads):
            with thread.lock:
                if thread.is_active and thread.frame is not None:
                    frame = thread.frame.copy()
                    # Adicionar overlays
                    cv2.putText(frame, CAMERA_CONFIG[idx]['name'], (10, frame.shape[0]-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if thread.motion_detected:
                        cv2.putText(frame, "MOTION", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if thread.recording:
                        cv2.putText(frame, "REC", (10, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f"{CAMERA_CONFIG[idx]['name']} Offline", (50, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frames.append(frame)
        
        # Criar e exibir mosaico
        mosaic = create_adaptive_mosaic(frames)
        if mosaic is not None:
            cv2.imshow('IP Camera Mosaic', mosaic)
        
        # Controle de FPS
        elapsed = (time.time() - start_time) * 1000
        delay = max(1, int((1000 / TARGET_FPS) - elapsed))
        
        if cv2.waitKey(delay) == 27:
            break

finally:
    # Finalização segura
    for thread in webcam_threads:
        thread.stop()
        thread.join()
    
    cv2.destroyAllWindows()
    logging.info("System shutdown completed.")