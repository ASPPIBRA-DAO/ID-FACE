import cv2
import threading
import numpy as np
import time
import math

# Função para criar o mosaico adaptativo
def create_adaptive_mosaic(frames, target_size=(640, 480)):
    if not frames:
        return None
    
    # Calcula layout dinâmico
    num_frames = len(frames)
    cols = max(1, math.isqrt(num_frames))
    rows = math.ceil(num_frames / cols)
    
    mosaic_rows = []
    for i in range(rows):
        row_frames = frames[i*cols : (i+1)*cols]
        resized_row = [cv2.resize(f, target_size) for f in row_frames]
        
        # Preenche espaços vazios na última linha
        while len(resized_row) < cols:
            resized_row.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
            
        mosaic_rows.append(cv2.hconcat(resized_row))
    
    return cv2.vconcat(mosaic_rows)

# Classe para captura de cada câmera em uma thread separada
class CameraThread(threading.Thread):
    def __init__(self, url, idx, timeout=5):
        super().__init__()
        self.url = url
        self.idx = idx
        self.frame = None
        self.running = False
        self.cap = None
        self.is_active = False
        self.last_update = 0
        self.timeout = timeout  # Timeout em segundos
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        try:
            # Configura timeout de conexão (5000ms)
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            
            if not self.cap.isOpened():
                print(f"Falha ao conectar à câmera {self.idx}")
                return

            with self.lock:
                self.is_active = True
                self.last_update = time.time()
            
            print(f"Conectado à câmera {self.idx}")

            while self.running:
                start_read = time.time()
                ret, frame = self.cap.read()
                
                if not ret:
                    # Verifica timeout de leitura
                    if (time.time() - start_read) > (self.timeout / 2):
                        print(f"Timeout de leitura na câmera {self.idx}")
                        break
                    continue

                with self.lock:
                    self.frame = frame
                    self.last_update = time.time()

        except Exception as e:
            print(f"Erro na câmera {self.idx}: {str(e)}")
        
        finally:
            with self.lock:
                self.is_active = False
            if self.cap:
                self.cap.release()
            self.running = False

    def check_activity(self):
        with self.lock:
            if self.is_active and (time.time() - self.last_update) > self.timeout:
                print(f"Câmera {self.idx} marcada como inativa")
                self.is_active = False
            return self.is_active

    def stop(self):
        self.running = False

# Inicializa os classificadores para reconhecimento facial e detecção de pessoas
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Lista de URLs das câmeras
urls = [
    "http://192.168.1.2:8080/video",
    "http://192.168.1.4:8080/video",
    # Adicione mais URLs aqui, se necessário
]

# Inicialização das threads para cada câmera
webcam_threads = []
for idx, url in enumerate(urls):
    thread = CameraThread(url, idx, timeout=5)
    thread.start()
    webcam_threads.append(thread)

# Espera por conexões com timeout de 10 segundos
start_time = time.time()
while time.time() - start_time < 10:
    if any(t.is_active for t in webcam_threads):
        break
    time.sleep(0.1)

active_cams = [t for t in webcam_threads if t.is_active]
if not active_cams:
    print("Nenhuma câmera ativa. Encerrando...")
    exit()

print(f"Câmeras ativas: {len(active_cams)}")

try:
    TARGET_FPS = 30
    last_activity_check = time.time()
    
    while True:
        loop_start = time.time()
        
        # Verifica atividade a cada segundo
        if time.time() - last_activity_check > 1:
            for thread in webcam_threads:
                thread.check_activity()
            active_cams = [t for t in webcam_threads if t.is_active]
            last_activity_check = time.time()
        
        # Coleta e processa os frames de cada câmera
        frames = []
        for idx, thread in enumerate(webcam_threads):
            with thread.lock:
                # Se a câmera está ativa e possui um frame válido, faz a cópia
                if thread.is_active and thread.frame is not None:
                    frame = thread.frame.copy()
                    # Reconhecimento facial: converte para escala de cinza e detecta faces
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Detecção de pessoas usando o detector HOG
                    rects, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
                    for (x, y, w, h) in rects:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Exibe os contadores no frame
                    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Pessoas: {len(rects)}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    # Se a câmera estiver inativa, exibe mensagem de offline
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f"Camera {idx} Offline", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frames.append(frame)
        
        # Cria e exibe o mosaico com os frames processados
        mosaic = create_adaptive_mosaic(frames)
        if mosaic is not None:
            cv2.imshow('IP Camera Mosaic', mosaic)
        
        # Controle de FPS
        elapsed = (time.time() - loop_start) * 1000  # em milissegundos
        delay = max(1, int((1000 / TARGET_FPS) - elapsed))
        
        if cv2.waitKey(delay) == 27:
            break

finally:
    for thread in webcam_threads:
        thread.stop()
        thread.join()
    
    # Salva o último frame de cada câmera (se disponível)
    for idx, thread in enumerate(webcam_threads):
        if thread.frame is not None:
            cv2.imwrite(f"last_frame_{idx}.jpg", thread.frame)
    
    cv2.destroyAllWindows()
    print("Sistema encerrado com segurança.")
