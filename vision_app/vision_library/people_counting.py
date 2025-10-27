
import cv2
import numpy as np
from typing import Tuple
from . import utils, config

class PeopleCounter:
    """Processa quadros de vídeo para contar pessoas que cruzam uma região de interesse (ROI)."""
    
    def __init__(self) -> None:
        """Inicializa o contador com as configurações do projeto."""
        cfg = config.PEOPLE_COUNTING
        self.roi_coords: Tuple[int, int, int, int] = cfg["roi_coords"]
        self.threshold: int = cfg["threshold"]
        self.font: int = eval(cfg.get("font", "cv2.FONT_HERSHEY_SIMPLEX"))
        
        self.contador: int = 0
        self.liberado: bool = True

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Aplica pré-processamento ao quadro para análise."""
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 12)
        kernel = np.ones((8, 8), np.uint8)
        return cv2.dilate(img_th, kernel, iterations=2)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Processa um único quadro de vídeo, atualiza a contagem e retorna o quadro com anotações.
        
        Args:
            frame (np.ndarray): O quadro de vídeo a ser processado.
            
        Returns:
            Tuple[np.ndarray, int]: Uma tupla contendo o quadro processado com anotações e a contagem atual.
        """
        x, y, w, h = self.roi_coords
        frame_processed = frame.copy()
        
        img_dil = self._preprocess(frame)
        
        recorte = img_dil[y:y+h, x:x+w]
        brancos = cv2.countNonZero(recorte)
        
        if brancos > self.threshold and self.liberado:
            self.contador += 1
            self.liberado = False
        elif brancos < self.threshold:
            self.liberado = True
        
        cor = (0, 255, 0) if not self.liberado else (255, 0, 255)
        cv2.rectangle(frame_processed, (x, y), (x + w, y + h), cor, 4)
        cv2.putText(frame_processed, f"Count: {self.contador}", (x, y - 10), self.font, 1, (255, 0, 0), 3)

        return frame_processed, self.contador

# Bloco de teste permanece o mesmo
if __name__ == '__main__':
    cfg = config.PEOPLE_COUNTING
    video = utils.load_video(cfg["video_path"])
    
    if video:
        counter = PeopleCounter()
        
        while True:
            ret, frame = video.read()
            if not ret:
                print("Fim do vídeo ou erro na leitura.")
                break
            
            frame = cv2.resize(frame, (1100, 720))
            processed_frame, count = counter.process_frame(frame)
            
            cv2.imshow('Teste do PeopleCounter', processed_frame)
            print(f"Contagem atual: {count}")

            if cv2.waitKey(20) & 0xFF == 27:
                break
        
        video.release()
        cv2.destroyAllWindows()
