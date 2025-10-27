
import cv2
import numpy as np
import face_recognition as fr
from typing import Optional

def load_image(path: str) -> Optional[np.ndarray]:
    """Carrega uma imagem de um arquivo e a converte para o formato RGB.

    Args:
        path (str): O caminho para o arquivo de imagem.

    Returns:
        Optional[np.ndarray]: Uma matriz numpy representando a imagem em RGB, ou None se o arquivo não for encontrado.
    """
    try:
        img = fr.load_image_file(path)
        return img
    except FileNotFoundError:
        print(f"Erro: Arquivo de imagem não encontrado em {path}")
        return None

def load_video(path: str) -> Optional[cv2.VideoCapture]:
    """Carrega um vídeo de um arquivo.

    Args:
        path (str): O caminho para o arquivo de vídeo.

    Returns:
        Optional[cv2.VideoCapture]: Um objeto VideoCapture do OpenCV, ou None se o vídeo não puder ser aberto.
    """
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        print(f"Erro ao abrir o vídeo: {path}")
        return None
    return video
