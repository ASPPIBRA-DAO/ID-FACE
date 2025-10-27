
import cv2
import numpy as np
import face_recognition as fr
from typing import List, Tuple, Optional
from . import utils

def find_faces(image_path: str) -> Tuple[Optional[np.ndarray], List[tuple]]:
    """
    Encontra todos os rostos em uma imagem.

    Args:
        image_path (str): O caminho para o arquivo de imagem.

    Returns:
        Tuple[Optional[np.ndarray], List[tuple]]: Uma tupla contendo:
            - A imagem carregada como um array NumPy (ou None se não encontrada).
            - Uma lista de tuplas com as coordenadas dos rostos.
    """
    img = utils.load_image(image_path)
    if img is None:
        return None, []
    
    # A biblioteca face_recognition retorna uma lista de tuplas (top, right, bottom, left)
    face_locations: List[tuple] = fr.face_locations(img)
    return img, face_locations

def draw_face_locations(image: np.ndarray, locations: List[tuple]) -> np.ndarray:
    """
    Desenha retângulos ao redor dos rostos em uma imagem.

    Args:
        image (np.ndarray): A imagem (array NumPy) na qual desenhar.
        locations (List[tuple]): Uma lista de tuplas com as coordenadas (top, right, bottom, left).

    Returns:
        np.ndarray: Uma cópia da imagem com os retângulos desenhados.
    """
    img_with_boxes = image.copy()
    for (top, right, bottom, left) in locations:
        # O formato do face_locations é (top, right, bottom, left)
        cv2.rectangle(img_with_boxes, (left, top), (right, bottom), (0, 0, 255), 2)
    return img_with_boxes
