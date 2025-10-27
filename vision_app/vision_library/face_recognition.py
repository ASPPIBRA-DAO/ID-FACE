
import cv2
import numpy as np
import face_recognition as fr
from typing import List, Dict, Any

def get_face_encodings(image: np.ndarray, locations: List[tuple]) -> List[np.ndarray]:
    """Calcula os encodings para os rostos encontrados em uma imagem.

    Args:
        image (np.ndarray): A imagem (como array NumPy) contendo os rostos.
        locations (List[tuple]): Uma lista de coordenadas (top, right, bottom, left) para cada rosto.

    Returns:
        List[np.ndarray]: Uma lista de arrays NumPy, onde cada array é o encoding de 128 dimensões de um rosto.
    """
    return fr.face_encodings(image, locations)

def compare_faces(reference_encoding: np.ndarray, test_image: np.ndarray, test_locations: List[tuple]) -> List[Dict[str, Any]]:
    """
    Compara um encoding de referência com todos os rostos em uma imagem de teste.

    Args:
        reference_encoding (np.ndarray): O encoding do rosto de referência.
        test_image (np.ndarray): A imagem de teste (array NumPy).
        test_locations (List[tuple]): As localizações dos rostos na imagem de teste.

    Returns:
        List[Dict[str, Any]]: Uma lista de dicionários, um para cada rosto de teste. Cada dicionário contém:
            - "location": A tupla de coordenadas do rosto.
            - "is_match": Um booleano indicando se corresponde à referência.
            - "distance": A distância facial (quanto menor, mais similar).
    """
    test_encodings = get_face_encodings(test_image, test_locations)
    
    results = []
    for i, test_encoding in enumerate(test_encodings):
        # compare_faces retorna uma lista de booleanos
        is_match = fr.compare_faces([reference_encoding], test_encoding)[0]
        # face_distance retorna a distância numérica
        distance = fr.face_distance([reference_encoding], test_encoding)[0]
        results.append({
            "location": test_locations[i],
            "is_match": bool(is_match), # Garante que seja um booleano Python nativo
            "distance": float(distance), # Garante que seja um float
        })
    return results

def draw_recognition_results(image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
    """
    Desenha os resultados do reconhecimento (match/distância) em uma imagem.

    Args:
        image (np.ndarray): A imagem (array NumPy) na qual desenhar.
        results (List[Dict[str, Any]]): A lista de resultados da função compare_faces.

    Returns:
        np.ndarray: Uma cópia da imagem com as anotações de reconhecimento.
    """
    img_with_results = image.copy()
    for result in results:
        top, right, bottom, left = result["location"]
        is_match: bool = result["is_match"]
        distance: float = result["distance"]
        
        # Cor verde para match, vermelha para não match
        color = (0, 255, 0) if is_match else (0, 0, 255)
        label = f"Match: {is_match} (Dist: {distance:.2f})"
        
        # Desenha o retângulo e o texto
        cv2.rectangle(img_with_results, (left, top), (right, bottom), color, 2)
        cv2.putText(img_with_results, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
    return img_with_results
