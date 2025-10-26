import cv2
import numpy as np

# Carregar vídeo
video_path = '/home/sandro/Downloads/id face/meu_projeto/data/raw/videos/escalator.mp4'
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print(f"Erro ao abrir o vídeo: {video_path}")
    exit()

# Variáveis de controle
contador = 0
liberado = False
x, y, w, h = 490, 230, 30, 150  # Coordenadas da região de interesse

def processar_frame(frame):
    """ Aplica pré-processamento ao frame. """
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img_th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 12)
    kernel = np.ones((8, 8), np.uint8)
    return cv2.dilate(img_th, kernel, iterations=2)

while True:
    ret, img = video.read()
    if not ret:
        print("Fim do vídeo ou erro na leitura.")
        break
    
    img = cv2.resize(img, (1100, 720))
    img_dil = processar_frame(img)
    
    # Recortar a região de interesse
    recorte = img_dil[y:y+h, x:x+w]
    brancos = cv2.countNonZero(recorte)
    
    # Lógica de contagem
    if brancos > 4000 and liberado:
        contador += 1
        liberado = False
    elif brancos < 4000:
        liberado = True
    
    # Desenhar retângulos
    cor = (0, 255, 0) if not liberado else (255, 0, 255)
    cv2.rectangle(img, (x, y), (x + w, y + h), cor, 4)
    
    # Exibir informações no vídeo
    cv2.putText(img, str(brancos), (x-30, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.rectangle(img, (575, 155), (663, 240), (255, 255, 255), -1)
    cv2.putText(img, str(contador), (x+100, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
    
    # Exibir frame
    cv2.imshow('Video', img)
    
    # Controle de saída
    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # Pressionar ESC para sair
        break

video.release()
cv2.destroyAllWindows()