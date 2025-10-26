import cv2

# Substitua a URL pela URL do seu stream de vídeo da câmera IP
url = "http://192.168.1.4:8080/video"

# Carregar o classificador de face pré-treinado do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Abertura do vídeo da câmera IP
webcam = cv2.VideoCapture(url)

if webcam.isOpened():
    validacao, frame = webcam.read()
    while validacao:
        validacao, frame = webcam.read()
        
        # Converter para escala de cinza para melhorar a detecção
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar faces na imagem
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Desenhar retângulos ao redor das faces detectadas
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Video da Webcam", frame)
        key = cv2.waitKey(5)
        if key == 27:  # ESC
            break
    
    # Salvar a última imagem capturada com detecção de face
    cv2.imwrite("FotoLira.png", frame)

webcam.release()
cv2.destroyAllWindows()
