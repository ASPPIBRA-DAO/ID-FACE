import face_recognition
import cv2

def detectar_faces(imagem_path):
    try:
        # Carregar a imagem
        image = face_recognition.load_image_file(imagem_path)
        
        # Encontrar todas as faces na imagem
        face_locations = face_recognition.face_locations(image, model='hog')  # Ou 'cnn' para mais precisão
        
        # Contar e exibir o número de rostos detectados
        num_faces = len(face_locations)
        print(f'Número de rostos detectados: {num_faces}')

        # Exibir coordenadas das faces detectadas
        for i, (top, right, bottom, left) in enumerate(face_locations, 1):
            print(f'Rosto {i}: Topo={top}, Direita={right}, Baixo={bottom}, Esquerda={left}')

        # Desenhar retângulos ao redor dos rostos usando OpenCV
        image_cv = cv2.imread(imagem_path)
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)

        # Exibir imagem com os rostos detectados
        cv2.imshow('Rostos Detectados', image_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f'Erro ao processar a imagem: {e}')

# Solicitar ao usuário o caminho da imagem
imagem_path = input("Digite o caminho da imagem: ")
detectar_faces(imagem_path)
