import face_recognition as fr
import os
import matplotlib.pyplot as plt
import cv2  # OpenCV para desenhar retângulos e texto

# Caminhos das imagens (idealmente, usar caminhos relativos ou de um arquivo de configuração)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw', 'images')
elon_img_path = os.path.join(DATA_DIR, "elon01.jpg") # Imagem de referência
elon_test_img_path = os.path.join(DATA_DIR, "elon_test.jpg") # Imagem a ser testada

# Verificar se os arquivos existem
if not os.path.exists(elon_img_path):
    print(f"Erro: Arquivo {elon_img_path} não encontrado.")
    exit()
if not os.path.exists(elon_test_img_path):
    print(f"Erro: Arquivo {elon_test_img_path} não encontrado.")
    exit()

# Carregar imagens e converter para RGB
imgElon = fr.load_image_file(elon_img_path)
imgElonTest = fr.load_image_file(elon_test_img_path)

# Detectar rostos e obter codificações
faceLocsElon = fr.face_locations(imgElon)
faceLocsTest = fr.face_locations(imgElonTest)
encodesElon = fr.face_encodings(imgElon)
encodesTest = fr.face_encodings(imgElonTest)

# Verificar se há rostos detectados antes de continuar
if len(encodesElon) == 0 or len(encodesTest) == 0:
    print("Erro: Nenhum rosto detectado em uma ou ambas as imagens.")
    exit()

# Assumindo que a imagem de treinamento (elon_img_path) tem apenas uma face de referência.
encodeElonRef = encodesElon[0]

# Iterar sobre cada rosto detectado (faceLocsTest) e sua codificação (encodesTest) na imagem de teste
for encodeTest, faceLocTest in zip(encodesTest, faceLocsTest):
    # Compara a face de referência com a face de teste atual
    comparacoes = fr.compare_faces([encodeElonRef], encodeTest)
    distancia = fr.face_distance([encodeElonRef], encodeTest)[0] # Pega o primeiro valor de distância
    
    # Exibir resultados no console
    print(f"Rosto em {faceLocTest}: Similar = {comparacoes[0]}, Distância = {distancia:.2f}")
    
    # Desenhar retângulos e exibir informações na imagem de teste
    top, right, bottom, left = faceLocTest
    # Desenhar retângulo verde ao redor do rosto
    cv2.rectangle(imgElonTest, (left, top), (right, bottom), (0, 255, 0), 2)
    # Colocar texto com a distância
    cv2.putText(imgElonTest, f"Dist: {distancia:.2f}", (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Configurar as imagens lado a lado
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Exibir as imagens
axes[0].imshow(imgElon)
axes[0].set_title("Imagem de Treinamento")
axes[0].axis("off")  # Desativa os eixos para exibição limpa

axes[1].imshow(imgElonTest)
axes[1].set_title("Imagem de Teste")
axes[1].axis("off")  # Desativa os eixos para exibição limpa

# Ajustar o layout para que as imagens fiquem centralizadas
plt.tight_layout()
plt.show()
