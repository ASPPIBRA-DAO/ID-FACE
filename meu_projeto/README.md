# Documentação do Código de Comparação Facial

## Objetivo

Este script realiza a detecção de rostos em duas imagens e compara se esses rostos são semelhantes utilizando a biblioteca `face_recognition`. Ele também desenha retângulos ao redor dos rostos nas imagens e exibe as distâncias de similaridade entre os rostos, mostrando as imagens de treino e teste lado a lado.

---

## Requisitos

- **Python** 3.x
- Bibliotecas:
  - `face_recognition`
  - `matplotlib`
  - `opencv-python`

Instale as dependências com:

```bash
pip install face_recognition matplotlib opencv-python
```

Explicação do Código

1. Importações

```bash
import face_recognition as fr
import os
import matplotlib.pyplot as plt
import cv2  # OpenCV para desenhar retângulos e texto
```

face_recognition: Detecta rostos e codifica características faciais.

os: Interage com o sistema de arquivos.

matplotlib.pyplot: Exibe imagens lado a lado.

cv2: Desenha retângulos e textos nas imagens.

2. Caminhos das Imagens

```bash
elon_img_path = "/caminho/para/imagem/treino.jpg"
elon_test_img_path = "/caminho/para/imagem/teste.jpg"
```

Define os caminhos das imagens de treino e teste.

3. Verificação de Arquivos

```bash
if not os.path.exists(elon_img_path):
    print(f"Erro: Arquivo {elon_img_path} não encontrado.")
    exit()
```

Verifica se os arquivos existem. Encerra o programa se não forem encontrados.

4. Carregamento de Imagens

```bash
imgElon = fr.load_image_file(elon_img_path)
imgElonTest = fr.load_image_file(elon_test_img_path)
```

Carrega as imagens em formato RGB usando face_recognition.

5. Detecção de Rostos

```bash
faceLocsElon = fr.face_locations(imgElon)
encodesElon = fr.face_encodings(imgElon)
```

face_locations(): Detecta coordenadas dos rostos.

face_encodings(): Gera vetores numéricos das características faciais.

6. Validação de Rostos Detectados

```bash
if len(encodesElon) == 0 or len(encodesTest) == 0:
    print("Erro: Nenhum rosto detectado.")
    exit()
```

Garante que pelo menos um rosto foi detectado em cada imagem.

7. Comparação Facial

```bash
comparacao = fr.compare_faces(encodesElon, encodeTest)
distancia = fr.face_distance(encodesElon, encodeTest)
```

compare_faces(): Retorna True/False para correspondência.

face_distance(): Calcula a distância entre características (valores menores = maior similaridade).

8. Anotações nas Imagens

```bash
cv2.rectangle(imgElonTest, (left, top), (right, bottom), (0, 255, 0), 2)
cv2.putText(imgElonTest, f"Dist: {distancia[i]:.2f}", ...)
```

Desenha retângulos verdes ao redor dos rostos.

Adiciona texto com a distância de similaridade formatada.

9. Exibição dos Resultados

```bash
plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(imgElon)
axes[1].imshow(imgElonTest)
plt.tight_layout()
plt.show()
```

Exibe as imagens de treino e teste lado a lado usando matplotlib.

Conclusão
O código realiza detecção, comparação e visualização de rostos de forma eficiente. Possíveis melhorias incluem:

Validação de múltiplos rostos

Otimização do processamento de imagens

Conversão automática de cores para RGB
