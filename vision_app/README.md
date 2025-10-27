
# Projeto Vision App

## 1. Visão Geral

Este é um projeto de visão computacional estruturado para separar uma biblioteca de lógica reutilizável de uma aplicação de frontend. Ele consiste em dois componentes principais:

*   **`vision_library`**: Uma biblioteca Python modular e desacoplada que lida com todas as tarefas de processamento de imagem e vídeo, como contagem de pessoas e reconhecimento facial.
*   **`app_frontend`**: Um diretório placeholder destinado a abrigar o código da aplicação do usuário final (ex: um app mobile, web ou desktop) que consome a `vision_library`.

### Funcionalidades da Biblioteca

*   **Contagem de Pessoas:** Conta objetos que cruzam uma Região de Interesse (ROI).
*   **Detecção de Faces:** Localiza rostos em imagens.
*   **Reconhecimento de Faces:** Compara e verifica a identidade de rostos.

### Qualidade de Código e Boas Práticas

*   **Tipagem Estática (`Type Hinting`):** Todo o código na `vision_library` é 100% tipado. Isso significa que todas as funções e métodos têm suas entradas e saídas claramente definidas, o que ajuda a prevenir bugs e melhora a experiência de desenvolvimento com autocompletar e análise estática.
*   **Documentação Completa:** Além deste `README`, todo o código possui docstrings detalhadas que explicam o que cada função faz, seus parâmetros e o que ela retorna.

---

## 2. Estrutura do Projeto

A raiz do projeto é o diretório `vision_app`, que tem uma estrutura clara para promover a separação de responsabilidades:

```
vision_app/
│
├── app_frontend/         # (Placeholder) O código da sua UI/aplicativo vai aqui.
│
├── data/                 # Contém imagens e vídeos para teste.
│
├── vision_library/       # A biblioteca de visão computacional (o "motor").
│   ├── __init__.py       # Fachada da biblioteca, facilita as importações.
│   ├── config.py         # Configurações centralizadas (caminhos, limiares).
│   ├── utils.py          # Funções de utilidade (carregar mídias).
│   ├── people_counting.py  # Lógica para contagem de pessoas.
│   ├── face_detection.py   # Lógica para detecção de rostos.
│   └── face_recognition.py # Lógica para reconhecimento de rostos.
│
├── README.md             # Esta documentação.
│
└── requirements.txt      # Dependências Python para a vision_library.
```

---

## 3. Configuração do Ambiente de Desenvolvimento

A configuração de um ambiente de visão computacional pode ser complexa, pois envolve dependências tanto do sistema operacional quanto do Python. Siga estes passos cuidadosamente.

Os comandos devem ser executados a partir do diretório raiz `vision_app/`.

### Passo 1: Instalar Pré-requisitos do Sistema (CMake)

A biblioteca `face-recognition` depende do pacote `dlib`, que precisa ser compilado durante a instalação. Este processo de compilação requer a ferramenta de sistema **`CMake`**.

**A. Para sistemas baseados em Debian/Ubuntu (como o ambiente deste IDE):**

Use o `apt` para instalar o `cmake`.

```bash
# Atualiza a lista de pacotes do seu sistema
sudo apt-get update

# Instala o cmake
sudo apt-get install -y cmake
```

**B. Para outros sistemas (macOS, Windows, etc.):**

*   **macOS:** Use o [Homebrew](https://brew.sh/): `brew install cmake`
*   **Windows:** Baixe o instalador do [site oficial do CMake](https://cmake.org/download/) e certifique-se de marcar a opção "Add CMake to the system PATH".

### Passo 2: Criar um Ambiente Python Isolado

É uma **forte recomendação** usar um ambiente virtual para isolar as dependências deste projeto e evitar conflitos com outros pacotes Python no seu sistema.

```bash
# Cria um diretório de ambiente virtual chamado 'venv'
python -m venv venv

# Ativa o ambiente virtual
# No Linux ou macOS:
source venv/bin/activate

# No Windows (use Git Bash ou PowerShell):
# venv\Scripts\activate
```

### Passo 3: Instalar as Dependências Python

Com o ambiente virtual ativado (você deve ver `(venv)` no seu prompt do terminal), instale todas as bibliotecas Python necessárias com um único comando:

```bash
# Instala opencv, dlib, face-recognition, etc.
pip install -r requirements.txt
```

Se todos os passos foram seguidos corretamente, o ambiente estará pronto para rodar a aplicação.


---

## 4. Configuração

Todas as configurações são centralizadas em `vision_library/config.py`. Os caminhos para os arquivos de mídia são relativos à pasta raiz `vision_app`.

**Exemplo de `config.py`:**
```python
# Configurações para Reconhecimento Facial
FACE_COMPARISON = {
    "reference_image": "data/raw/images/elon01.jpg",
    "test_image": "data/raw/images/elon_test.jpg"
}

# Configurações para Contagem de Pessoas
PEOPLE_COUNTING = {
    "video_path": "data/raw/videos/escalator.mp4",
    # ... outros parâmetros
}
```

---

## 5. Guia de Uso da Biblioteca (`vision_library`)

Para usar a biblioteca em sua aplicação, você pode importar os módulos diretamente da `vision_library`.

### Exemplo 1: Contagem de Pessoas

```python
import cv2
from vision_library import PeopleCounter, utils, config

# 1. Carrega as configurações
cfg = config.PEOPLE_COUNTING
video = utils.load_video(cfg["video_path"])

# 2. Inicializa o contador
counter = PeopleCounter()

# (Loop de processamento de vídeo omitido para brevidade)
# ... dentro do loop ...
# processed_frame, count = counter.process_frame(frame)
```

### Exemplo 2: Reconhecimento Facial

```python
from vision_library import face_detection, face_recognition, config

# 1. Carrega as configurações
cfg = config.FACE_COMPARISON

# 2. Encontra o rosto de referência
ref_img, ref_locs = face_detection.find_faces(cfg["reference_image"])
if not ref_locs:
    raise SystemExit("Rosto de referência não encontrado.")

# 3. Gera o encoding do rosto
ref_encoding = face_recognition.get_face_encodings(ref_img, ref_locs)[0]

# (Resto da lógica de comparação omitida)
```

---

## 6. Como Testar Módulos Individualmente

Para verificar a funcionalidade de um módulo da biblioteca de forma isolada, execute-o como um script a partir do diretório raiz `vision_app/`. O Python se encarregará de encontrar o módulo dentro do pacote `vision_library`.

**Exemplo: Testando o módulo de contagem de pessoas:**

```bash
# Estando no diretório vision_app/
python -m vision_library.people_counting
```

**Exemplo: Testando o módulo de reconhecimento facial:**
```bash
# Estando no diretório vision_app/
python -m vision_library.face_recognition
```
