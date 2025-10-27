
"""
Este pacote contém a lógica principal para a aplicação de visão computacional.

Ele expõe as seguintes classes e funções para serem usadas por interfaces externas:

- PeopleCounter: Uma classe para contar pessoas em um fluxo de vídeo.
- face_detection: Um módulo para encontrar rostos em imagens.
- face_recognition: Um módulo para comparar e reconhecer rostos.
- utils: Funções de utilidade, como carregar mídias.
- config: Módulo de configuração para acesso a parâmetros.
"""

# Importa as principais classes e módulos para o nível do pacote
from .people_counting import PeopleCounter
from . import face_detection
from . import face_recognition
from . import utils
from . import config

# Define o que é exportado quando se usa 'from meu_projeto.src import *'
__all__ = [
    "PeopleCounter",
    "face_detection",
    "face_recognition",
    "utils",
    "config"
]
