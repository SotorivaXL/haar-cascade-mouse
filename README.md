# Treinamento Automático de Classificador Haar/LBP (OpenCV)

*Versão otimizada – 2025‑05*

Este projeto fornece uma pipeline automatizada para treinamento de classificadores Haar ou LBP utilizando os utilitários opencv_createsamples e opencv_traincascade. Ideal para quem deseja criar detectores personalizados de objetos com o OpenCV.

---

## Funcionalidades

- Anotação interativa de imagens positivas.
- Geração automática da lista de imagens negativas (bg.txt).
- Criação do arquivo .vec com aumento de dados.
- Treinamento completo com parâmetros configuráveis.
- Saída final no formato cascade.xml pronto para uso.

## Objetivo

Facilitar o processo de treinamento de classificadores personalizados para detecção de objetos com OpenCV, sem exigir conhecimentos avançados de linha de comando ou scripts fragmentados. Ideal para pesquisadores, estudantes ou profissionais que
desejam treinar detectores para objetos específicos como:

- Ferramentas manuais
- Equipamentos eletrônicos
- Mãos ou gestos
- Produtos em linhas de montagem
- Etc.

---

## Funcionalidades

-  Anotação assistida (interface GUI via OpenCV)
-  Geração automática de arquivos .txt para positivos e negativos
-  Criação de arquivos .vec com aumento de dados
-  Treinamento completo via opencv_traincascade
-  Controle de parâmetros por linha de comando

---

## Requisitos

| Recurso                     | Detalhes                                 |
|----------------------------|------------------------------------------|
| Python                     | 3.7+                                     |
| OpenCV                     | opencv-python e OpenCV com binários C++|
| SO recomendado             | Windows (ajustável para Linux/macOS)     |
| Executáveis necessários    | opencv_createsamples.exe, opencv_traincascade.exe |

---

## Estrutura de diretórios

.
├── auto_pipeline.py            # script principal para treino do modelo
├── detect_custom.py            # script principal para execução de testes de detecção
├── dataset/
│   ├── positives/              # imagens com o objeto desejado
│   └── negatives/              # imagens sem o objeto
├── annotations/                # arquivos gerados automaticamente (.txt, .vec)
└── training/                   # diretório de saída com cascade.xml
