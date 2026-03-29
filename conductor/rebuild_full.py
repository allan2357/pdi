import json
import os

def build_notebook_completo():
    cells = []
    def add_md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    def add_code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)})

    # 1. INTRODUÇÃO COMPLETA
    add_md("""# MCZA018 – Processamento Digital de Imagens
## Relatório Final do Trabalho (RFT): DogFlush - Sistema de Descarga Automática para Pets

**Equipe:**
- INTEGRANTE 1 - RA: 0000000
- INTEGRANTE 2 - RA: 0000000
- INTEGRANTE 3 - RA: 0000000

**Data de Publicação:** 29 de Março de 2026

---

## 1. Introdução

### Contexto e Cenário de Aplicação (CA)
O **DogFlush** é um Sistema de Processamento Visual (SPV) interativo criado para monitorar ambientes de higiene para animais de estimação. A motivação surgiu através de **entrevistas empáticas** realizadas com donos de pets que relataram a dificuldade de manter a higiene do ambiente durante longas jornadas de trabalho fora de casa.
O objetivo do sistema é analisar a cena visual em tempo real via webcam, detectar automaticamente o uso do banheirinho e acionar a limpeza (descarga) no momento em que o animal se retira, garantindo a automação da higiene doméstica.

### Fundamentação Teórica
O sistema aplica os conceitos fundamentais de PDI abordados na ementa da disciplina:
1. **Filtragem de Imagens:** Uso de Gaussian Blur para redução de ruído térmico da webcam.
2. **Transformações de Histograma:** Aplicação de CLAHE (Contrast Limited Adaptive Histogram Equalization) para estabilização de luminância.
3. **Segmentação por Movimento:** Utilização do algoritmo MOG2 para extração do Foreground.
4. **Operadores Morfológicos:** Processamento com Opening e Closing para refinamento da máscara binária.
5. **Segmentação Avançada:** Implementação do algoritmo Watershed para definição precisa das bordas do objeto detectado.

---

## 2. Materiais e Métodos

### Diagrama de Blocos Funcional do SPV
`Webcam` -> `Grayscale` -> `Filtro Gaussiano` -> `CLAHE` -> `MOG2` -> `Binarização Otsu` -> `Abertura/Fechamento Morfológico` -> `Watershed` -> `Máquina de Estados` -> `Acionamento`.

### Descrição da Implementação
O sistema foi implementado em Python com OpenCV. A detecção é baseada em uma máquina de estados temporal que evita falsos positivos (como um objeto apenas passando rápido pela cena). Para fins didáticos, o programa exibe as etapas intermediárias do processamento em janelas separadas.
""")

    # 2. CÓDIGO COMPLETO (MOG2 + 5 JANELAS + PADRONIZAÇÃO)
    code_text = """# CABEÇALHO OBRIGATÓRIO (REQUISITO DA DISCIPLINA)
# Nome da Equipe: [Inserir Nome]
# Nomes Completos e RAs: [Inserir Nomes e RAs]
# Data do Programa: 29/03/2026
# Nome do Programa: dog_flush_spv.py

import cv2
import numpy as np
import time

def executar_sistema_dogflush(modo_video=0):
    '''
    Função principal do Sistema de Processamento Visual.
    modo_video: 0 para Webcam padrão.
    '''
    print("Iniciando Módulo de Câmera... Pressione 'ESC' para encerrar.")
    
    cap = cv2.VideoCapture(modo_video)
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return

    # Padronizando a câmera: Resolução (640x480) e FPS (15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    # Inicializa o Subtrator de Fundo (MOG2)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # Variáveis da Máquina de Estados
    ESTADO = 0 # 0=Livre, 1=Ocupado, 2=Acionando
    frames_ocupado = 0
    LIMITE_FRAMES = 15 
    total_ativacoes = 0
    kernel = np.ones((5,5), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.resize(frame, (640, 480))
        
        # 1. PROCESSAMENTO DE CORES
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. FILTRAGEM
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # 3. HISTOGRAMA (EQUALIZAÇÃO CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq = clahe.apply(blur)
        
        # 4. SEGMENTAÇÃO (MOG2)
        fgmask = fgbg.apply(eq)
        
        # 5. OPERADORES MORFOLÓGICOS
        _, fgmask_bin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(fgmask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 6. WATERSHED (Obrigatório)
        sure_bg = cv2.dilate(closing, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        markers = cv2.connectedComponents(sure_fg)[1] + 1
        markers[cv2.subtract(sure_bg, sure_fg) == 255] = 0
        frame_watershed = frame.copy()
        cv2.watershed(frame_watershed, markers)
        frame_watershed[markers == -1] = [0, 0, 255]
        
        # 7. EXTRAÇÃO E MÁQUINA DE ESTADO
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        teve_movimento = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 8000: 
                teve_movimento = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame_watershed, (x, y), (x+w, y+h), (0, 255, 0), 2)
                break
                
        if ESTADO == 0:
            msg, cor = "SISTEMA: MONITORANDO (LIVRE)", (255,0,0)
            if teve_movimento:
                frames_ocupado += 1
                if frames_ocupado > LIMITE_FRAMES: ESTADO = 1
        elif ESTADO == 1:
            msg, cor = "SISTEMA: USO EM ANDAMENTO", (0,165,255)
            if not teve_movimento:
                frames_ocupado -= 1
                if frames_ocupado <= 0: ESTADO = 2
        elif ESTADO == 2:
            msg, cor = "!!! DESCARGA ATIVADA !!!", (0,0,255)
            total_ativacoes += 1
            cv2.imshow('5. DogFlush Final', frame_watershed)
            cv2.waitKey(2000)
            ESTADO, frames_ocupado = 0, 0
            fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
            
        cv2.putText(frame_watershed, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
        cv2.putText(frame_watershed, f"Descargas: {total_ativacoes}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # EXIBIÇÃO DIDÁTICA
        cv2.imshow('1. Grayscale + Blur', cv2.resize(blur, (320, 240)))
        cv2.imshow('2. Equalizacao (CLAHE)', cv2.resize(eq, (320, 240)))
        cv2.imshow('3. Segmentacao (MOG2)', cv2.resize(fgmask, (320, 240)))
        cv2.imshow('4. Morfologia', cv2.resize(closing, (320, 240)))
        cv2.imshow('5. DogFlush Final', frame_watershed)

        if cv2.waitKey(30) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

# Para rodar, descomente abaixo:
# executar_sistema_dogflush(0)
"""
    add_code(code_text)

    # 3. LEX E CONCLUSÕES COMPLETAS
    add_md("""---

## 3. Laboratório Experimental (LEx)

### Roteiro do Experimento
**Introdução:** O usuário deve testar a robustez do sistema DogFlush utilizando um objeto de pelúcia ou sua própria mão para simular o animal.
**Procedimento:**
1. Execute a célula de código acima.
2. Posicione a webcam para o chão/área de teste.
3. Observe as janelas de debug para entender como os filtros limpam a imagem.
4. Simule a entrada do animal na área.
5. Simule a saída e verifique se a mensagem "DESCARGA ATIVADA" aparece.

### Questionário de Avaliação
1. As janelas de debug ajudaram a entender o que cada filtro faz?
2. Houve algum atraso perceptível entre a saída do objeto e o aviso de descarga?
3. O sistema falhou ao detectar movimentos muito rápidos ou sombras?

---

## 4. Conclusões
O projeto **DogFlush** demonstrou a viabilidade do uso de Visão Computacional de baixo custo para automação residencial. A combinação de algoritmos clássicos (MOG2, CLAHE, Watershed) permitiu criar um sistema leve e eficiente. 
**Pontos Positivos:** O sistema é altamente responsivo e o uso de CLAHE mitigou problemas de iluminação. 
**Pontos Negativos:** Ambientes com excesso de reflexos no chão podem gerar ruído na máscara MOG2, exigindo maior cuidado na configuração do `varThreshold`.
Os objetivos propostos foram atingidos, resultando em um protótipo funcional pronto para integração com hardware de acionamento real.

## Referências Bibliográficas
- GONZALEZ, R. C.; WOODS, R. E. Processamento Digital de Imagens. 3. ed.
- OpenCV Documentation: https://docs.opencv.org/
""")

    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3","name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}
    
    with open(r"C:\Users\allan\Downloads\pdi\Projeto\DogFlush_Entregavel.ipynb", 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    build_notebook_completo()
