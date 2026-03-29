# Mudanças
- Modificar a função `executar_sistema_dogflush` no arquivo `DogFlush_Entregavel.ipynb` para incluir configurações de propriedades do OpenCV (`cap.set`).
- Definir a largura do frame da câmera para 640 pixels (`cv2.CAP_PROP_FRAME_WIDTH`).
- Definir a altura do frame para 480 pixels (`cv2.CAP_PROP_FRAME_HEIGHT`).
- Limitar o FPS (Frames Por Segundo) para 15 (`cv2.CAP_PROP_FPS`). Isso alivia a carga de processamento da CPU e padroniza a variável `LIMITE_FRAMES` da Máquina de Estados para funcionar igual em qualquer webcam.

# Verificação
- Executar a célula atualizada no Jupyter Notebook.
- Observar se o fluxo do vídeo ficou mais leve e uniforme e se o sistema de detecção mantém a consistência independentemente da câmera utilizada.