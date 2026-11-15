import os
import time
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyautogui

# Configuração de segurança do PyAudio/PyAutoGUI (opcional, evita falhas bruscas se mover o mouse para o canto)
pyautogui.FAILSAFE = False

# Carrega o modelo treinado
MODEL_PATH = "hand_model.pkl"
TASK_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
  raise FileNotFoundError(
      "Modelo 'hand_model.pkl' não encontrado. Execute o 'train_model.py'"
      " primeiro!"
  )

modelo = joblib.load(MODEL_PATH)
print("Modelo carregado com sucesso!")

# Configurações da Câmera
CAMERA_INDEX = 0
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
  raise RuntimeError("Não foi possível abrir a webcam.")

# Configuração do MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Conexões dos dedos para visualização
conexoes = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

inicio = time.monotonic()
ultimo_timestamp = 0
gesto_anterior = None

print("\nIniciando controle por gestos...")
print("Pressione 'Q' na janela do OpenCV para sair.")

with HandLandmarker.create_from_options(options) as detector:
  while True:
    sucesso, frame = camera.read()
    if not sucesso or frame is None:
      break

    # Espelha a imagem para ficar natural (como um espelho)
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp = int((time.monotonic() - inicio) * 1000)
    timestamp = max(timestamp, ultimo_timestamp + 1)
    ultimo_timestamp = timestamp

    resultado = detector.detect_for_video(imagem_mp, timestamp)

    texto_status = "Nenhuma mao detectada"

    if resultado.hand_landmarks:
      landmarks = resultado.hand_landmarks[0]
      altura, largura, _ = frame.shape

      # Desenha os pontos na tela
      pontos_tela = []
      for p in landmarks:
        px = int(p.x * largura)
        py = int(p.y * altura)
        pontos_tela.append((px, py))
        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

      for a, b in conexoes:
        cv2.line(frame, pontos_tela[a], pontos_tela[b], (255, 0, 0), 2)

      # Normaliza os dados (mesmo processo feito no collect_data.py)
      pulso = landmarks[0]
      escala = max(
          max(abs(p.x - pulso.x), abs(p.y - pulso.y)) for p in landmarks
      )
      escala = max(escala, 0.0001)

      dados = []
      for p in landmarks:
        x = (p.x - pulso.x) / escala
        y = (p.y - pulso.y) / escala
        z = (p.z - pulso.z) / escala
        dados.extend([x, y, z])

      # Faz a predição com o modelo treinado
      predicao = modelo.predict([dados])[0]
      probabilidades = modelo.predict_proba([dados])[0]
      confianca = max(probabilidades)

      # --- AÇÕES PARA O JOGO ---
      if confianca > 0.75:
        texto_status = f"Gesto: {predicao} ({confianca * 100:.1f}%)"

        # Mapeamento para o movimento do personagem:
        if predicao == "dois_dedos":
          # Mantém a seta para a direita pressionada continuamente
          pyautogui.keyDown("right")
          pyautogui.keyUp("left")
        elif predicao == "indicador":
          # Mantém a seta para a esquerda pressionada continuamente
          pyautogui.keyDown("left")
          pyautogui.keyUp("right")
        elif predicao == "aberta":
          # Salto
          if gesto_anterior != "aberta":
            pyautogui.press("space")
          pyautogui.keyUp("right")
          pyautogui.keyUp("left")
        else:
          # Se for outro gesto, solta as setas
          pyautogui.keyUp("right")
          pyautogui.keyUp("left")

        gesto_anterior = predicao
      else:
        texto_status = "Incerto..."
        # Segurança: solta as teclas se perder o rastreamento da mão
        pyautogui.keyUp("right")
        pyautogui.keyUp("left")

    # Exibe informações na janela do vídeo
    cv2.putText(
        frame,
        texto_status,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Hand Controller - Jogo", frame)

    # Fecha se apertar 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

camera.release()
cv2.destroyAllWindows()
print("Encerrado.")