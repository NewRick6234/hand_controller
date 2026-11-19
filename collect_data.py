import os
import time
import cv2
import mediapipe as mp
import pandas as pd

# CONFIGURACOES
CAMERA_INDEX = (
    0  # 0 para webcam integrada, 1 ou superior para câmeras USB externas
)
MODEL_PATH = "hand_landmarker.task"
DATASET_PATH = "dataset/gestures.csv"

INTERVALO = 0.15

GESTOS = {
    ord("1"): "aberta",
    ord("2"): "fechada",
    ord("3"): "apontar_esquerda",
    ord("4"): "apontar_direita",
}

NOMES_GESTOS = {
    "aberta": "Mao Aberta (Neutro)",
    "fechada": "Mao Fechada (Pulo)",
    "apontar_esquerda": "Apontar Esquerda",
    "apontar_direita": "Apontar Direita",
}

os.makedirs("dataset", exist_ok=True)

# CARREGA CONTAGEM ATUAL DE AMOSTRAS
contagens = {g: 0 for g in GESTOS.values()}
if os.path.exists(DATASET_PATH) and os.path.getsize(DATASET_PATH) > 0:
  try:
    df_temp = pd.read_csv(DATASET_PATH)
    if "label" in df_temp.columns:
      for k, v in df_temp["label"].value_counts().items():
        if k in contagens:
          contagens[k] = int(v)
  except Exception:
    pass

# COLUNAS DO CSV
colunas = ["label"]

for i in range(21):
  colunas.extend([f"x{i}", f"y{i}", f"z{i}"])

# VERIFICA MODELO
if not os.path.exists(MODEL_PATH):
  raise FileNotFoundError("Arquivo hand_landmarker.task nao encontrado!")

# CONFIGURA MEDIAPIPE TASKS
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ABRE A WEBCAM DO PC DIRETAMENTE
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
  raise RuntimeError(
      "Nao foi possivel abrir a webcam do PC. Verifique o indice da camera."
  )

# VARIAVEIS
gesto_ativo = None
ultimo_registro = 0
inicio = time.monotonic()
ultimo_timestamp = 0

# CONEXOES DOS DEDOS
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

print("=" * 60)
print("COLETA DE DADOS INICIADA (Webcam)")
print("1 - Mao aberta (Neutro / Parado)")
print("2 - Mao fechada (Pulo - Espaco)")
print("3 - Apontar para a Esquerda (Andar Esquerda)")
print("4 - Apontar para a Direita (Andar Direita)")
print("ESPACO - Pausar coleta")
print("Q - Sair e finalizar")
print("=" * 60)

with HandLandmarker.create_from_options(options) as detector:

  while True:
    sucesso, frame = camera.read()

    if not sucesso or frame is None:
      print("Erro ao capturar imagem.")
      break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp = int((time.monotonic() - inicio) * 1000)

    timestamp = max(timestamp, ultimo_timestamp + 1)
    ultimo_timestamp = timestamp

    resultado = detector.detect_for_video(imagem_mp, timestamp)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla in GESTOS:
      gesto_ativo = GESTOS[tecla]

    elif tecla == ord(" "):
      gesto_ativo = None

    if resultado.hand_landmarks:
      landmarks = resultado.hand_landmarks[0]

      # DESENHA OS PONTOS
      altura, largura, _ = frame.shape

      pontos_tela = []

      for p in landmarks:
        px = int(p.x * largura)
        py = int(p.y * altura)

        pontos_tela.append((px, py))

        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

      # DESENHA CONEXOES
      for a, b in conexoes:
        cv2.line(frame, pontos_tela[a], pontos_tela[b], (255, 0, 0), 2)

      # NORMALIZA COORDENADAS
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

      # COLETA
      agora = time.monotonic()

      if gesto_ativo and agora - ultimo_registro >= INTERVALO:
        linha = [gesto_ativo] + dados

        arquivo_existe = os.path.exists(DATASET_PATH) and (
            os.path.getsize(DATASET_PATH) > 0
        )

        df = pd.DataFrame([linha], columns=colunas)

        df.to_csv(
            DATASET_PATH,
            mode="a",
            header=not arquivo_existe,
            index=False,
        )

        ultimo_registro = agora
        contagens[gesto_ativo] += 1
        print(f"[{gesto_ativo}] Amostras: {contagens[gesto_ativo]}")

    # INFORMACOES NA TELA
    status = NOMES_GESTOS.get(gesto_ativo, "PAUSADO")
    cor_status = (0, 255, 0) if gesto_ativo else (0, 165, 255)

    cv2.putText(
        frame,
        f"Gesto Ativo: {status}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        cor_status,
        2,
    )

    info_counts = (
        f"Aberta: {contagens['aberta']} | "
        f"Fechada: {contagens['fechada']} | "
        f"Esq: {contagens['apontar_esquerda']} | "
        f"Dir: {contagens['apontar_direita']}"
    )
    cv2.putText(
        frame,
        info_counts,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "1:Aberta | 2:Fechada | 3:Esq | 4:Dir | ESPACO:Pausar | Q:Sair",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

    cv2.imshow("Hand Controller - Coleta", frame)

    if tecla == ord("q"):
      break

camera.release()
cv2.destroyAllWindows()

print("Coleta finalizada!")