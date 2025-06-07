import cv2
from ultralytics import YOLO
import numpy as np
from sklearn.cluster import KMeans

video_path = "videos/video.mp4"
TAMANHO_MIN_BOLA = 5
N_FRAMES_CLUSTER = 10

model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture(video_path)
posse = {"Equipa 1": 0, "Equipa 2": 0}
cores_buffer = []
frame_count = 0
mapa_clusters = None
kmeans_base = None


def obter_cor_média(bbox, frame):
    x1, y1, x2, y2 = map(int, bbox)
    jogador_crop = frame[y1:y2, x1:x2]
    if jogador_crop.size == 0:
        return None
    hsv = cv2.cvtColor(jogador_crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mascara = (s > 50) & (v > 50)
    if np.count_nonzero(mascara) == 0:
        return None
    h_m = h[mascara].mean()
    s_m = s[mascara].mean()
    v_m = v[mascara].mean()
    return (h_m, s_m, v_m)


def identificar_posse(frame, results):
    global cores_buffer, frame_count, mapa_clusters, kmeans_base
    jogadores_cores, jogadores_bbox = [], []
    bola = None

    for box in results.boxes:
        cls = int(box.cls[0])
        bbox = box.xyxy[0].cpu().numpy()

        if cls == 0:  # Pessoa
            cor = obter_cor_média(bbox, frame)
            if cor is None:
                continue
            jogadores_cores.append(cor)
            jogadores_bbox.append(bbox)

        elif cls == 32:  # Bola
            largura = bbox[2] - bbox[0]
            altura = bbox[3] - bbox[1]
            if largura < TAMANHO_MIN_BOLA or altura < TAMANHO_MIN_BOLA:
                continue
            bola = bbox

    if len(jogadores_cores) < 2:
        return None

    cores_np = np.array(jogadores_cores)[:, :2]

    if frame_count < N_FRAMES_CLUSTER:
        cores_buffer.extend(cores_np)
        frame_count += 1
        if frame_count == N_FRAMES_CLUSTER:
            kmeans_base = KMeans(n_clusters=2, random_state=0).fit(cores_buffer)
            centros = kmeans_base.cluster_centers_
            if centros[0][0] < centros[1][0]:
                mapa_clusters = {0: "Equipa 1", 1: "Equipa 2"}
            else:
                mapa_clusters = {0: "Equipa 2", 1: "Equipa 1"}
        return None

    if kmeans_base is None:
        return None

    labels = kmeans_base.predict(cores_np)

    for i, bbox in enumerate(jogadores_bbox):
        nome = mapa_clusters[labels[i]]
        cor_texto = (0, 0, 255) if nome == "Equipa 1" else (255, 0, 0)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), cor_texto, 2)
        cv2.putText(frame, nome, (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_texto, 1)

    if bola is None:
        return None

    def dist(bbox):
        xj, yj = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        xb, yb = (bola[0] + bola[2]) / 2, (bola[1] + bola[3]) / 2
        return np.sqrt((xj - xb) ** 2 + (yj - yb) ** 2)

    idx_mais_proximo = np.argmin([dist(bbox) for bbox in jogadores_bbox])
    equipa_idx = labels[idx_mais_proximo]
    equipa_nome = mapa_clusters[equipa_idx]

    cv2.rectangle(frame, (int(bola[0]), int(bola[1])), (int(bola[2]), int(bola[3])), (0, 255, 255), 2)
    cv2.putText(frame, f"POSSE: {equipa_nome}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return equipa_nome


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    equipa = identificar_posse(frame, results)
    if equipa:
        posse[equipa] += 1

    cv2.imshow("Football Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

total = posse["Equipa 1"] + posse["Equipa 2"]
print("\n=== POSSE DE BOLA FINAL ===")
if total == 0:
    print("Nenhuma posse detetada.")
else:
    print(f"Equipa 1: {100 * posse['Equipa 1'] / total:.1f}%")
    print(f"Equipa 2: {100 * posse['Equipa 2'] / total:.1f}%")