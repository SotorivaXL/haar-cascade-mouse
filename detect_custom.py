# detect_custom.py – v3  (contagem + snapshots)
import cv2, argparse, time, os
from pathlib import Path
from datetime import datetime

def _parse_size(text):
    if not text: return None
    if "x" in text.lower():
        w, h = map(int, text.lower().split("x"))
        return (w, h)
    return None

def _iou(a, b):
    """Interseção-sobre-União para descartar a mesma box em quadros seguidos"""
    xA, yA, wA, hA = a; xB, yB, wB, hB = b
    xa, ya = max(xA, xB), max(yA, yB)
    xb, yb = min(xA+wA, xB+wB), min(yA+hA, yB+hB)
    inter = max(0, xb-xa) * max(0, yb-ya)
    union = wA*hA + wB*hB - inter
    return inter/union if union else 0

def detect(cascade_path, cam_idx, scale, neighbors,
           min_size=None, max_size=None, resize_w=None,
           save_dir="detections", iou_thresh=0.5):
    clf = cv2.CascadeClassifier(cascade_path)
    if clf.empty():
        raise SystemExit(f"[ERRO] Não foi possível carregar {cascade_path}")

    os.makedirs(save_dir, exist_ok=True)
    total_detected = 0
    prev_boxes = []

    cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"[ERRO] Webcam {cam_idx} não encontrada")
    print("[INFO] Pressione 'q' para sair")

    t0, frames, fps = time.time(), 0, 0.0  # ← inicializa fps
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # opcional: resize para ganhar FPS
        if resize_w and frame.shape[1] > resize_w:
            r = resize_w / frame.shape[1]
            frame = cv2.resize(frame, (resize_w, int(frame.shape[0]*r)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)

        rects = clf.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbors,
            minSize=min_size, maxSize=max_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # ── contagem + snapshot ───────────────────────────────────
        new_boxes = []
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            # considera novo se IoU < 0.5 com TODOS boxes do quadro anterior
            if all(_iou((x,y,w,h), pb) < iou_thresh for pb in prev_boxes):
                new_boxes.append((x,y,w,h))

        if new_boxes:
            total_detected += len(new_boxes)
            # salva frame inteiro com timestamp e total acumulado
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            out_path = Path(save_dir) / f"mouse_{total_detected:04d}_{ts}.png"
            cv2.imwrite(str(out_path), frame)
            print(f"[+] Snapshot salvo: {out_path}")

        prev_boxes = rects        # usa boxes deste quadro para o próximo

        # overlay: total detectado + FPS
        frames += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps = frames / elapsed  # atualiza fps
            t0, frames = time.time(), 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Mouse detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Detecção de mouse em tempo real + contagem/snapshot")
    ap.add_argument("--cascade", default="training_mouse/cascade.xml")
    ap.add_argument("--source",  default="0", help="índice da webcam")
    ap.add_argument("--scale",   type=float, default=1.25)
    ap.add_argument("--neighbors", type=int, default=8)
    ap.add_argument("--minSize", type=str, default="30x30")
    ap.add_argument("--maxSize", type=str, default=None)
    ap.add_argument("--resize",  type=int, default=640)
    ap.add_argument("--saveDir", type=str, default="detections",
                    help="pasta onde salvar snapshots")
    args = ap.parse_args()

    detect(args.cascade, args.source, args.scale, args.neighbors,
           min_size=_parse_size(args.minSize),
           max_size=_parse_size(args.maxSize),
           resize_w=args.resize if args.resize > 0 else None,
           save_dir=args.saveDir)