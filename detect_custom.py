# detect_custom.py  –  v2 (2025‑05)
import cv2, argparse, time
from pathlib import Path

def _parse_size(text):
    """Converte 'WxH' → (w, h) ou None."""
    if not text: return None
    if "x" in text.lower():
        w, h = map(int, text.lower().split("x"))
        return (w, h)
    return None

def detect(cascade_path, source, scale, neighbors,
           min_size=None, max_size=None, resize_w=None):
    clf = cv2.CascadeClassifier(cascade_path)
    if clf.empty():
        raise SystemExit(f"[ERRO] Não foi possível carregar {cascade_path}")

    def _process_frame(frame):
        if resize_w and frame.shape[1] > resize_w:
            r = resize_w / frame.shape[1]
            frame = cv2.resize(frame, (resize_w, int(frame.shape[0]*r)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)           # ↓ influência de sombras / brilho

        rects = clf.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbors,
            minSize=min_size, maxSize=max_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame

    # ── Vídeo / webcam ──────────────────────────────────────────────
    try:
        cam_idx = int(source)
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise ValueError
        print("[INFO] Pressione 'q' para sair")
        t0, frames = time.time(), 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = _process_frame(frame)
            frames += 1
            if time.time() - t0 >= 1:
                fps = frames / (time.time() - t0)
                cv2.putText(frame, f"{fps:.1f} FPS", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                t0, frames = time.time(), 0
            cv2.imshow("Mouse detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    # ── Imagem ───────────────────────────────────────────────────────
    except ValueError:
        if not Path(source).is_file():
            raise SystemExit(f"[ERRO] Arquivo '{source}' não encontrado")
        img = cv2.imread(source)
        if img is None:
            raise SystemExit(f"[ERRO] Falha ao abrir {source}")
        img = _process_frame(img)
        cv2.imshow("Mouse detection", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Teste do cascade de mouse (LBP)")
    ap.add_argument("--cascade", default="training_mouse/cascade.xml")
    ap.add_argument("--source",  default="0",
                    help="índice webcam ou caminho de imagem/vídeo")
    ap.add_argument("--scale",   type=float, default=1.25,
                    help="scaleFactor (>1). ↑→ menos falsos, mais veloz")
    ap.add_argument("--neighbors", type=int, default=8,
                    help="minNeighbors. ↑→ menos falsos")
    ap.add_argument("--minSize", type=str, default="30x30",
                    help="menor janela a detectar. Formato WxH")
    ap.add_argument("--maxSize", type=str, default=None,
                    help="maior janela a detectar. Formato WxH")
    ap.add_argument("--resize",  type=int, default=640,
                    help="redimensiona largura do frame (0=off)")
    args = ap.parse_args()

    detect(args.cascade,
           args.source,
           args.scale,
           args.neighbors,
           min_size=_parse_size(args.minSize),
           max_size=_parse_size(args.maxSize),
           resize_w=args.resize if args.resize > 0 else None)