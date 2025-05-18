# auto_pipeline.py
import os
import cv2
import argparse
import subprocess

# -----------------------------------------------------------
# Caminhos para os binários do OpenCV
# -----------------------------------------------------------
OPENCV_BIN_DIR = r"C:\opencv\build\x64\vc15\bin"
CREATESAMPLES_EXE = os.path.join(OPENCV_BIN_DIR, "opencv_createsamples.exe")
TRAINCASCADE_EXE = os.path.join(OPENCV_BIN_DIR, "opencv_traincascade.exe")

# -----------------------------------------------------------
# Utilidades
# -----------------------------------------------------------

def _norm(p: str) -> str:
    return p.replace("\\", "/")

# -----------------------------------------------------------
# POSITIVES.TXT ----------------------------------------------------------------------
# -----------------------------------------------------------

def annotate_positives(positives_dir: str, annotations_file: str, max_size=None) -> int:
    project_root = _norm(os.getcwd())
    imgs = sorted(os.listdir(positives_dir))
    info_dir = _norm(os.path.dirname(os.path.abspath(annotations_file)))
    os.makedirs(info_dir, exist_ok=True)

    total = 0
    with open(annotations_file, "w", encoding="utf-8") as f:
        for img_name in imgs:
            abs_img = _norm(os.path.join(project_root, positives_dir, img_name))
            img = cv2.imread(abs_img)
            if img is None:
                print(f"[WARN] Imagem inválida: {abs_img}")
                continue
            h0, w0 = img.shape[:2]
            scale = 1.0
            if max_size:
                max_w, max_h = max_size
                scale = min((max_w / w0) if w0 > max_w else 1.0,
                            (max_h / h0) if h0 > max_h else 1.0)
                if scale < 1.0:
                    img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
            rois = []
            while True:
                x, y, w, h = cv2.selectROI(f"Anotar: {img_name}", img, showCrosshair=True)
                if w == 0 or h == 0:
                    break
                if scale != 1.0:
                    x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
                if x < 0 or y < 0 or x + w > w0 or y + h > h0:
                    print(f"[WARN] ROI inválida {img_name}: {(x,y,w,h)} > {(w0,h0)}")
                    continue
                rois.append((x, y, w, h))
                cv2.rectangle(img, (int(x*scale), int(y*scale)), (int((x+w)*scale), int((y+h)*scale)), (0,255,0),2)
                cv2.imshow("Preview", img)
            cv2.destroyAllWindows()
            if not rois:
                continue
            rel_img = _norm(os.path.relpath(abs_img, start=info_dir))
            line = rel_img + " " + str(len(rois))
            for x, y, w, h in rois:
                line += f" {x} {y} {w} {h}"
            f.write(line + "\n")
            total += 1
    print(f"[+] positives.txt salvo ({total} linhas)")
    return total

# -----------------------------------------------------------
# BG.TXT -----------------------------------------------------------------------------
# -----------------------------------------------------------

def generate_negatives_list(negatives_dir: str, bg_file: str) -> int:
    project_root = _norm(os.getcwd())
    imgs = sorted(os.listdir(negatives_dir))
    os.makedirs(os.path.dirname(bg_file), exist_ok=True)
    count = 0
    with open(bg_file, "w", encoding="utf-8") as f:
        for img_name in imgs:
            abs_img = _norm(os.path.join(project_root, negatives_dir, img_name))
            if os.path.isfile(abs_img):
                f.write(abs_img + "\n")
                count += 1
    print(f"[+] bg.txt salvo ({count} imagens)")
    return count

# -----------------------------------------------------------
# .VEC --------------------------------------------------------------------------------
# -----------------------------------------------------------

def create_vec(info_file: str, vec_file: str, num: int, w: int, h: int):
    info_abs = _norm(os.path.abspath(info_file))
    vec_abs = _norm(os.path.abspath(vec_file))
    os.makedirs(os.path.dirname(vec_abs), exist_ok=True)
    cmd = [CREATESAMPLES_EXE, "-info", info_abs, "-num", str(num), "-w", str(w), "-h", str(h), "-vec", vec_abs]
    print("[*] Executando:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=os.path.dirname(info_abs))
    print(f"[+] .vec gerado em {vec_abs}")
    return vec_abs

# -----------------------------------------------------------
# TREINAMENTO ------------------------------------------------------------------------
# -----------------------------------------------------------

def train_cascade(out_dir: str, vec_abs: str, bg_abs: str, numPos: int, numNeg: int, stages: int, w: int, h: int):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        TRAINCASCADE_EXE, "-data", out_dir, "-vec", vec_abs, "-bg", bg_abs,
        "-numPos", str(numPos), "-numNeg", str(numNeg), "-numStages", str(stages),
        "-w", str(w), "-h", str(h), "-featureType", "HAAR",
        "-minHitRate", "0.995", "-maxFalseAlarmRate", "0.5"
    ]
    print("[*] Executando:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[+] Cascade treinado -> {os.path.join(out_dir,'cascade.xml')}")

# -----------------------------------------------------------
# MAIN --------------------------------------------------------------------------------
# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pipeline Haar Cascade – corrige caminho .vec")
    parser.add_argument("--positives", required=True)
    parser.add_argument("--negatives", required=True)
    parser.add_argument("--annotations", default="annotations/positives.txt")
    parser.add_argument("--bg", default="annotations/bg.txt")
    parser.add_argument("--vec", default="annotations/positives.vec")
    parser.add_argument("--out", default="training")
    parser.add_argument("--w", type=int, default=24)
    parser.add_argument("--h", type=int, default=24)
    parser.add_argument("--stages", type=int, default=10)
    parser.add_argument("--maxWidth", type=int, default=1024)
    parser.add_argument("--maxHeight", type=int, default=768)
    args = parser.parse_args()

    max_size = (args.maxWidth, args.maxHeight)

    pos_count = annotate_positives(args.positives, args.annotations, max_size)
    if pos_count == 0:
        raise SystemExit("[ERRO] Nenhuma imagem positiva anotada.")

    neg_count = generate_negatives_list(args.negatives, args.bg)

    vec_abs = create_vec(args.annotations, args.vec, pos_count, args.w, args.h)
    bg_abs = _norm(os.path.abspath(args.bg))

    numPos_train = max(10, int(pos_count * 0.9))
    numNeg_train = max(10, min(neg_count, int(numPos_train * 1.5)))
    print(f"[INFO] numPos={numPos_train}, numNeg={numNeg_train}")

    train_cascade(args.out, vec_abs, bg_abs, numPos_train, numNeg_train, args.stages, args.w, args.h)