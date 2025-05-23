import os, cv2, argparse, subprocess, math
from datetime import datetime

# -----------------------------------------------------------
# Caminhos para binários OpenCV (ajuste conforme sua instalação)
# -----------------------------------------------------------
OPENCV_BIN_DIR    = r"C:\opencv\build\x64\vc15\bin"
CREATESAMPLES_EXE = os.path.join(OPENCV_BIN_DIR, "opencv_createsamples.exe")
TRAINCASCADE_EXE  = os.path.join(OPENCV_BIN_DIR, "opencv_traincascade.exe")

def _norm(p: str) -> str:
    """
    Converte barras invertidas em barras normais ('/') para
    garantir compatibilidade com parâmetros do OpenCV.
    """
    return p.replace("\\", "/")

def _run(cmd, cwd="."):
    """
    Imprime e executa o comando passado via subprocess.
    Se houver erro, interrompe a execução (check=True).
    """
    print("\n[*] " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True, cwd=cwd)

# -----------------------------------------------------------
# 1. Anotar imagens positivas
# -----------------------------------------------------------
def annotate_positives(positives_dir, annotations_file, max_size=None):
    """
    Abre cada imagem positiva, permite desenhar ROIs com o mouse
    e salva no arquivo de anotações no formato:
      caminho_relativo n_rois x1 y1 w1 h1 x2 y2 w2 h2 ...
    """
    imgs = sorted(os.listdir(positives_dir))
    os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
    total = 0

    with open(annotations_file, "w", encoding="utf-8") as fout:
        for img_name in imgs:
            abs_img = _norm(os.path.join(os.getcwd(), positives_dir, img_name))
            img = cv2.imread(abs_img)
            if img is None:
                print("[WARN] Imagem inválida:", img_name)
                continue

            h0, w0 = img.shape[:2]
            scale = 1
            # Se especificado max_size, reduz a imagem para facilitar anotação
            if max_size:
                max_w, max_h = max_size
                scale = min(
                    max_w / w0 if w0 > max_w else 1,
                    max_h / h0 if h0 > max_h else 1
                )
                if scale < 1:
                    img = cv2.resize(
                        img,
                        (int(w0 * scale), int(h0 * scale)),
                        interpolation=cv2.INTER_AREA
                    )

            rois = []
            # Loop de seleção de ROI até que o usuário cancele (w==0 ou h==0)
            while True:
                x, y, w, h = cv2.selectROI(f"Anotar: {img_name}", img, showCrosshair=True)
                if w == 0 or h == 0:
                    break
                # Converte coordenadas de volta, se houve redimensionamento
                if scale < 1:
                    x, y, w, h = map(lambda v: int(v / scale), (x, y, w, h))

                rois.append((x, y, w, h))
                # Desenha retângulo na pré-visualização
                cv2.rectangle(
                    img,
                    (int(x * scale), int(y * scale)),
                    (int((x + w) * scale), int((y + h) * scale)),
                    (0, 255, 0), 2
                )
                cv2.imshow("Preview", img)

            cv2.destroyAllWindows()

            if not rois:
                # Se nenhuma ROI foi anotada, pula a imagem
                continue

            # Caminho relativo do script até a imagem anotada
            rel_img = _norm(os.path.relpath(abs_img, start=os.path.dirname(annotations_file)))
            # Monta linha de anotação
            line = (
                f"{rel_img} {len(rois)} " +
                " ".join(f"{x} {y} {w} {h}" for x, y, w, h in rois)
            )
            fout.write(line + "\n")
            total += 1

    print(f"[+] {annotations_file} salvo ({total} linhas)")
    return total

# -----------------------------------------------------------
# 2. Listar imagens negativas
# -----------------------------------------------------------
def generate_bg(negatives_dir, bg_file):
    """
    Gera um arquivo de texto com a lista de todos os
    caminhos absolutos das imagens negativas, uma por linha.
    """
    imgs = sorted(os.listdir(negatives_dir))
    os.makedirs(os.path.dirname(bg_file), exist_ok=True)

    with open(bg_file, "w", encoding="utf-8") as f:
        for img_name in imgs:
            abs_path = _norm(os.path.join(os.getcwd(), negatives_dir, img_name))
            if os.path.isfile(abs_path):
                f.write(abs_path + "\n")

    print(f"[+] {bg_file} salvo ({len(imgs)} imagens)")
    return len(imgs)

# -----------------------------------------------------------
# 3. Criar .vec com aumento de dados
# -----------------------------------------------------------
def create_vec(info_file, vec_file, num, w, h, max_angle=18):
    """
    Executa opencv_createsamples para gerar um
    arquivo .vec com amostras positivas aumentadas.
    """
    info_abs, vec_abs = map(_norm, map(os.path.abspath, [info_file, vec_file]))
    os.makedirs(os.path.dirname(vec_abs), exist_ok=True)

    _run([
        CREATESAMPLES_EXE,
        "-info", info_abs,
        "-num", str(num),
        "-w", str(w), "-h", str(h),
        "-bgcolor", "0", "-bgthresh", "0",
        "-maxxangle", f"{max_angle/180:.3f}",   # rotações ±max_angle°
        "-maxyangle", "0.0",
        "-maxzangle", "0.0",
        "-vec", vec_abs
    ], cwd=os.path.dirname(info_abs))

    return vec_abs

# -----------------------------------------------------------
# 4. Treinar cascade
# -----------------------------------------------------------
def train_cascade(out_dir, vec, bg, numPos, numNeg, stages, w, h,
                  feature="LBP", min_hit=0.995, max_false=0.40):
    """
    Executa opencv_traincascade para treinar o classificador
    em múltiplos estágios, salvando o modelo em out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    _run([
        TRAINCASCADE_EXE,
        "-data", _norm(out_dir),
        "-vec", vec,
        "-bg",  bg,
        "-numPos", str(numPos),
        "-numNeg", str(numNeg),
        "-numStages", str(stages),
        "-w", str(w), "-h", str(h),
        "-featureType", feature,
        "-minHitRate", f"{min_hit:.3f}",
        "-maxFalseAlarmRate", f"{max_false:.3f}",
        "-precalcValBufSize", "4096",     # buffers em MB para acelerar
        "-precalcIdxBufSize", "4096",
        "-mode", "ALL"
    ])
    # Reporta o tamanho do modelo final
    size_kb = os.path.getsize(os.path.join(out_dir, "cascade.xml")) / 1024
    print(f"[+] cascade.xml pronto ({size_kb:.1f} KB)")

# -----------------------------------------------------------
# 5. Script principal
# -----------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Pipeline Haar/LBP Cascade – versão otimizada")
    # Argumentos obrigatórios e opcionais
    ap.add_argument("--positives", required=True, help="pasta com imgs positivas")
    ap.add_argument("--negatives", required=True, help="pasta com imgs negativas")
    ap.add_argument("--out",       default="training", help="pasta destino do modelo")

    ap.add_argument("--annotations", default="annotations/positives.txt")
    ap.add_argument("--bg",          default="annotations/bg.txt")
    ap.add_argument("--vec",         default="annotations/positives.vec")

    ap.add_argument("--w",      type=int, default=30, help="janela base W")
    ap.add_argument("--h",      type=int, default=30, help="janela base H")
    ap.add_argument("--stages", type=int, default=25)
    ap.add_argument("--feature", choices=["HAAR","LBP"], default="HAAR")
    ap.add_argument("--minHit",  type=float, default=0.995)
    ap.add_argument("--maxFalse",type=float, default=0.40)
    ap.add_argument("--maxWidth", type=int, default=1024)
    ap.add_argument("--maxHeight",type=int, default=768)
    ap.add_argument(
        "--aug", type=float, default=2.5,
        help="fator de aumento de amostras sintéticas (vec)"
    )
    args = ap.parse_args()

    # Início do treinamento com timestamp
    print(
        "\n=== Treinamento iniciado",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "===\n"
    )

    # 1) Coleta e anotação de positivos
    pos_count = annotate_positives(
        args.positives,
        args.annotations,
        max_size=(args.maxWidth, args.maxHeight)
    )
    if pos_count == 0:
        raise SystemExit("[ERRO] Nenhuma imagem positiva anotada.")

    # 2) Listagem de negativos
    neg_count = generate_bg(args.negatives, args.bg)
    if neg_count == 0:
        raise SystemExit("[ERRO] Nenhuma imagem negativa encontrada.")

    # 3) Geração de amostras .vec
    vec_samples = max(int(pos_count * args.aug), pos_count + 50)
    vec_abs = create_vec(args.annotations, args.vec, vec_samples, args.w, args.h)

    # 4) Ajuste automático de numPos e numNeg
    numPos = int(pos_count * 1)
    numNeg = int(min(neg_count, numPos * 2))
    print(f"[INFO] Treino: numPos={numPos}, numNeg={numNeg}")

    # 5) Treinamento do cascade
    train_cascade(
        args.out,
        vec_abs,
        _norm(os.path.abspath(args.bg)),
        numPos,
        numNeg,
        args.stages,
        args.w,
        args.h,
        feature=args.feature,
        min_hit=args.minHit,
        max_false=args.maxFalse
    )