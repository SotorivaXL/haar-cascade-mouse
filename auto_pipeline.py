# auto_pipeline.py
import os
import cv2
import argparse
import subprocess

# Caminhos absolutos dos executáveis do OpenCV
OPENCV_BIN_DIR = r"C:\opencv\build\x64\vc15\bin"
CREATESAMPLES_EXE = os.path.join(OPENCV_BIN_DIR, "opencv_createsamples.exe")
TRAINCASCADE_EXE = os.path.join(OPENCV_BIN_DIR, "opencv_traincascade.exe")


def annotate_positives(positives_dir, annotations_file, max_size=None):
    """
    Abre cada imagem positiva e permite desenhar múltiplas ROIs.
    Redimensiona para max_size, se aplicável. Gera anotação com caminho absoluto:
      C:/caminho/para/img.jpg N x1 y1 w1 h1 [...]
    """
    positives_dir = os.path.abspath(positives_dir)
    with open(annotations_file, 'w') as f:
        for img_name in sorted(os.listdir(positives_dir)):
            rel_path = os.path.join(positives_dir, img_name)
            abs_path = os.path.abspath(rel_path)
            img = cv2.imread(abs_path)
            if img is None:
                print(f"[WARN] Não foi possível ler {abs_path}")
                continue
            original_h, original_w = img.shape[:2]
            scale = 1.0

            # Ajusta tamanho máximo
            if max_size:
                max_w, max_h = max_size
                scale_w = max_w / original_w if original_w > max_w else 1.0
                scale_h = max_h / original_h if original_h > max_h else 1.0
                scale = min(scale_w, scale_h)
                if scale < 1.0:
                    img = cv2.resize(img, (int(original_w*scale), int(original_h*scale)), interpolation=cv2.INTER_AREA)

            rois = []
            while True:
                roi = cv2.selectROI(f"Anotar: {img_name}", img, showCrosshair=True)
                if roi[2] == 0 or roi[3] == 0:
                    break
                x, y, w, h = roi
                # Converte para coordenadas originais se redimensionou
                if scale != 1.0:
                    x = int(x / scale)
                    y = int(y / scale)
                    w = int(w / scale)
                    h = int(h / scale)
                rois.append((x, y, w, h))
                # Desenha preview
                cv2.rectangle(img, (int(x*scale), int(y*scale)), (int((x+w)*scale), int((y+h)*scale)), (0,255,0), 2)
                cv2.imshow("Preview", img)
            cv2.destroyAllWindows()

            if rois:
                line = abs_path + ' ' + str(len(rois))
                for x, y, w, h in rois:
                    line += f" {x} {y} {w} {h}"
                f.write(line + "\n")
    print(f"[+] Anotações salvas em {annotations_file}")


def generate_negatives_list(negatives_dir, bg_file):
    """
    Gera bg.txt com caminhos absolutos das imagens negativas.
    """
    negatives_dir = os.path.abspath(negatives_dir)
    with open(bg_file, 'w') as f:
        for img_name in sorted(os.listdir(negatives_dir)):
            abs_path = os.path.join(negatives_dir, img_name)
            if os.path.isfile(abs_path):
                f.write(abs_path + "\n")
    print(f"[+] Lista de negativos salva em {bg_file}")


def create_vec(annotations_file, vec_file, num, w, h, opencv_bin=CREATESAMPLES_EXE):
    """
    Chama opencv_createsamples com caminhos absolutos.
    """
    cmd = [opencv_bin, '-info', os.path.abspath(annotations_file), '-num', str(num), '-w', str(w), '-h', str(h), '-vec', os.path.abspath(vec_file)]
    print("[*] Executando:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[+] .vec gerado em {vec_file}")


def train_cascade(output_dir, vec_file, bg_file, numPos, numNeg, numStages, w, h, opencv_bin=TRAINCASCADE_EXE):
    """
    Chama opencv_traincascade com caminhos absolutos.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        opencv_bin,
        '-data', os.path.abspath(output_dir),
        '-vec', os.path.abspath(vec_file),
        '-bg', os.path.abspath(bg_file),
        '-numPos', str(numPos),
        '-numNeg', str(numNeg),
        '-numStages', str(numStages),
        '-w', str(w),
        '-h', str(h),
        '-featureType', 'HAAR',
        '-minHitRate', '0.995',
        '-maxFalseAlarmRate', '0.5'
    ]
    print("[*] Executando:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[+] Cascade treinado! XML em {os.path.join(output_dir, 'cascade.xml')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline Haar Cascade with absolute paths")
    parser.add_argument('--positives',  required=True, help='pasta com imagens positivas')
    parser.add_argument('--negatives',  required=True, help='pasta com imagens negativas')
    parser.add_argument('--annotations', default='annotations/positives.txt', help='arquivo de anotações')
    parser.add_argument('--bg',          default='annotations/bg.txt', help='arquivo de negativos')
    parser.add_argument('--vec',         default='annotations/positives.vec', help='arquivo .vec')
    parser.add_argument('--out',         default='training', help='pasta de saída')
    parser.add_argument('--w',    type=int, default=24, help='width')
    parser.add_argument('--h',    type=int, default=24, help='height')
    parser.add_argument('--numPos',   type=int, default=30, help='positives')
    parser.add_argument('--numNeg',   type=int, default=50, help='negatives')
    parser.add_argument('--numStages',type=int, default=10, help='stages')
    parser.add_argument('--maxWidth', type=int, help='max width')
    parser.add_argument('--maxHeight',type=int, help='max height')
    args = parser.parse_args()

    max_size = (args.maxWidth, args.maxHeight) if args.maxWidth and args.maxHeight else None

    annotate_positives(args.positives, args.annotations, max_size)
    generate_negatives_list(args.negatives, args.bg)
    create_vec(args.annotations, args.vec, args.numPos, args.w, args.h)
    train_cascade(args.out, args.vec, args.bg, args.numPos, args.numNeg, args.numStages, args.w, args.h)