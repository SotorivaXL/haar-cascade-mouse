# detect_custom.py
import cv2
import argparse

def detect(cascade_path, source, scale, neighbors):
    clf = cv2.CascadeClassifier(cascade_path)
    # webcam se for dígito, senão imagem
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
        while True:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = clf.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors)
            for x, y, w, h in rects:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.imshow('Webcam Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    except ValueError:
        img = cv2.imread(source)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = clf.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors)
        for x, y, w, h in rects:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imshow('Image Detection', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testar Haar Cascade customizado')
    parser.add_argument('--cascade',   default='training/cascade.xml',
                        help='caminho para cascade.xml gerado')
    parser.add_argument('--source',    default='0',
                        help='arquivo de imagem ou índice da webcam (0,1,...)')
    parser.add_argument('--scale', type=float, default=1.1,
                        help='scaleFactor para detectMultiScale')
    parser.add_argument('--neighbors', type=int, default=5,
                        help='minNeighbors para detectMultiScale')
    args = parser.parse_args()

    detect(args.cascade, args.source, args.scale, args.neighbors)
