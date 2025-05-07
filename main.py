import cv2

# Carrega o modelo Haar Cascade treinado para detectar mouses
model_mouse = cv2.CascadeClassifier('cascade/cascade.xml//////')  # Caminho para seu modelo treinado

# Carrega a imagem de teste
img = cv2.imread('mause/.jpg')  # Substitua por sua imagem de teste
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detecta mouses na imagem
mouses = model_mouse.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30))  # Ajuste se necessário)

# Desenha retângulos ao redor dos mouses detectados
for (x, y, w, h) in mouses:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Exibe a imagem com as detecções
cv2.imshow('Mouse Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()