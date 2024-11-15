import cv2
import mediapipe as mp
import os

class Hand_Detector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.9, carpeta_destino="U", n_imagenes=60):
        # Inicializar Mediapipe y sus utilidades de dibujo
        self.mp_mano = mp.solutions.hands
        self.mano = self.mp_mano.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_dibujo = mp.solutions.drawing_utils
        self.captura = cv2.VideoCapture(0)

        # Verificar si la cámara está disponible
        if not self.captura.isOpened():
            exit()

        # Variables para las coordenadas y el recorte
        self.alto, self.ancho = 0, 0
        self.puntos = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 16, 18, 19, 20]
        self.coordenadas = []
        self.captura_imagenes = False
        self.contador_imagenes = 0
        self.n_imagenes = n_imagenes

        # Configuración de la carpeta destino
        self.carpeta_destino = carpeta_destino
        if not os.path.exists(self.carpeta_destino):
            os.makedirs(self.carpeta_destino)

    def ProcesarFrame(self, frame):
        # Actualizar las dimensiones de la imagen
        self.alto, self.ancho, _ = frame.shape
        x_medio = self.ancho // 2  # Línea central en X
        
        # Convertir el frame a RGB para Mediapipe
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = self.mano.process(color)
        
        # Limpiar coordenadas anteriores
        self.coordenadas.clear()

        if resultado.multi_hand_landmarks:
            for mano_landmarks in resultado.multi_hand_landmarks:
                # Dibujar landmarks
                self.mp_dibujo.draw_landmarks(
                    frame,
                    mano_landmarks,
                    self.mp_mano.HAND_CONNECTIONS,
                    self.mp_dibujo.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    self.mp_dibujo.DrawingSpec(color=(0, 255, 255), thickness=2)
                )

                # Obtener la coordenada central de la mano (punto 9)
                pto_i5 = mano_landmarks.landmark[9]
                cx, cy = int(pto_i5.x * self.ancho), int(pto_i5.y * self.alto)

                # Determinar si la mano está en el lado izquierdo o derecho de la pantalla
                if cx < x_medio:
                    # Mano en la izquierda - limitar cuadro a la mitad izquierda
                    x1, y1 = max(0, cx - 100), max(0, cy - 100)
                    x2, y2 = min(x_medio, x1 + 200), min(self.alto, y1 + 200)
                else:
                    # Mano en la derecha - limitar cuadro a la mitad derecha
                    x1, y1 = max(x_medio, cx - 100), max(0, cy - 100)
                    x2, y2 = min(self.ancho, x1 + 200), min(self.alto, y1 + 200)

                # Dibujar el cuadro de seguimiento
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Si se está en el modo de captura de imágenes, guardar la foto
                if self.captura_imagenes and self.contador_imagenes < self.n_imagenes:
                    recorte = frame[y1:y2, x1:x2]
                    nombre_foto = os.path.join(self.carpeta_destino, f"U_{self.contador_imagenes + 181}.jpg")
                    cv2.imwrite(nombre_foto, recorte)
                    print(f"Foto {self.contador_imagenes + 181} guardada como {nombre_foto}")
                    self.contador_imagenes += 1

        return frame
    
    def Iniciar(self):
        print("Presiona 'Espacio' para comenzar a capturar fotos o 'q' para salir.")
        
        while True:
            ret, frame = self.captura.read()
            if not ret:
                break

            # Modo espejo
            frame = cv2.flip(frame, 1)

            # Procesar el frame y obtener el frame con las manos detectadas
            frame = self.ProcesarFrame(frame)

            # Mostrar frame
            cv2.imshow('Camara', frame)

            # Comenzar la captura cuando se presiona la tecla 'Espacio'
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Espacio
                if self.contador_imagenes == 0:
                    print("Iniciando captura de fotos...")
                self.captura_imagenes = True

            # Capturar las coordenadas de los landmarks al presionar Enter
            if key == 13 and self.coordenadas:
                print("Coordenadas de landmarks:", self.coordenadas)

            # Salir del bucle si se presiona "Esc" (código ASCII = 27)
            if key == 27 or self.contador_imagenes >= self.n_imagenes:
                print("Captura finalizada o se presionó 'Esc' para salir.")
                break

        # Finalizar captura y cerrar ventanas
        self.captura.release()
        cv2.destroyAllWindows()

# Crear una instancia y ejecutar la detección
if __name__ == "__main__":
    detector_manos = Hand_Detector(n_imagenes=60)  # Aquí puedes cambiar 'n_imagenes' al número de fotos que deseas capturar
    detector_manos.Iniciar()