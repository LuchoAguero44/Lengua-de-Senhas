import cv2
import numpy as np
import tensorflow as tf
import SeguimientoManos as sm


model = tf.keras.models.load_model("modelo_gestos.h5")


detector = sm.detectormanos(Confdeteccion=0.9)


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

class_names = ['1', '2', '3','A', 'B', 'C','L','Te quiero'] #nokmbres de las clases

while True:
    
    ret, frame = cap.read()
    
   
    frame = detector.encontrarmanos(frame)
    
   
    resultado = detector.encontrarposicion(frame, dibujarPuntos=False, dibujarBox=True)
    
    
    print(resultado)
    
    
    if len(resultado) == 2:  
        lista, bbox = resultado
    elif len(resultado) == 3:  
        lista, bbox, otro_valor = resultado
    else:
        lista, bbox = [], []  

  
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        xmin, ymin = max(0, xmin - 20), max(0, ymin - 20)  
        xmax, ymax = min(frame.shape[1], xmax + 20), min(frame.shape[0], ymax + 20)
        
        
        recorte = frame[ymin:ymax, xmin:xmax]
        recorte = cv2.resize(recorte, (448,448))  
        recorte = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB) 
        recorte = recorte / 255.0  
        
        
        recorte = np.expand_dims(recorte, axis=0)
        
        
        prediccion = model.predict(recorte)
        clase_index = np.argmax(prediccion)
        clase_gesto = class_names[clase_index]
        
        
        cv2.putText(frame, clase_gesto, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    
    cv2.imshow("Reconocimiento de Gestos", frame)

    
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
