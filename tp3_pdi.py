import cv2
import numpy as np
import matplotlib.pyplot as plt

numero_video = 4

ruta_video_orig = f'tirada_{numero_video}.mp4'

ruta_video_modif = f'Video-Output_{numero_video}.mp4'

##############- Funciones de filtrado de color -#############

def filtrar_color_verde(img):
    # Convertir la imagen a formato HSV
    image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Definir los valores de umbral para el color verde
    lower_green = np.array([30, 20, 20])
    upper_green = np.array([90, 255, 255])
    # Crear la máscara
    mask = cv2.inRange(image_HSV, lower_green, upper_green)
    # Aplicar la máscara a la imagen original
    image_filtered = cv2.bitwise_and(img, img, mask=mask)
    return image_filtered, mask

def filtrar_color_rojo(img,inf,sup):
    # Convertir la imagen a formato HSV
    image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Definir los valores de umbral para el color verde
    lower_green = np.array([inf, 1, 1])
    upper_green = np.array([sup, 255, 255])
    # Crear la máscara
    mask = cv2.inRange(image_HSV, lower_green, upper_green)
    # Aplicar la máscara a la imagen original
    image_filtered = cv2.bitwise_and(img, img, mask=mask)
    return image_filtered, mask



############- Procesamiento de video y obtención de frames -############


cap = cv2.VideoCapture(ruta_video_orig)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
numero_frame = 0
lista_mask = []
lista_nro_frames = []
lista_stats = []
lista_frame = []
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        numero_frame += 1
        frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
        # Usamos la función de filtrado de color verde para obtener la máscara 
        filtered_image_verde, mask_verde = filtrar_color_verde(frame)
        contours_verde, _ = cv2.findContours(mask_verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Encontramos el contorno más grande, creamos máscara en blanco y la aplicamos
        largest_contour = max(contours_verde, key=cv2.contourArea)
        mask_roi = np.zeros_like(frame)
        cv2.drawContours(mask_roi, [largest_contour], 0, (255, 255, 255), thickness=cv2.FILLED)
        image_roi = cv2.bitwise_and(frame, mask_roi)
        # Usamos la función de filtrado de color rojo para obtener la máscara 
        filtered_image_rojo, mask_rojo = filtrar_color_rojo(image_roi,1,10)
        filtered_image_rojo1, mask_rojo1 = filtrar_color_rojo(image_roi,170,180)
        mask_rojo_final = cv2.bitwise_or(mask_rojo, mask_rojo1)
        # Dilatamos las máscaras           
        kernel = np.ones((3, 3), np.uint8)
        mask_dilatada = cv2.dilate(mask_rojo_final, kernel, iterations=1)
        # Nos quedamos con los frames con cinco dados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dilatada, 8, cv2.CV_32S)
        if num_labels == 6:
            lista_mask.append(mask_dilatada)
            lista_nro_frames.append(numero_frame)
            lista_stats.append(stats)
            lista_frame.append(frame)
    else:
        break

cap.release()


############- Tratamiento de los frames -###############

def calcular_diferencias_bbox(stats1, stats2):
    # Extraer las coordenadas de los bounding boxes 
    bbox1 = stats1[:, :4]
    bbox2 = stats2[:, :4]
    # Calcular las diferencias entre los bounding boxes
    diferencias = np.sqrt(np.sum((bbox2 - bbox1) ** 2, axis=1))
    return diferencias

# Buscamos los índices de los frames donde los dados no se mueven 

lista_indices = []

for i in range(1,len(lista_nro_frames)-1):
    if  np.all(np.round(calcular_diferencias_bbox(lista_stats[i-1],lista_stats[i]),decimals=2) <2) and np.all(np.round(calcular_diferencias_bbox(lista_stats[i],lista_stats[i+1]),decimals=2)<2):
        lista_indices.append(i-1)
        lista_indices.append(i)
        lista_indices.append(i+1)


conjunto_sin_duplicados = set(lista_indices)

lista_sin_duplicados = list(conjunto_sin_duplicados)
      

lista_sin_duplicados = lista_sin_duplicados[3:]

# Obtenemos las listas de frames, stats, máscaras y nro de frames según índices obtenidos

lista_frame_final = []
lista_stats_final = []
lista_mask_final = []
lista_nro_frames_final = []
for i in range(len(lista_nro_frames)):
    for j in lista_sin_duplicados:
        if i == j:
            lista_nro_frames_final.append(lista_nro_frames[j])
            lista_frame_final.append(lista_frame[j])
            lista_stats_final.append(lista_stats[j])
            lista_mask_final.append(lista_mask[j])


# Función de filtrado por área
def filtrar_por_area(componentes_filtrados):
    _, etiquetas, stats, _ = cv2.connectedComponentsWithStats(componentes_filtrados)
    area_minima = 10
    area_maxima = 50
    componentes_filtrados_aspecto = np.zeros_like(componentes_filtrados)
    for i, stat in enumerate(stats):
        x, y, w, h, area = stat
        if area_minima <= area <= area_maxima:
            componentes_filtrados_aspecto[etiquetas == i] = 255
    return componentes_filtrados_aspecto

###############- Dibujo de bounding boxes y números de los dados en los frames -#################

lista_frame_final_modif = []
num_labels1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(lista_mask_final[6], 8, cv2.CV_32S)

for i in range(len(lista_frame_final)):
    nuevo_frame = lista_frame_final[i].copy()
    for j in range(1, num_labels1):       
        # Creamos una máscara para aislar cada dado en la imagen
        mask = np.zeros_like(nuevo_frame)
        mask = cv2.rectangle(mask, (stats1[j, cv2.CC_STAT_LEFT], stats1[j, cv2.CC_STAT_TOP]-5), 
                            (stats1[j, cv2.CC_STAT_LEFT] + stats1[j, cv2.CC_STAT_WIDTH], stats1[j, cv2.CC_STAT_TOP]- 5 + stats1[j, cv2.CC_STAT_HEIGHT] + 5),
                            (255, 255, 255), thickness=cv2.FILLED)
        resultado = cv2.bitwise_and(lista_frame_final[0], mask)
        # Recortamos la región del dado y la convertirmos a escala de grises
        mask_grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_uint8 = mask_grayscale.astype(np.uint8)
        x, y, w, h = cv2.boundingRect(mask_uint8)
        resultado_recortado = resultado[y:y+h, x:x+w]
        imagen_gris = cv2.cvtColor(resultado_recortado, cv2.COLOR_BGR2GRAY)
        # Umbralamos cada dado para obtener los puntos
        _, f_point = cv2.threshold(imagen_gris, 185, 255, cv2.THRESH_BINARY)
        f_point = f_point.astype(np.uint8)
        # Dilatamos los puntos
        elemento_estructural_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dado_dilatado = cv2.dilate(f_point, elemento_estructural_3, iterations=1)   
        dado_dilatado_filtrado = filtrar_por_area(dado_dilatado)
        # Obtenemos la información de los puntos
        num_labels_dados, labels_dados, stats_dados, centroids_dados = cv2.connectedComponentsWithStats(dado_dilatado_filtrado, 8, cv2.CV_32S)
        # Dibujamos los bounding boxes sobre los dados
        nuevo_frame = cv2.rectangle(nuevo_frame, (stats1[j, cv2.CC_STAT_LEFT], stats1[j, cv2.CC_STAT_TOP]), 
                                    (stats1[j, cv2.CC_STAT_LEFT] + stats1[j, cv2.CC_STAT_WIDTH] , stats1[j, cv2.CC_STAT_TOP] + stats1[j, cv2.CC_STAT_HEIGHT]), (255, 0, 0), 2)
        # Dibujamos los números de los dados
        nuevo_frame = cv2.putText(nuevo_frame,str(num_labels_dados-1),(stats1[j, cv2.CC_STAT_LEFT] + 10, stats1[j, cv2.CC_STAT_TOP] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)         
    lista_frame_final_modif.append(nuevo_frame) 



############- Leer y grabar el video -##########################

cap = cv2.VideoCapture(ruta_video_orig)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
numero_frame = 0
numero_lista = 0
out = cv2.VideoWriter(ruta_video_modif, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        numero_frame += 1
        if numero_frame in lista_nro_frames_final:
            # volver el frame al tamaño original
            frame_original_size = cv2.resize(lista_frame_final_modif[numero_lista], dsize=(width, height))
            out.write(frame_original_size)
            numero_lista += 1
        else:
            out.write(frame)  
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows() 


############- Mostrar el video final -##############################


cap = cv2.VideoCapture(ruta_video_modif)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()





