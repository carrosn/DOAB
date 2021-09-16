# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 14:33:07 2021

@author: luiscrn
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 10:33:15 2021

@author: luiscrn
"""
import datetime
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from collections import Counter, defaultdict
from skimage.metrics import structural_similarity as ssim
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from lib import utils
import cv2

track_temp=[]
track_master=[]
track_temp2=[]
track_masterTemp=[]
top_contour_dict = defaultdict(int)
top_contour_dictTemp = defaultdict(int)
obj_detected_dict = defaultdict(int)
clases=utils.load_classes("coco.names")
frameno=0
consecutiveframe=20
conse=250
back_Temp=[]
print(torch.__version__)
print(cv2.__version__)
print(clases)
model = torch.hub.load('ultralytics/yolov5', 'yolov5x') #carga del modelo preeentrenado desde ultralytics
#captura de la secuencia de video
cap = cv2.VideoCapture('data//video1.avi')
#cap = cv2.VideoCapture('data//AVSS_AB1.mp4')
#captura tiempo inicial de procesamiento
t_start = datetime.datetime.now()
print(t_start)
while(1):
    
    ret, frame = cap.read()
    #frame=cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    
   
    if ret==0:
        break
    #Captura primer fotograma fondo
    if frameno==0:
       firstframe = frame
       firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
       back_Temp.append(firstframe_gray)
     
       
   #Captura progresiva fotograma fondo    
    if frameno==consecutiveframe:
       firstframe = frame
       firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
       back_Temp.append(firstframe_gray)
       conse=conse+250

    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray,(21,21),0)
    kernel = np.ones((5,5),np.uint8) 
    frame_diff = cv2.absdiff(firstframe_gray, frame_gray)
    close_operated_image = cv2.morphologyEx(frame_diff, cv2.MORPH_CLOSE, kernel,iterations=2)
    _, thresholded = cv2.threshold(close_operated_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    median = cv2.GaussianBlur(thresholded, (21,21),0)
  
    #Detección Canny Edge 
    edged = cv2.Canny(median,10,200) #Cualquier gradiente entre 30 y 150 se considera aristas.
    cv2.imshow('CannyEdgeDet',edged)
    kernel2 = np.ones((5,5),np.uint8) 
    thresh2 = cv2.morphologyEx(edged,cv2.MORPH_CLOSE, kernel2,iterations=2)
    cv2.imshow('Morph_Close', thresh2)
    #Crea una copia del umbral para encontrar contornos   
    cnts,hierarchy = cv2.findContours(thresh2.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    frameno = frameno + 1
    mycnts =[] # every new frame, set to empty list. 
    # loop over the contours
    for c in cnts:
         # calculo del centride
        M = cv2.moments(c)
        if M['m00'] == 0: 
            pass
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cv2.contourArea(c) < 200 or cv2.contourArea(c)>20000:
                pass
            else:
                mycnts.append(c)
                  
                # captura del marco de cada controno
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                
                
                
                rois_o =[]  
                rois_d=[]
                rois_o.append(firstframe_gray[y:(y + h),x:(x + w)])  
                rois_d.append(frame_gray[y:(y + h),x:(x + w)])
                roiyo,roixo=rois_o[0].shape
                roiyd,roixd=rois_d[0].shape
                #roi comparación se uliza para validar cada region de interes del fotograma actual contra el fondo
                if (roiyo>=7 and roixo>=7) and (roiyd>=7 and roixd>=7) :
                 s = ssim(rois_o[0], rois_d[0])#calcula el ssim la similitud
                else :
                 s=0
            
                #umbral para validar la similitud
                if s<0.3 :
                    sumcxcy=cx+cy 
                    track_temp.append([cx+cy,frameno])
                    track_master.append([cx+cy,frameno])
                    countuniqueframe = set(j for i, j in track_master)
                    
                    
                    if len(countuniqueframe)>consecutiveframe or False: 
                        minframeno=min(j for i, j in track_master)
                        for i, j in track_master:
                            if j != minframeno: # se crea una nueva lista que descarta los últimos frames 
                                track_temp2.append([i,j])
                            else:
                                if j>10 :
                                 track_masterTemp.append([i,j])
                    
                        track_master=list(track_temp2) # actualiza la lista master centrides
                        track_temp2=[]
                        
                       
                        countcxcyTemp = Counter(i for i, j in track_masterTemp)
                       
                        for i,j in countcxcyTemp.items(): 
                            if j>=consecutiveframe:
                                top_contour_dictTemp[i] += 1
                       
                           
                        
                        countcxcy = Counter(i for i, j in track_master)
                        
                        for i,j in countcxcy.items(): 
                            if j>=consecutiveframe:
                                top_contour_dict[i] += 1
                                
                                
                        salida = {}
                        if sumcxcy in top_contour_dictTemp:
                            keys = top_contour_dictTemp.keys() | top_contour_dict.keys()
                           
                            for key in keys:
                                if top_contour_dictTemp.get(key) != top_contour_dict.get(key):
                                  if top_contour_dictTemp.get(key) is not None and top_contour_dict.get(key) is not None:
                                      salida[key] = [top_contour_dictTemp.get(key)+ top_contour_dict.get(key)]
                     
                        if sumcxcy in salida:
                           
                           
                            
                            if top_contour_dict[sumcxcy]>100:
                                results = model(frame[y-50:(y + h+20),x:(x-50 + w+50)])# proceso de detección de YOLOv5
                                                             
                                if len(results.xywh[0]) == 1 :
                                    #obtiene las coordenadas del objeto detectado por YOLOv5s
                                    x1=results.xyxy[0][0][0].cpu().numpy()
                                    y1=results.xyxy[0][0][1].cpu().numpy()
                                    x2=results.xyxy[0][0][2].cpu().numpy()
                                    y2=results.xyxy[0][0][3].cpu().numpy()
                                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                                    # cv2.rectangle(thresh2, (x, y), (x + w, y + h), (255, 0, 0), 3)
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                                    cv2.rectangle(thresh2, (x, y), (x + w, y + h), (255, 0, 0), 3)
                                    #busca e imprime la etiqueta de la clase
                                    cv2.putText(frame,'%s'%(clases[int(results.xyxyn[0][:, -1].cpu().numpy()[0])]), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                                    print ('Detectado : ', sumcxcy,frameno, top_contour_dict)
                                    elapsedTime= (datetime.datetime.now()-t_start).total_seconds()

                                else :# en dcaso de que el objeto no sea detectado
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                                    cv2.rectangle(thresh2, (x, y), (x + w, y + h), (255, 0, 0), 3)
                                    cv2.putText(frame,'%s'%('Objeto no identificado'), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                                    print ('Detectado : ', sumcxcy,frameno, top_contour_dict)
                                    elapsedTime= (datetime.datetime.now()-t_start).total_seconds()
                                    print(elapsedTime)
                                    print(frameno)
                                  
                                                              
                                obj_detected_dict[sumcxcy]=frameno
                            
    for i, j in list(obj_detected_dict.items()):
        if frameno - obj_detected_dict[i]>150:
            obj_detected_dict.pop(i)
            
            # Establece recuento de, por ejemplo, 448 en cero. porque no se ha "activado" durante 200 fotogramas. Probablemente, haber sido eliminado
            top_contour_dict[i]=0
           
    cv2.imshow('Morph_Close', thresh2)        
    cv2.imshow('main',frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
     
 
cap.release()
cv2.destroyAllWindows()
