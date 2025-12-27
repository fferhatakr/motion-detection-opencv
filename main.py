"""
Bu proje webcam üzerinden hareket algılama yapar.
- Normal durumda 10 saniyede bir log kaydı alır
- Anormal harekette anında kayıt yapar
- Kayıtlar CSV dosyasına yazılır
"""

import cv2
from datetime import datetime
import numpy as np
import pandas as pd
import time
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logs = []
log_folder = os.path.join(BASE_DIR, "logs")
os.makedirs(log_folder, exist_ok=True)
csv_path = os.path.join(log_folder, "system_logs.csv")


last_normal_log_time = 0
normal_log_interval = 10  




cap = cv2.VideoCapture(0) #webcam kamerasını açma
ret , square = cap.read()
prev_gray = cv2.cvtColor(square,cv2.COLOR_BGR2GRAY) #webcemdeki goruntuyu griye cevirme

try:
    while True:
        ret , frame = cap.read() #goruntuyu okuma
        if not ret: #Kamera açık degilse?
            print("Camera could not be opened") 
            exit()

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #goruntuyu gri renge çevirme
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        frame_diff = cv2.absdiff(prev_gray,gray) 
        cv2.imshow("Frame Difference", frame_diff)  #Farklı Olan 2.Kamerayı Goster
        cv2.imshow("Normal Camera", frame) #Normal Kamerayı Göster

        threshold_value = 15
        ret, thresh = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)

        cv2.imshow("Threshold",thresh)
        prev_gray = gray #Esitleme

    
        current_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        motion_score = np.sum(thresh) / 255
        motion_detected = False
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) > 1000:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

           


        if motion_detected:
            logs.append({
                "time": timestamp,
                "status": "ANORMAL HAREKET",
                "score": motion_score
            })

        elif current_time - last_normal_log_time >= normal_log_interval:
            logs.append({
                "time": timestamp,
                "status": "Sistem aktif (normal)",
                "score": motion_score
            })
            last_normal_log_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'): #Cıkıs icin q 
            break
        

finally:
    if logs:
        df = pd.DataFrame(logs)
        df.to_csv(csv_path, index=False)
        print(f"{len(df)} The registration was successfully saved.") #Kaydı csv ye kaydeder
    
    cap.release() #Temizleme
    cv2.destroyAllWindows()#Temizleme



    









