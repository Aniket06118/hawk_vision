import numpy as np
import cv2
import time
from hawk_vision import query_model
from PIL import Image
cap=cv2.VideoCapture(0)

last_vlm_call=0
interval_seconds=3

frames=[]
gap=5
count=0

while True:
    ret,frame=cap.read()

    if not ret:
        break
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frames.append(gray)

    if len(frames)>gap+1:
        frames.pop(0)
    
    cv2.putText(frame,f'frame count {count}',(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    if len(frames)>gap:
        diff=cv2.absdiff(frames[0],frames[-1])
        _, thresh=cv2.threshold(diff,30,255,cv2.THRESH_BINARY)
        contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c)<2000:
                continue

            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        motion= any(cv2.contourArea(c)>2000 for c in contours)

        if motion:
            current_time=time.time()
            if motion and current_time-last_vlm_call>interval_seconds:
                #middle_frame=frames[len(frames)//2]
                print("VLM IN ACTION , LETS GOO !!")
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                response=query_model(pil_image,"what do you see in the image ? ")
                print(response,"\n")
                last_vlm_call=current_time

            cv2.putText(frame,"Motion detected",(10,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            #print("MOTIONNNN!!!!!!!!!!")
          #  cv2.imwrite(f"motion_frame_{count}.jpg",frame)
           # print(f"saved : motion_frame_{count}.jpg")

        cv2.imshow("motion detected",frame)
        count+=1

        if cv2.waitKey(1) & 0xFF ==27:
            break
    
cap.release()
cv2.destroyAllWindows()



        



