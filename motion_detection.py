import numpy as np
import cv2
import time
from hawk_vision import query_model
from PIL import Image
from summary import summarize

video_path=r"C:\Users\Aniket\Desktop\testting.mp4"
cap=cv2.VideoCapture(video_path)




last_vlm_frame=0
frames=[]
gap=5
count=0
vlm_responses=[]


fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 3)  # every 3 seconds of video
print(f"FPS: {fps}, VLM will trigger every {frame_interval} frames")


while True:
    ret,frame=cap.read()

    if not ret:
        break

    count+=1
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frames.append(gray)

    if len(frames)>gap+1:
        frames.pop(0)
    
    #cv2.putText(frame,f'frame count {count}',(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    if len(frames)>gap:
        diff=cv2.absdiff(frames[0],frames[-1])
        _, thresh=cv2.threshold(diff,30,255,cv2.THRESH_BINARY)
        contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # for c in contours:
        #     if cv2.contourArea(c)<2000:
        #         continue

        #     x,y,w,h=cv2.boundingRect(c)
        #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        motion= any(cv2.contourArea(c)>4500 for c in contours)

        if motion:
            
            if count - last_vlm_frame >= frame_interval:
                print(f"\n[Frame {count}] Motion detected! Querying VLM...")
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                #ipy_display(pil_image)
                response = query_model(pil_image, "describe the activity being done in the image")
                print(f"VLM: {response}\n")
                last_vlm_frame = count
                vlm_responses.append(response)
                if len(vlm_responses)==5:
                    print("--- LLM SUMMARY ---")
                    print(summarize(vlm_responses))
                    print("-------------------")
                    vlm_responses.clear()
            #print("MOTIONNNN!!!!!!!!!!")
          #  cv2.imwrite(f"motion_frame_{count}.jpg",frame)
           # print(f"saved : motion_frame_{count}.jpg")

        #cv2.imshow("motion detected",frame)
        

        if cv2.waitKey(1) & 0xFF ==27:
            break
    
cap.release()
cv2.destroyAllWindows()



        



