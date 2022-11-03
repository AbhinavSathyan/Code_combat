import cv2
import math
import time
import numpy as np

limit = 80 #km/hr

file = open("D://project S6//TrafficRecord//SpeedRecord.txt","w")
file.write("ID \t SPEED\n------\t-------\n")
file.close()


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}

        self.id_count = 0
        #self.start = 0
        #self.stop = 0
        self.et=0
        self.s1 = np.zeros((1,1000))
        self.s2 = np.zeros((1,1000))
        self.s = np.zeros((1,1000))
        self.f = np.zeros(1000)
        self.capf = np.zeros(1000)
        self.count = 0
        self.exceeded = 0


    def update(self, objects_rect):
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            #CHECK IF OBJECT IS DETECTED ALREADY
            same_object_detected = False

            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 70:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True

                    #START TIMER
                    if (y >= 410 and y <= 430):
                        self.s1[0,id] = time.time()

                    #STOP TIMER and FIND DIFFERENCE
                    if (y >= 235 and y <= 255):
                        self.s2[0,id] = time.time()
                        self.s[0,id] = self.s2[0,id] - self.s1[0,id]

                    #CAPTURE FLAG for capturing image
                    if (y<235):
                        self.f[id]=1


            #NEW OBJECT DETECTION
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
                self.s[0,self.id_count]=0
                self.s1[0,self.id_count]=0
                self.s2[0,self.id_count]=0

        # ASSIGN NEW ID to OBJECT
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    #SPEEED FUNCTION
    def getsp(self,id):
        if (self.s[0,id]!=0):
            s = 80.15 / self.s[0, id]
        else:
            s = 0

        return int(s)

    #SAVE VEHICLE DATA
    def capture(self,img,x,y,h,w,sp,id):
        if(self.capf[id]==0):
            self.capf[id] = 1
            self.f[id]=0
            crop_img = img[y-5:y + h+5, x-5:x + w+5]
            n = str(id)+"_speed_"+str(sp)
            file = 'D://project S6//TrafficRecord//' + n + '.jpg'
            cv2.imwrite(file, crop_img)
            self.count += 1
            filet = open("D://project S6//TrafficRecord//SpeedRecord.txt", "a")
            if(sp>limit):
                file2 = 'D://project S6//TrafficRecord//exceeded//' + n + '.jpg'
                cv2.imwrite(file2, crop_img)
                filet.write(str(id)+" \t "+str(sp)+"<---exceeded\n")
                self.exceeded+=1
            else:
                filet.write(str(id) + " \t " + str(sp) + "\n")
            filet.close()


    #SPEED_LIMIT
    def limit(self):
        return limit

    #TEXT FILE SUMMARY
    def end(self):
        file = open("D://project S6//TrafficRecord//SpeedRecord.txt", "a")
        file.write("\n-------------\n")
        file.write("-------------\n")
        file.write("SUMMARY\n")
        file.write("-------------\n")
        file.write("Total Vehicles :\t"+str(self.count)+"\n")
        file.write("Exceeded speed limit :\t"+str(self.exceeded))
        file.close()




import cv2
from tracker2 import *
import numpy as np
end = 0

#Creater Tracker Object
tracker = EuclideanDistTracker()

#cap = cv2.VideoCapture("Resources/traffic3.mp4")
cap = cv2.VideoCapture("Project Name.mp4")
f = 25
w = int(1000/(f-1))
print(w)


#Object Detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=None,varThreshold=None)
#100,5

#KERNALS FOR MASKING
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8)
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernal_e = np.ones((5,5),np.uint8)

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    height,width,_ = frame.shape
    #print(height,width)
    #540,960


    #Extract ROI
    roi = frame[50:540,200:960]



    #DIFFERENT MASKING METHOD 
    fgmask = fgbg.apply(roi)
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)
    e_img = cv2.erode(mask2, kernal_e)


    contours,_ = cv2.findContours(e_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        #THRESHOLD
        if area > 1500:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
            detections.append([x,y,w,h])

    #Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id


        if(tracker.getsp(id)<tracker.limit()):
            cv2.putText(roi,str(id)+" "+str(tracker.getsp(id)),(x,y-15), cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:
            cv2.putText(roi,str(id)+ " "+str(tracker.getsp(id)),(x, y-15),cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 165, 255), 3)

        s = tracker.getsp(id)
        if (tracker.f[id] == 1 and s != 0):
            tracker.capture(roi, x, y, h, w, s, id)

    # DRAW LINES

    cv2.line(roi, (0, 410), (960, 410), (0, 0, 255), 2)
    cv2.line(roi, (0, 430), (960, 430), (0, 0, 255), 2)

    cv2.line(roi, (0, 235), (960, 235), (0, 0, 255), 2)
    cv2.line(roi, (0, 255), (960, 255), (0, 0, 255), 2)


    #DISPLAY
    #cv2.imshow("Mask",mask2)
    #cv2.imshow("Erode", e_img)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(w-10)
    if key==27:
        tracker.end()
        end=1
        break

if(end!=1):
    tracker.end()

cap.release()
cv2.destroyAllWindows()
