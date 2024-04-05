import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import datetime

import pypyodbc as odbc
import sys
#from PIL import Image
#import numpy as np


# SQL Connection #
DRIVER = 'SQL Server'
SERVER_NAME = 'MIDHUN\SQLEXPRESS'
DATABASE_NAME = 'bagCounter'

conn_string = f"""
            Driver={{{DRIVER}}};
            Server={SERVER_NAME};
            Database={DATABASE_NAME};
            Trust_Connection=yes;
        """

try:
    conn = odbc.connect(conn_string)
    print('connecting...')
except Exception as e:
    print(e)
    print('Connection Terminated')
    sys.exit()
else:
    print('Success')
    cursor = conn.cursor()


def left_click_detect(event, x, y, flags, points):
    if (event == cv2.EVENT_LBUTTONDOWN):
        print(f"\tClick on {x}, {y}")

model = YOLO('./runs/detect/train/weights/best.pt')
tracker=Tracker()

count = 0
cy1 = 560
cy2 = 695
offset = 5
ids=[]

cap = cv2.VideoCapture('Packer2D4.mp4')

down = {}
counter_down = set()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    result = results[0]
    output_image = frame
    #img = Image.fromarray(result.plot()[:, :, :: -1]).convert('RGB')
    #open_cv_image = np.array(img)
    #open_cv_image = open_cv_image[:, :, ::-1].copy()
    #output_image = cv2.resize(frame, (900,800))

    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    #print(px)

    list = []

    for index, row in px.iterrows():
        #print(row)
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        list.append([x1, y1, x2, y2])
    #print("List:",list)


    bbox_id = tracker.update(list)
    #print(bbox_id)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(output_image, (cx, cy), 4, (0, 0, 255), -1)  # draw ceter points of bounding box
        cv2.putText(output_image, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        if cy1 <= cy and cy2 >= cy:
            cv2.rectangle(output_image, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
        if (cy == cy1 and cy < cy2) or (cy1 < (cy+offset) and cy1 > (cy - offset)):
            if id in ids:
                break
            else:
                count+=1
                ids.append(id)
                current_time = datetime.datetime.now().replace(microsecond=0)
                print("Count:",count," at:",current_time)
                try:
                    add_str = f"""USE {DATABASE_NAME} INSERT INTO BagDetails VALUES('{current_time}',{count})"""
                    #print(add_str)
                    cursor.execute(add_str)
                except Exception as e:
                    print(e)

        elif (cy == cy2) or (cy2 < (cy+offset) and cy2 > (cy - offset)):
            if id in ids:
                break
            else:
                count+=1
                ids.append(id)
                current_time = datetime.datetime.now().replace(microsecond=0)
                print("Count:", count, " at:", current_time)
                try:
                    add_str = f"""USE {DATABASE_NAME} INSERT INTO BagDetails VALUES('{current_time}',{count})"""
                    #print(add_str)
                    cursor.execute(add_str)
                except Exception as e:
                    print(e)




    text_color = (255, 0, 0)
    green_color = (0, 255, 0)  # (B, G, R)
    red_color = (0,0,255)
    count_color = (0, 0, 255)

    cv2.line(output_image, (540, cy1), (900, cy1), red_color, 2)
    cv2.putText(output_image, ('Counter Line 1'), (520, cy1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(output_image,(400,cy2),(900,cy2),red_color,2)
    cv2.putText(output_image, ('Counter Line 2'), (380, cy2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


    cv2.putText(output_image, ('Count:'+str(count)), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, count_color, 4, cv2.LINE_AA)

    cv2.imshow("RESULT", output_image)
    cv2.setMouseCallback('RESULT', left_click_detect)


    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
cursor.commit()
cursor.close()