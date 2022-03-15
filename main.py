import cv2
import matplotlib.pyplot as plt




config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = []
file_name = 'names.txt'
with open(file_name, 'rt') as abc:
    classLabels = abc.read().rstrip('\n').split('\n')

print(classLabels)

model.setInputSize(320,320) 
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5)) #mobilenet---> [-1,1]
model.setInputSwapRB(True)  #automatic conversion color


''''''


'''img= cv2.imread('test1.jpg') #forimage mode
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            cv2.rectangle(img, boxes, (255, 0, 0), 2)
            cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
                        color=(0, 255, 0), thickness=3)

cv2.waitKey(0)'''




#video
cap=cv2.VideoCapture("test.mp4")

if not cap.isOpened():
    cap= cv2.VideoCapture(1)

if not cap.isOpened():
    raise IOError("Cannot open video!")



font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN


'''frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
result = cv2.VideoWriter('output.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)'''
while True:
    ret,frame= cap.read()
    ClassIndex, confidece, bbox= model.detect(frame, confThreshold= 0.55)
    print(ClassIndex)


    print(ClassIndex)
    if(len(ClassIndex)!=0):

        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            cv2.rectangle(frame, boxes, (255, 0, 0), 2)
            cv2.putText(frame, classLabels[ClassInd - 1].upper(), (boxes[0] + 10, boxes[1] + 30), font, fontScale=font_scale,
                        color=(0, 255, 0), thickness=2)

            '''cv2.putText(frame, (str(confidece*100),2),(boxes[0]+200,boxes[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)'''




        cv2.imshow('Object Detection', frame)

    if cv2.waitKey(2) & 0xFF== ord('q'):
        break

cap.release()
'''result.release()'''
cv2.destroyAllWindows
