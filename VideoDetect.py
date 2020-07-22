from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import time


#opencv
def cv_switch(image,strs,local,sizes,color):
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   #因为一般绘图像素点是RGB顺序，而cv2默认BGR，需要转换一下顺序
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg) 
    font = ImageFont.truetype("simhei.ttf",sizes, encoding="utf-8")  #字体文件，字体大小，编码
    draw.text(local, strs, color, font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image



#src为读取的图像标识
#classifier为识别物体的分类
#strs为识别物体的标签
#colors为框的颜色
#minSize为识别物体的最小尺寸，如果待识别物体小于这个尺寸，则不识别
#minSize为识别物体的最大尺寸，如果待识别物体大于这个尺寸，则不识别
def myClassifier(src,classifier,strs,colors,minSize,maxSize):
    gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    obj = classifier.detectMultiScale(gray,scaleFactor = 1.15, minNeighbors = 2,minSize=minSize,maxSize=maxSize)
    for (x,y,w,h) in obj:
        cv2.rectangle(src,(x,y),(x+w,y+h),(255,0,255),2) #绘制框，（x,y）为识别物体的左上角顶点坐标，（w，h）为宽和高
        src=cv_switch(src,strs,(x-15,y-15),15,(0,255,255))#标签字体和颜色=cv.puttext
    return src


#读取需要检测的视频
video=cv2.VideoCapture("./testdemo/012.mp4")

#导入训练完成的数据集（xml文件，opencv自带一部分haarcascades的样本）
# cars=cv2.CascadeClassifier("cars.xml")      #汽车
person=cv2.CascadeClassifier("person.xml")  #人
fullbody=cv2.CascadeClassifier("fullbody.xml")  #全身
upperbody=cv2.CascadeClassifier("upperbody.xml")  #上半身
frontface=cv2.CascadeClassifier("frontface.xml")  #人脸
eye=cv2.CascadeClassifier("eye.xml")  #眼睛
hand=cv2.CascadeClassifier("hand.xml")  #手
while True:
    _,rec=video.read()
    # frame=myClassifier(frame,volkswagen,"汽车",(0,255,255),(40,40),(70,70))
    # #frame = myClassifier(frame, cars, "汽车", (0, 255, 255), (40, 40), (70, 70))
    rec = myClassifier(rec, person, "person", (0, 0, 255), (0, 0), (70, 70))   #深蓝色
    rec = myClassifier(rec, hand, "person", (0, 0, 255), (0, 0), (70, 70))    #天蓝
    rec = myClassifier(rec, fullbody, "person", (0, 0, 255), (0, 0), (70, 70))  #粉色
    rec = myClassifier(rec, upperbody, "person", (255, 0, 255), (0, 0), (270, 300))       #黑色
    rec = myClassifier(rec, frontface, "person", (0, 0, 255), (0, 0), (70, 70))     #红色

    cv2.imshow("rec",rec)
    c = cv2.waitKey(30)

    if c == 27:
        cv2.destroyAllWindows()

        break