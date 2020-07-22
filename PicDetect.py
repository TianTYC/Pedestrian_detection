import cv2

# 判断两个矩形是否为包含关系，包含返回True,不包含返回False
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox < ix and oy < iy and ox+ow > ix+iw and oy+oh > iy+ih

# 在图中绘制矩形框
def draw_person(src,person_xy):
    x, y, w, h = person_xy
    #绘制矩形框
    cv2.rectangle(src,(x, y), (x+w, y+h), (0, 255, 255), 2)
    #opencv添加标签 类型，文本，左上角坐标(整数)，字样，大小，颜色，粗细
    cv2.putText(img, 'person', (x-5, y-5), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
#读取图片
img = cv2.imread('testperson9.jpg')
# video=cv2.VideoCapture("010.mp4")

# 调用
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
found, w = hog.detectMultiScale(img)

found_filtered = []

for ri, r in enumerate(found):  # 同时列出数据下标和数据
    for qi, q in enumerate(found):
        if ri != qi and is_inside(r, q):
            break
        else:
            found_filtered.append(r)  # 把检测出的行人位置放在found_filtered中
    for person in found_filtered:
        draw_person(img, person)

cv2.imshow('detect result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
