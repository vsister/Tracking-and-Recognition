from xml.dom import minidom
import cv2


file = open('P2L_S4/P2L_S4_C3.2.txt','r')
dom = minidom.parse("P2L_S4/P2L_S4_C3.2.xml")
dom.normalize()
frames=dom.getElementsByTagName("frame")
number_of_frames = len(frames)
for i in range(number_of_frames):
    frame = frames[i]
    if len(frame.childNodes)>0:
        number = frame.getAttribute("number")
        #print(frame.childNodes)
    for child in frame.childNodes:
        if child.nodeType != child.TEXT_NODE:
            id = child.getAttribute("id")
            x1 = int(child.getElementsByTagName("leftEye")[0].getAttribute("x"))
            x2 = int(child.getElementsByTagName("rightEye")[0].getAttribute("x"))
            y1 = int(child.getElementsByTagName("leftEye")[0].getAttribute("y"))
            y2 = int(child.getElementsByTagName("rightEye")[0].getAttribute("y"))
            w = x2 - x1 + 1
            x3 = x1 - round(w/2)
            y3 = max(y1, y2) - w
            x4 = x2 + round(w/2)
            y4 = min(y1,y2) + w + round(w/2)
            new_line = str(number) + ' ' + str(id) + ' ' + str(x3) + ' ' + str(y3) + ' ' + str(x4) + ' ' + str(y4) + '\n'
            file.write(new_line)

file.close()