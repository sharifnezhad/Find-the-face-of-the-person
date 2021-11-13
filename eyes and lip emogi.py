import cv2
from skimage import transform, img_as_float


def emoji_eyes_smile(frame,eye_img,smil_img):


    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_detector.detectMultiScale(frame_gray,1.5)
    frame = img_as_float(frame)
    for (x,y,w,h) in face:
        roi_gray=frame_gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w,:]
        roi_color2=frame[y:y+h,x:x+w,:]

        eyes=eye_detector.detectMultiScale(roi_gray,1.3,minNeighbors=8)
        for (ex,ey,ew,eh) in eyes:
            # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye_resize = transform.resize(eye_img, (ew,eh))
            eye_img = img_as_float(eye_resize)
            roi_color[ey:ey+eh,ex:ex+ew, 0] *= 1 - eye_img[:,:,3]
            roi_color[ey:ey+eh,ex:ex+ew, 1] *= 1 - eye_img[:,:,3]
            roi_color[ey:ey+eh,ex:ex+ew, 2] *= 1 - eye_img[:,:,3]
            roi_color[ey:ey+eh,ex:ex+ew, :] += eye_img[:,:,:3]

        smile=smile_detector.detectMultiScale(roi_gray,1.5,minNeighbors=14)
        for (sx,sy,sw,sh) in smile:
            # cv2.rectangle(roi_color2,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
            smail_resize = transform.resize(smil_img, (sh,sw))
            smil_img = img_as_float(smail_resize)

            roi_color2[sy:sy+sh,sx:sx+sw, 0] *= 1 - smil_img[:,:,3]
            roi_color2[sy:sy+sh,sx:sx+sw, 1] *= 1 - smil_img[:,:,3]
            roi_color2[sy:sy+sh,sx:sx+sw, 2] *= 1 - smil_img[:,:,3]
            roi_color2[sy:sy+sh,sx:sx+sw, :] += smil_img[:,:,:3]
    return frame

face_detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")

cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("This is the fps ", fps)

while True:
    ret, frame = cap.read()

    if ret == False:
        break
    eye_img = cv2.imread('eye.png', cv2.IMREAD_UNCHANGED)
    smil_img = cv2.imread('smil.png', cv2.IMREAD_UNCHANGED)
    if eye_img.shape[2] < 4 and smil_img.shape[2] < 4:
        print('sorry can\'t mask')
    frame = emoji_eyes_smile(frame, eye_img, smil_img)


    cv2.imshow('face emoji', frame)
    cv2.waitKey(10)
