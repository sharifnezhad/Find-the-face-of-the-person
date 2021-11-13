import cv2
from skimage import transform, img_as_float


def emoji_face(frame,img):
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_detector.detectMultiScale(frame_gray,1.5)
    frame = img_as_float(frame)
    for x,y,w,h in face:
        img_resize = transform.resize(img, (w,h))
        img = img_as_float(img_resize)
        frame[y:y+h,x:x+w, 0] *= 1 - img_resize[:,:,3]
        frame[y:y+h,x:x+w, 1] *= 1 - img_resize[:,:,3]
        frame[y:y+h,x:x+w, 2] *= 1 - img_resize[:,:,3]
        frame[y:y+h,x:x+w, :] += img_resize[:,:,:3]
    return frame
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("This is the fps ", fps)

while True:
    ret, frame = cap.read()

    if ret == False:
        break
    if cap.isOpened() == False:
        print("Error File Not Found")

    img = cv2.imread("starry-eyed-emoji.png", cv2.IMREAD_UNCHANGED)

    if (img.shape[2] < 4):
        print('sorry can\'t mask')

    frame = emoji_face(frame, img)


    cv2.imshow('face emoji', frame)
    cv2.waitKey(10)
