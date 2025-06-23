import cv2

def display_image(img):
    cv2.namedWindow("MyImage", cv2.WINDOW_NORMAL)
    cv2.imshow("MyImage", img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
