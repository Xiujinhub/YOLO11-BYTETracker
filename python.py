from ultralytics import YOLO
import cv2

model = YOLO(r"weights\best.pt")
results = model.predict(source=r"E:\1\data_small\images\0_1.jpg", conf=0.25)
for r in results:
    print(r)
    img_with_boxes = r.plot()
    # cv2.imshow("Result", img_with_boxes)
    # cv2.waitKey(0)  # 按任意键关闭窗口
    # cv2.destroyAllWindows()