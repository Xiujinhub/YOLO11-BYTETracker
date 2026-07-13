import os.path as osp
import cv2
import time
import os
from loguru import logger
from tracking.byte_tracker import BYTETracker
from ultralytics import YOLO
import numpy as np
import time

class MyTimer(object):
    """ 时间计时器实例"""

    def __init__(self):
        """
        total_time: 累积的总时间。
        calls: 计时器被调用的次数。
        start_time: 开始计时的时间点。
        diff: 单次计时的差值。
        average_time: 平均每次调用的时间。
        duration: 最近一次计时的结果，可以是平均时间或单次时间差。
        """
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.

    # 启动计时器。
    def start(self):
        """ 使用 time.time() 而不是 time.clock()，因为 time.clock() 在多线程环境中可能不准确。"""
        # 记录当前时间作为计时的起点。
        self.start_time = time.time()

    # 停止计时器并计算时间差。
    def stop(self, average=True):
        """ average: 如果为 True，返回平均时间；否则，返回单次时间差。"""
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    # 重置计时器。
    def clear(self):
        """ 将所有计时器相关变量重置为初始状态。"""
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None, class_ids=None, class_names=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])

        # 构造显示文本：类别名 + ID
        if class_ids is not None and class_names is not None and i < len(class_ids):
            cls_id = class_ids[i]
            if cls_id >= 0 and cls_id in class_names:
                id_text = f"{class_names[cls_id]}:{obj_id}"
            else:
                id_text = f"{obj_id}"
        else:
            id_text = f"{obj_id}"

        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


class PredictorYolo11:
    def __init__(self, model_path, input_size=(640, 640)):
        self.model = YOLO(model_path)
        self.input_size = input_size

    def predict(self, image: np.ndarray, timer=None):
        timer.start()
        img_info = {"id": 0, "file_name": None}
        height, width = image.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = image
        img_info["ratio"] = min(self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1])
        tensorboard_data = None
        # --------推理---------
        results_list = self.model.predict(
            source=image,
            task='detect',
            conf=0.1
        )
        for results in results_list:  # 遍历检测结果,这是一个ultralytics.engine.results.Results对象
            # tensorboard_data = results.boxes.numpy().data  # 推理的原始张量的numpy形式的data数据
            tensorboard_data = results.boxes.data  # 推理的原始张量的numpy形式的data数据
        return tensorboard_data, img_info


# 检测图像的类型
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def iou(box1, box2):
    """计算两个框的IoU，box1和box2格式均为[x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / (union_area + 1e-9)


""" 视频推理的演示方法 demo """
def imageflow_demo(predictor, vis_folder, current_time, args):
    """
    Args:
        predictor: # 检测头
        vis_folder:  # 可视化视频输出文件夹
        current_time: # 当前时间，用来命名可视化文档和视频
        args: # 参数类
    Returns:
    """
    # ------------ video or camera ------------
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 可视化视频 fps 与原视频保持一致
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)   # 可视化文件夹
    os.makedirs(save_folder, exist_ok=True)
    logger.info(f"save_folder{save_folder}")
    if args.demo == "video":
        filename = os.path.basename(args.path)
        save_path = osp.join(save_folder, filename)
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    # 保持输出的可视化视频与原视频名字一致
    logger.info(f"video save_path is {save_path}")
    # 创建一个MP4写入对象
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
    # 初始化跟踪器
    tracker = BYTETracker(args, frame_rate=fps)  # 根据帧率决定缓存区
    # 初始化计时器
    timer = MyTimer()
    # 检测的图像标记帧数号
    frame_id = 0
    results = []
    while True:
        # 检测三十次日志输出一次
        if frame_id % args.counts == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            # 推理帧图像，
            # cv2.imshow("image", frame)
            # cv2.waitKey(0)
            outputs, img_info = predictor.predict(frame, timer)
            # print("yolo11:", outputs)
            if outputs is not None:
                # 提取检测框和类别
                # outputs形状: N x (x1,y1,x2,y2,conf,cls_id)
                det_boxes = outputs[:, :4].cpu().numpy() if hasattr(outputs, 'cpu') else outputs[:, :4]
                det_scores = outputs[:, 4].cpu().numpy() if hasattr(outputs, 'cpu') else outputs[:, 4]
                det_cls_ids = outputs[:, 5].cpu().numpy().astype(int) if hasattr(outputs, 'cpu') else outputs[:, 5].astype(int)

                # logger.info(f"检测头推理的结果：{outputs[0]}")   # 二维张量， 每一行都是7个数
                online_targets = tracker.update(outputs, [img_info['height'], img_info['width']], img_size=args.input_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_class_ids = []   # 存储每个跟踪目标对应的类别ID

                # print('online_targets', online_targets)
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    # print("tlwh[2] / tlwh[3]", tlwh[2] / tlwh[3])
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    # if tlwh[2] * tlwh[3] > args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                        # 为当前跟踪目标匹配一个类别ID（基于IoU最大的检测框）
                        best_iou = 0
                        best_cls = -1
                        t_box = [tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]]  # 转换为[x1,y1,x2,y2]
                        for i in range(len(det_boxes)):
                            iou_val = iou(t_box, det_boxes[i])
                            if iou_val > best_iou:
                                best_iou = iou_val
                                best_cls = det_cls_ids[i]
                        online_class_ids.append(best_cls)

                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.stop()
                # 获取类别名称字典
                class_names_dict = predictor.model.names
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time,
                    class_ids=online_class_ids, class_names=class_names_dict
                )
            else:
                timer.stop()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

def main(args):
    logger.info(f"args.model_path:{args.model_path}")
    predictor = PredictorYolo11(model_path=args.model_path, input_size=args.input_size)
    current_time = time.localtime()
    imageflow_demo(predictor, args.save_result, current_time, args)


class Args:
    # config
    demo= "video"
    path = r"E:\code\27-ReconTrack\HkcTracker-master\data\videos\car-1.mp4"
    # path = r"D:\kend\work\Hk_Tracker\data\dataset\test_images"
    save_result= r"vis_folder\demo_output"
    fps= 50
    counts = 50
    # model
    model_path = r"weight\best.pt"
    input_size = (640, 640)
    fp16 = False  # cpu no half
    # tracker
    track_thresh = 0.5  # 追踪置信度阈值,低于此值则二次匹配追踪
    track_buffer = 30    # 缓存帧数，决定最大丢失时间 2秒
    match_thresh = 0.8   # 代价匹配阈值，太低容易目标id被重合的id带走
    min_box_area = 6     # 最小边界框面积阈值
    aspect_ratio_thresh = 3 # 边界框宽高比阈值, 过滤掉过于“扁平”的检测框


if __name__ == "__main__":
    # args = make_parser().parse_args()
    args = Args()
