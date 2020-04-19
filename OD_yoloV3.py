from numpy import expand_dims
from keras.models import load_model
from yoloV3 import make_yolov3_model, WeightReader
from keras.preprocessing.image import load_img, img_to_array
from yolo_utils import decode_netout, correct_yolo_boxes, do_nms, get_boxes, draw_boxes

net_h, net_w = 416, 416
obj_thresh, nms_thresh = 0.5, 0.45
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def load_image_pixels(filename, shape):
    image = load_img(filename)
    width, height = image.size
    image = load_img(filename, target_size=shape)
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0
    image = expand_dims(image, 0)
    return image, width, height


if __name__ == '__main__':
    # model = make_yolov3_model()
    # weight_reader = WeightReader('yolov3.weights')
    # weight_reader.load_weights(model)
    # model.save('model.h5')

    model = load_model('model.h5')
    photo_filename = './test/1e93af02988f37f4.jpg'
    image, image_w, image_h = load_image_pixels(photo_filename, (net_w, net_w))
    yolos = model.predict(image)
    print([a.shape for a in yolos])

    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    class_threshold = 0.6
    boxes = list()
    for i in range(len(yolos)):
        boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    do_nms(boxes, nms_thresh)

    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    draw_boxes(photo_filename, v_boxes, v_labels, v_scores)

