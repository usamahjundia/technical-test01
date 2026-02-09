import onnxruntime as ort
import numpy as np
import cv2

def letterbox_image(image, target_size):
    
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    h, w = image.shape[:2]
    
    max_dim = max(h, w)
    scale = max_dim / max(target_size)
    
    if h > w:
        pad_total = h - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = 0
        pad_bottom = 0
    else:
        pad_total = w - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        pad_left = 0
        pad_right = 0
    
    if len(image.shape) == 3:
        padded = cv2.copyMakeBorder(
            image,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
    else:
        padded = cv2.copyMakeBorder(
            image,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=114
        )
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_LINEAR)
    return resized, scale, pad_top, pad_left

def detections_to_original(detections, original_image, scale, pad_top, pad_left):
    if len(detections) == 0:
        return detections

    detections = np.array(detections)
    original_detections = detections.copy()

    original_h, original_w = original_image.shape[:2]

    bboxes = detections[:, :4].copy()
    
    bboxes[:, [0, 2]] *= scale
    bboxes[:, [1, 3]] *= scale
    
    bboxes[:, 0] -= pad_left
    bboxes[:, 1] -= pad_top
    
    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, original_w)
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, original_h)
    
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, original_w - bboxes[:, 0])
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, original_h - bboxes[:, 1])
    
    original_detections[:, :4] = bboxes
    
    return original_detections

class RFDETRInference:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self._session = ort.InferenceSession(model_path)
        self._input_size = self._session.get_inputs()[0].shape[2]
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None]
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None]
        self._threshold = threshold
    
    # asusmes RGB
    def preprocess(self, image: np.array):
        preproc_image = image.astype(np.float32) / 255.0
        preproc_image = (preproc_image - self._mean) / self._std
        preproc_image = np.transpose(preproc_image, (2, 0, 1))[None]
        return preproc_image

    def postprocess_detections(self, bboxes, labels):
        bboxes[:, 0] -= bboxes[:, 2]/2
        bboxes[:, 1] -= bboxes[:, 3]/2
        labels_sigmoid = 1/(1+np.exp(-labels))
        class_probs = np.max(labels_sigmoid, axis=1)
        class_index = np.argmax(labels_sigmoid, axis=1)
        mask = class_probs > self._threshold
        return bboxes[mask], class_index[mask]

    def predict(self, image: np.array):
        letterboxed, scale, pad_top, pad_left = letterbox_image(image, self._input_size)
        preprocessed_image = self.preprocess(letterboxed)
        outputs = self._session.run(
            None,
            {
                "input": preprocessed_image
            }
        )
        bbox, labels = outputs
        bbox = bbox[0]
        labels = labels[0]
        bbox *= self._input_size
        bbox_remapped = detections_to_original(bbox, image, scale, pad_top, pad_left)
        bbox_remapped, labels = self.postprocess_detections(bbox_remapped, labels)
        return bbox_remapped, labels

if __name__ == "__main__":
    # test inference script
    detector = RFDETRInference("output/inference_model.onnx")
    image = cv2.imread("facemask_dataset/images/maksssksksss812.png", cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_draw = image.copy()
    output = detector.predict(image_rgb)
    for det in output[0]:
        x, y, w, h = det
        image_draw = cv2.rectangle(image_draw, (int(x), int(y)), (int(x + w), int(y+h)), color=(0, 255, 0), thickness=1)
    cv2.imwrite("detection_result.jpg", image_draw)
    # i had opencv headless, if you install normal opencv you can uncomment below to see the image isntantly
    # cv2.imshow("detection_result", image_draw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        
    
        
        