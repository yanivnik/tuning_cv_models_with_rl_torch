import matplotlib.pyplot as plt
import numpy as np
import supervision
import transformers
import torch


def get_detection_image(image, bboxes_and_labels, confidence_threshold=0.9, id2label={}, show_image=True):
    """
    Converts a given image and a list of bounding boxes and labels into a single image with bounding boxes and labels.
    Args:
        image (PIL.Image): Image to annotate.
        bboxes_and_labels (dict): A dict with keys 'boxes', 'labels' and 'scores'. Should be the output of the
                                  DetrImageProcessor.post_process method
        confidence_threshold (float): Minimum confidence of a bounding box to be shown.
        id2label (dict): Dictionary mapping class ids to class names.
        show_image (bool): Whether to show the image or not. The annotated image is returned in both cases.
    """
    detections = supervision.Detections.from_transformers(transformers_results=bboxes_and_labels)
    labels = [f"{id2label[class_id]}({confidence:0.2f})" for _, _, confidence, class_id, _ 
               in detections if confidence > confidence_threshold]
    annotated_image = supervision.BoxAnnotator().annotate(scene=np.array(image),
                                                          detections=detections[detections.confidence > confidence_threshold],
                                                          labels=labels)
    if show_image:
        plt.imshow(annotated_image)
        plt.show()
    return annotated_image


def relative_to_absolute_bboxes(boxes, target_sizes):
    """
    Converts bounding boxes from relative-to-center (float value in [0,1.0]) coordinates 
    to absolute (integer value in [0,H] or [0,W]) coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in relative-to-center format. Shape: [N, 4]
        target_sizes (tuple): Tuple of (height, width) of the target image.
    """
    boxes = transformers.image_transforms.center_to_corners_format(boxes)
    img_h, img_w = target_sizes
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
    return boxes * scale_fct