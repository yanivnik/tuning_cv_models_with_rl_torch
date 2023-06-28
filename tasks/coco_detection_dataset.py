# Code based on: https://github.com/roboflow/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb

from transformers import DetrImageProcessor
import torchvision
import os

from tasks.detection_utils import relative_to_absolute_bboxes

class CocoDetection(torchvision.datasets.CocoDetection):
    image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

    def __init__(self, 
                 root_dir: str, 
                 split: str
                 ):
        data_dir = os.path.join(root_dir, f'{split}2017')        
        annotation_file_path = os.path.join(root_dir, 'annotations', f'instances_{split}2017.json')
        super().__init__(data_dir, annotation_file_path)


    def __getitem__(self, idx):
        image, annotations = super().__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=image, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        orig_size = image.size[::-1]
        target['boxes'] = relative_to_absolute_bboxes(target['boxes'], orig_size) # Convert [0,1] coordinates to absolute [0,HW] coordinates
        return pixel_values, target, orig_size
    
    @staticmethod
    def coco_collate_fn(batch):
        # Collate func to pad values in batch to the same size (maximum size of element in batch).

        pixel_values = [item[0] for item in batch]
        labels_and_bboxes = [{'boxes': item[1]['boxes'], 
                              'labels': item[1]['class_labels']} 
                              for item in batch]
        encoding = CocoDetection.image_processor.pad(pixel_values, return_tensors="pt")
        
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels_and_bboxes': labels_and_bboxes,
            'orig_sizes': [item[2] for item in batch]
        }
