import torch
import torchvision
from transformers import DetrForObjectDetection, DetrFeatureExtractor, DetrImageProcessor

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tasks.coco_detection_dataset import CocoDetection
from utils import to_device
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


class DetectionTask(object):
    def __init__(self, 
                 root_data_dir: str, 
                 batch_size: int = 4):
        super().__init__()
        self.model, self.image_processor = self._build_model()
        self.train_loader = self._build_dataloader(root_data_dir, batch_size=batch_size, split='train')
        self.val_loader = self._build_dataloader(root_data_dir, batch_size=batch_size, split='val')

    def _build_model(self, device='cuda'):
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", ignore_mismatched_sizes=True).to(device=device)
        image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
        return model, image_processor

    def _build_dataloader(self, root_dir, batch_size=4, split='train'):
        assert split in ['train', 'val', 'test']
        shuffle = (split == 'train')
        dataset = CocoDetection(root_dir, split)
        dataloader = torch.utils.data.DataLoader(dataset, collate_fn=CocoDetection.coco_collate_fn, 
                                                 batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def evaluate(self, device='cuda', verbose=True):
        """
        Evaluate an object detection model, and return - 
            1. The mean average precision (mAP) of the model on the given dataset.
            2. The average recall (at max 100 predictions per image) of the model on the given dataset.
        """
        self.model.eval()
        metric = MeanAveragePrecision(max_detection_thresholds=[100])
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if verbose:
                    print(f'Batch {i} / {len(self.val_loader)}')
                pixel_values, pixel_mask, targets, orig_sizes = batch.values()
                preds = DetectionTask.detect_objects(self.model, self.image_processor, pixel_values, pixel_mask, orig_sizes, device)
                metric.update(preds, to_device(targets, device))
        
        results = metric.compute()
        return results['map'], results['mar_100']
    
    def reward_finetune(self, 
                        tb_logger: SummaryWriter, 
                        steps: int = 100_000, 
                        device: str = 'cuda'):
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-6)

        dataloader_iter = iter(self.train_loader)
        for step in tqdm(range(steps)):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                batch = next(dataloader_iter)

            pixel_values, pixel_mask, targets, orig_sizes = batch.values()
            preds =  DetectionTask.detect_objects(self.model, self.image_processor, pixel_values, pixel_mask, orig_sizes, device)
            targets = to_device(targets, device)
            # baseline = ... # TODO UNDERSTAND WHAT SHOULD BE USED AS THE BASELINE, IF ANY
            rewards = DetectionTask.compute_reward(preds, targets, type='recall').to(device=device)

            # We treat the average confidence of all BBox predictions in a given image as the likelihood estimate
            avg_confidences = torch.stack([torch.mean(p['scores']).nan_to_num() for p in preds]).clamp(1e-20, 1.0)
            loss = -1 * torch.mean(torch.log(avg_confidences) * rewards) # -1 because we want to perform gradient ascent

            optim.zero_grad()
            loss.backward()
            optim.step()

            tb_logger.add_scalar('detection/reward', torch.mean(rewards).nan_to_num(), step)
            tb_logger.add_scalar('detection/loss', loss, step)

    @staticmethod
    @torch.no_grad()
    #def compute_reward(preds, targets, type='map'):
    def compute_reward(preds, targets, type='map'):
        """
        Computes reward for a given batch of images.
        The reward can promote the model to detect objects with high IoU or high mAP.
        """
        if type == 'recall':
            recall_rewards = torch.zeros(len(preds), dtype=torch.float32).requires_grad_(False)
            iou_threshold = 0.8
            for i in range(len(preds)):
                if len(preds[i]['labels']) == 0 or len(targets[i]['labels']) == 0:
                    continue
                classes = targets[i]['labels'].unique().tolist()
                for cls in classes:
                    target_boxes = targets[i]['boxes'][targets[i]['labels'] == cls]
                    pred_boxes = preds[i]['boxes'][preds[i]['labels'] == cls]

                    # Compute IoU matrix between all ground truth boxes and all predicted boxes,
                    # and look at the IoUs which are above some threshold
                    iou_matrix = torchvision.ops.box_iou(target_boxes, pred_boxes)
                    iou_matrix = iou_matrix > iou_threshold

                    # Check how many GT boxes (rows in the matrix) have at least one prediction with IoU > threshold
                    count_of_matched_gt_boxes = torch.any(iou_matrix, dim=-1).sum()

                    # Check how many predictions are duplicate (i.e. - their GT box was already matched to another prediction)
                    count_of_duplicate_boxes = (torch.sum(iou_matrix, dim=-1) - 1).clamp(0).sum()

                    # Compute reward as the number of matched ground truth boxes, minus the number of duplicate predictions.
                    recall_rewards[i] += (count_of_matched_gt_boxes - count_of_duplicate_boxes * 0.3).cpu()

                recall_rewards[i] /= len(classes) # Average over all classes

            return recall_rewards
        elif type == 'map':
            # This is a different implementation from the paper, for a per-example mAP reward.
            # It currently has some problems, as I didn't fully understand which supervised loss did the authors use
            # in this section and how did it combine with the recall reward.
            mean_aps_sep = [MeanAveragePrecision()([preds[i]], [targets[i]])['map'].detach() for i in range(len(preds))]
            return torch.stack(mean_aps_sep)
        else:
            raise NotImplementedError


    @staticmethod
    def detect_objects(model: DetrForObjectDetection, 
                      feature_extractor: DetrFeatureExtractor, 
                      pixel_values: torch.tensor,
                      pixel_mask: torch.tensor,
                      orig_sizes: list,
                      device: str = 'cuda'):
        outputs = model(pixel_values=pixel_values.to(device), pixel_mask=pixel_mask.to(device))
        results = feature_extractor.post_process_object_detection(outputs, target_sizes=orig_sizes)
        return results