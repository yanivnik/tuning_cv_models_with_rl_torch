import os
import torch
from tasks import object_detection
from utils import set_deterministic
from settings import datasets_dir
from utils import get_tb_logger

# TODO: ADD COMMAND LINE ARGUMENTS FOR TASK SELECTION, PARAMETER SETTING, ETC.

def main():
    set_deterministic()

    detection = object_detection.DetectionTask(datasets_dir, batch_size=16)
    original_mAP, original_recall = detection.evaluate()
    # original_mAP, original_recall = torch.tensor(0.3959), torch.tensor(0.4857) # Found after total evaluation, hardcoded for now to save time. TODO REMOVE
    print(f'Pre-finetuning meanAP: {original_mAP}, Avg Recall@100: {original_recall}')

    tb_logger = get_tb_logger()
    detection.reward_finetune(tb_logger, steps=100_000)
    finetuned_mAP, finetuned_recall = detection.evaluate()
    print(f'Post-finetuning meanAP: {finetuned_mAP}, Avg Recall@100: {finetuned_recall}')

    torch.save(detection.model.state_dict(), os.path.join(tb_logger.log_dir, 'model.pt'))

if __name__ == '__main__':
    main()
