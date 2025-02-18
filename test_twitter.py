import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import logging
from torch.autograd import Variable
from torch.utils.data import DataLoader
from SEER_twitter import MultiModal
from tqdm import tqdm
from twitter_dataset import *

# Set logging verbosity to warning and error levels for transformers
logging.set_verbosity_warning()
logging.set_verbosity_error()

# Set CUDA_VISIBLE_DEVICES to control GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Check if CUDA is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


# Helper function to transfer data to the appropriate device
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# Helper function to test the model
def test(rumor_module, test_loader):
    rumor_module.load_state_dict(torch.load('model/twitter/twitter_detection_model_0.931.pth'))  # 从训练的模型加载

    rumor_module.eval()
    loss_f_rumor = torch.nn.CrossEntropyLoss()

    rumor_count = 0
    loss_total = 0
    rumor_label_all = []
    rumor_pre_label_all = []
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label) in enumerate(test_loader):
            # Transfer data to the appropriate device
            input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label = to_var(input_ids), to_var(
                attention_mask), to_var(token_type_ids), to_var(image), to_var(imageclip), to_var(textclip), to_var(
                label)

            # Encode image and text data using pre-trained CLIP models
            image_clip = clipmodel.encode_image(imageclip)
            text_clip = clipmodel.encode_text(textclip)

            # Forward pass through the MultiModal model
            pre_rumor = rumor_module(input_ids, attention_mask, token_type_ids, image, text_clip, image_clip)
            loss_rumor = loss_f_rumor(pre_rumor, label)
            pre_label_rumor = pre_rumor.argmax(1)

            loss_total += loss_rumor.item() * input_ids.shape[0]
            rumor_count += input_ids.shape[0]

            # Collect predictions and labels
            rumor_pre_label_all.append(pre_label_rumor.detach().cpu().numpy())
            rumor_label_all.append(label.detach().cpu().numpy())

        # Calculate accuracy and confusion matrix
        loss_rumor_test = loss_total / rumor_count
        rumor_pre_label_all = np.concatenate(rumor_pre_label_all, 0)
        rumor_label_all = np.concatenate(rumor_label_all, 0)

        acc_rumor_test = accuracy_score(rumor_pre_label_all, rumor_label_all)
        conf_rumor = confusion_matrix(rumor_pre_label_all, rumor_label_all)

    return acc_rumor_test, loss_rumor_test, conf_rumor


if __name__ == "__main__":
    batch_size = 16

    validate_set = twitter_dataset(is_train=False)

    test_loader = DataLoader(
        validate_set,
        batch_size=batch_size,
        num_workers=8,
        # num_workers=8,
        collate_fn=collate_fn,
        shuffle=False
    )

    # Initialize the MultiModal model
    rumor_module = MultiModal()
    rumor_module.to(device)


    # Evaluate on the test set
    acc_rumor_test, loss_rumor_test, conf_rumor = test(rumor_module, test_loader)

    # Print results
    print('-----------rumor detection----------------')
    print(
        "|| acc_rumor_test = %.3f || loss_rumor_test = %.3f" %
        (acc_rumor_test, loss_rumor_test))

    print('-----------rumor_confusion_matrix---------')
    print(conf_rumor)