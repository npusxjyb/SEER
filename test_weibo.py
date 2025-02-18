import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import logging
from torch.autograd import Variable
from torch.utils.data import DataLoader
from SEER import MultiModal
from tqdm import tqdm
from weibo_dataset import *
from sklearn.manifold import TSNE
from time import time
import pandas as pd

# Set logging verbosity to warning and error levels for transformers
logging.set_verbosity_warning()
logging.set_verbosity_error()

# Set CUDA_VISIBLE_DEVICES to control GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check if CUDA is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


# Helper function to transfer data to the appropriate device
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# Helper function to test the model
def test(rumor_module, test_loader):
    rumor_module.load_state_dict(torch.load('model/weibo/weibo_detection_model_0.929.pth'))  # 从训练的模型加载

    rumor_module.eval()
    loss_f_rumor = torch.nn.CrossEntropyLoss()

    rumor_count = 0
    loss_total = 0
    rumor_label_all = []
    rumor_pre_label_all = []
    text_all = torch.empty(0,device=device)
    image_all = torch.empty(0, device=device)
    emo_all = torch.empty(0, device=device)
    correlation_all = torch.empty(0, device=device)
    final_feature_all = torch.empty(0, device=device)
    lable_all = torch.empty(0, device=device)
    pre_lable = torch.empty(0, device=device)
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label, prompt_input_ids,prompt_attention_mask, prompt_token_type_ids) in enumerate(test_loader):
            # Transfer data to the appropriate device
            input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label, prompt_input_ids, prompt_attention_mask, prompt_token_type_ids = to_var(input_ids), to_var(attention_mask), to_var(token_type_ids), to_var(image), to_var(imageclip), to_var(textclip), to_var(label), to_var(prompt_input_ids), to_var(prompt_attention_mask), to_var(prompt_token_type_ids)

            # Encode image and text data using pre-trained CLIP models
            image_clip = clipmodel.encode_image(imageclip)
            text_clip = clipmodel.encode_text(textclip)

            # Forward pass through the MultiModal model
            pre_rumor, emo_rumor_lable, text_prime, image_prime, emo_prime, correlation,final_feature = rumor_module(input_ids, attention_mask, token_type_ids, image, text_clip,image_clip, prompt_input_ids, prompt_attention_mask,prompt_token_type_ids)

            text_all = torch.cat((text_all, text_prime), dim=0)
            image_all = torch.cat((image_all, image_prime), dim=0)
            emo_all = torch.cat((emo_all, emo_prime), dim=0)
            correlation_all = torch.cat((correlation_all, correlation), dim=0)
            final_feature_all = torch.cat((final_feature_all, final_feature), dim=0)

            lable_all = torch.cat((lable_all, label), dim=0)

            loss_rumor = loss_f_rumor(pre_rumor, label)
            pre_label_rumor = pre_rumor.argmax(1)

            pre_lable = torch.cat((pre_lable, pre_label_rumor), dim=0)

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

    return acc_rumor_test, loss_rumor_test, conf_rumor,text_all,image_all,emo_all,correlation_all,final_feature_all,lable_all,pre_lable

if __name__ == "__main__":
    batch_size = 16

    validate_set = weibo_dataset(is_train=False)

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
    acc_rumor_test, loss_rumor_test, conf_rumor,text_all,image_all,emo_all,correlation_all,final_feature_all,lable_all,pre_lable = test(rumor_module, test_loader)



    # Print results
    print('-----------rumor detection----------------')
    print(
        "|| acc_rumor_test = %.3f || loss_rumor_test = %.3f" %
        (acc_rumor_test, loss_rumor_test))

    print('-----------rumor_confusion_matrix---------')
    print(conf_rumor)