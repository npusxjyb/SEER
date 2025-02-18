from random import random
import torch
import torch.nn as nn
import math
import random
import torch.backends.cudnn as cudnn
import numpy as np
import copy
from transformers import  BertConfig, BertModel, SwinModel


# Set a manual seed for reproducibility
manualseed = 666
random.seed(manualseed)
np.random.seed(manualseed)
torch.manual_seed(manualseed)
torch.cuda.manual_seed(manualseed)
cudnn.deterministic = True


# Load BERT model and configure its output
model_name = 'data/bert-base-chinese'
#model_name = 'data/bert-base-multilingual-cased'
config = BertConfig.from_pretrained(model_name, num_labels=2)
config.output_hidden_states = False


# Definition of the Transformer model
class Transformer(nn.Module):
    def __init__(self, model_dimension, number_of_heads, number_of_layers, dropout_probability, log_attention_weights=False):
        super().__init__()
        # All of these will get deep-copied multiple times internally
        mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha)
        self.encoder = Encoder(encoder_layer, number_of_layers)
        self.init_params()
    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    def forward(self, text, image):
        src_representations_batch1 = self.encode(text, image)
        src_representations_batch2 = self.encode(image, text)
        
        return src_representations_batch1, src_representations_batch2

    def encode(self, src1, src2):
        src_representations_batch = self.encoder(src1, src2)  # forward pass through the encoder
        return src_representations_batch

class Encoder(nn.Module):
    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'
        self.encoder_layers = get_clones(encoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(encoder_layer.model_dimension)
    def forward(self, src1, src2):
        # Forward pass through the encoder stack
        for encoder_layer in self.encoder_layers:
            src_representations_batch = encoder_layer(src1, src2)
        return self.norm(src_representations_batch)

class EncoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention):
        super().__init__()
        num_of_sublayers_encoder = 2
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_encoder)

        self.multi_headed_attention = multi_headed_attention

        self.model_dimension = model_dimension

    def forward(self, srb1, srb2):
        encoder_self_attention = lambda srb1, srb2: self.multi_headed_attention(query=srb1, key=srb2, value=srb2)

        src_representations_batch = self.sublayers[0]( srb1, srb2, encoder_self_attention)
        return src_representations_batch

class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self,  srb1, srb2, sublayer_module):
        # Residual connection between input and sublayer output, details: Page 7, Chapter 5.4 "Regularization",
        return  srb1 + self.dropout(sublayer_module(self.norm(srb1), self.norm(srb2)))

class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value,):
        batch_size = query.shape[0]

        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]
        intermediate_token_representations, attention_weights = self.attention(query, key, value)

        if self.log_attention_weights:
            self.attention_weights = attention_weights
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)

        token_representations = self.out_projection_net(reshaped)

        return token_representations

# Utility function to create deep copies of a module
def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])

# Function to count trainable parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to analyze the shapes and names of parameters in a state dict
def analyze_state_dict_shapes_and_names(model):
    print(model.state_dict().keys())

    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')

# Definition of the Unimodal Detection model
class UnimodalDetection(nn.Module):
        def __init__(self, shared_dim=256, prime_dim = 16, pre_dim = 2):
            super(UnimodalDetection, self).__init__()

            self.text_uni = nn.Sequential(
                nn.Linear(1280, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU())

            self.image_uni = nn.Sequential(
                nn.Linear(1536, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU())

            self.prompt_uni = nn.Sequential(
                nn.Linear(1792, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU())

        def forward(self,text_encoding, image_encoding,prompt_encoding):


            text_prime = self.text_uni(text_encoding)
            image_prime = self.image_uni(image_encoding)
            prompt_prime = self.prompt_uni(prompt_encoding)
            return text_prime, image_prime,prompt_prime

# Definition of the emo Unimodal model
class UnimodalEmo(nn.Module):
        def __init__(self, shared_dim=256, prime_dim = 16, pre_dim = 2):
            super(UnimodalEmo, self).__init__()

            self.emo_uni = nn.Sequential(
                nn.Linear(1024, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU())

        def forward(self,emo_encoding):
            emo_prime = self.emo_uni(emo_encoding)
            return emo_prime

# Definition of the Cross-Modal model
class CrossModule(nn.Module):
    def __init__(
            self,
            corre_out_dim=48):
        super(CrossModule, self).__init__()
        self.corre_dim = 1024
        self.c_specific_1 = nn.Sequential(
            nn.Linear(self.corre_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.c_specific_p = nn.Sequential(
            nn.Linear(self.corre_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.c_specific_3 = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image, text1, image1, prompt):
        correlation_out_prompt = self.c_specific_p(torch.cat((text, prompt), 1).float())
        correlation_out = self.c_specific_1(torch.cat((text, image),1).float())
        correlation_out1 = self.c_specific_2(torch.cat((text1, image1),1).float())
        correlation_out2 = self.c_specific_3(torch.cat((correlation_out_prompt, correlation_out, correlation_out1),1))
        return correlation_out2

# Definition of the EmoModulel model
class EmoModule(nn.Module):
    def __init__(self):
        super().__init__()
        #self.LSTM = nn.LSTM(1280, 256,num_layers=1, batch_first=True,bidirectional=True)
        self.LSTM = nn.LSTM(768, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.ffn = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        # 多头注意力层
        self.attention = nn.MultiheadAttention(embed_dim=512,num_heads=8,dropout=0.1)

        self.LSTM1 = nn.LSTM(768, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.ffn1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        # 多头注意力层
        self.attention1 = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1)


    def forward(self, inputs,inputs1):


        lstm_hidden_states, _ = self.LSTM(inputs)

        text = lstm_hidden_states.permute(1, 0, 2)
        output, _ = self.attention(text, text, text)
        output = output.permute(1, 0, 2)

        emo_embedding = torch.sum(output, dim=1) / 300
        ffn_outputs = self.ffn(emo_embedding)
        logits = self.classifier(ffn_outputs)
        pre_label_rumor = logits.argmax(1)

        lstm_hidden_states1, _ = self.LSTM1(inputs1)
        prompt = lstm_hidden_states1.permute(1, 0, 2)
        output1, _ = self.attention1(prompt, prompt, prompt)
        output1 = output1.permute(1, 0, 2)

        emo_embedding1 = torch.sum(output1, dim=1) / 300
        ffn_outputs1 = self.ffn1(emo_embedding1)
        logits1 = self.classifier1(ffn_outputs1)
        pre_label_rumor1 = logits1.argmax(1)

        #lstm_hidden_states = lstm_hidden_states[:, -1, :]

        #ffn_outputs = self.ffn(lstm_hidden_states)

        #logits = self.classifier(ffn_outputs)
        #pre_label_rumor = logits.argmax(1)
        return pre_label_rumor,emo_embedding,pre_label_rumor1,emo_embedding1

# Definition of the MultiModal model
class MultiModal(nn.Module):
    def __init__(
            self,
            feature_dim = 96,
            h_dim =96
            ):
        super(MultiModal, self).__init__()

        # Initialize learnable parameters
        self.w = nn.Parameter(torch.rand(1))        # Learnable parameter for weighting similarity
        self.b = nn.Parameter(torch.rand(1))        # Learnable parameter for biasing similarity

        # Initialize learnable parameters
        self.w1 = nn.Parameter(torch.rand(1))  # Learnable parameter for emo weighting similarity
        self.b1 = nn.Parameter(torch.rand(1))  # Learnable parameter for emo biasing similarity

        self.fake_p = nn.Parameter(torch.rand(1))
        self.real_p = nn.Parameter(torch.rand(1))

        # Initialize the Transformer model for cross-modal attention
        self.trans = Transformer(model_dimension=512,  number_of_heads=8, number_of_layers=1, dropout_probability=0.1, log_attention_weights=False)
        
        # Initialize the Transformer model for cross-modal attention
        self.t_projection_net = nn.Linear(768, 512)         # Linear projection for text
        self.p_projection_net = nn.Linear(768, 512)         # Linear projection for prompt
        self.i_projection_net = nn.Linear(1024, 512)        # Linear projection for image

        # Load the Swin Transformer model for image processing
        self.swin = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224").cuda()
        for param in self.swin.parameters():
            param.requires_grad = True

        # Load BERT model for text processing      
        self.bert = BertModel.from_pretrained(model_name, config = config).cuda()
        for param in self.bert.parameters():
            param.requires_grad = True

         # Initialize unimodal representation modules
        self.uni_repre = UnimodalDetection()

        # Initialize emo unimodal representation modules
        self.uni_emo = UnimodalEmo()

        # Initialize cross-modal fusion module
        self.cross_module = CrossModule()

        # Initialize EmoModule module
        self.emo_module1 = EmoModule()
        self.emo_module2 = EmoModule()
        self.emo_module3 = EmoModule()
        self.emo_module4 = EmoModule()
        self.emo_module5 = EmoModule()
        self.emo_module6 = EmoModule()
        self.emo_module7 = EmoModule()
        self.emo_module8 = EmoModule()
        self.emo_module9 = EmoModule()
        self.emo_module10 = EmoModule()

        # Define classifier layers for final prediction
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2)
        )
        # 多头注意力层
        self.attention1 = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.1)
        self.attention2 = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.1)
        self.attention3 = nn.MultiheadAttention(embed_dim=1024, num_heads=8, dropout=0.1)

    def forward(self, input_ids, attention_mask, token_type_ids, image_raw, text, image, prompt_input_ids, prompt_attention_mask, prompt_token_type_ids):

        # Extract features using BERT for textual input
        BERT_feature = self.bert(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids) 
        last_hidden_states = BERT_feature['last_hidden_state']

        # Extract features using BERT for prompt input
        prompt_BERT_feature = self.bert(input_ids=prompt_input_ids,
                                 attention_mask=prompt_attention_mask,
                                 token_type_ids=prompt_token_type_ids)
        prompt_last_hidden_states = prompt_BERT_feature['last_hidden_state']

        # Compute raw prompt feature by averaging over tokens 16×768
        prompt_raw = prompt_last_hidden_states
        prompt = prompt_raw.permute(1, 0, 2)
        prompt_output, _ = self.attention1(prompt, prompt, prompt)
        prompt_output = prompt_output.permute(1, 0, 2)
        prompt_raw = torch.sum(prompt_output, dim=1)/300

        # Compute raw text feature by averaging over tokens16×768
        text_raw = last_hidden_states
        text_emb = text_raw.permute(1, 0, 2)
        text_output, _ = self.attention2(text_emb, text_emb, text_emb)
        text_output = text_output.permute(1, 0, 2)
        text_raw = torch.sum(text_output, dim=1)/300

        # Process the raw image feature using Swin Transformer
        image_swin = self.swin(image_raw)
        image_last_hidden_state = image_swin.last_hidden_state
        image_emb = image_last_hidden_state.permute(1, 0, 2)
        image_output, _ = self.attention3(image_emb, image_emb, image_emb)
        image_output = image_output.permute(1, 0, 2)
        image_raw = torch.sum(image_output, dim=1)/300

        # Generate unimodal representations for text and image
        text_prime, image_prime, prompt_prime = self.uni_repre(torch.cat([text_raw,text],1),torch.cat([image_raw,image],1),torch.cat([prompt_raw,text,image],1))

        # Project text and image features to a common space
        prompt_m = self.p_projection_net(prompt_last_hidden_states)#16×300×512
        text_m = self.t_projection_net(last_hidden_states)#16×300×512
        image_m =self.i_projection_net(image_last_hidden_state)#16×49×512

        # 共注意力
        text_att1, image_att1 = self.trans(text_m, image_m)
        text_att2, prompt_att1 = self.trans(text_m, prompt_m)
        prompt_att2, image_att2 = self.trans(prompt_m, image_m)

        text_att = (text_att1 + text_att2) / 2
        image_att = (image_att1 + image_att2) / 2
        prompt_att = (prompt_att1 + prompt_att2) / 2

        # Cross-modal fusion using the cross-module
        correlation = self.cross_module(text, image, torch.sum(text_att,dim = 1)/300, torch.sum(image_att,dim = 1)/49, torch.sum(prompt_att,dim = 1)/300)

        # 10个专家
        pre_emo1, emo_text_emd1, pro_emo1, emo_prompt_emd1 = self.emo_module1(last_hidden_states,prompt_last_hidden_states)
        pre_emo2, emo_text_emd2, pro_emo2, emo_prompt_emd2 = self.emo_module2(last_hidden_states,prompt_last_hidden_states)
        pre_emo3, emo_text_emd3, pro_emo3, emo_prompt_emd3 = self.emo_module3(last_hidden_states,prompt_last_hidden_states)
        pre_emo4, emo_text_emd4, pro_emo4, emo_prompt_emd4 = self.emo_module4(last_hidden_states,prompt_last_hidden_states)
        pre_emo5, emo_text_emd5, pro_emo5, emo_prompt_emd5 = self.emo_module5(last_hidden_states,prompt_last_hidden_states)
        pre_emo6, emo_text_emd6, pro_emo6, emo_prompt_emd6 = self.emo_module6(last_hidden_states,prompt_last_hidden_states)
        pre_emo7, emo_text_emd7, pro_emo7, emo_prompt_emd7 = self.emo_module7(last_hidden_states,prompt_last_hidden_states)
        pre_emo8, emo_text_emd8, pro_emo8, emo_prompt_emd8 = self.emo_module8(last_hidden_states,prompt_last_hidden_states)
        pre_emo9, emo_text_emd9, pro_emo9, emo_prompt_emd9 = self.emo_module9(last_hidden_states,prompt_last_hidden_states)
        pre_emo10, emo_text_emd10, pro_emo10, emo_prompt_emd10 = self.emo_module10(last_hidden_states,prompt_last_hidden_states)

        text_emo_final = (pre_emo1 + pre_emo2 + pre_emo3 + pre_emo4 + pre_emo5 + pre_emo6 + pre_emo7 + pre_emo8 + pre_emo9 + pre_emo10) / 10
        emo_text_emb = (emo_text_emd1 + emo_text_emd2 + emo_text_emd3 + emo_text_emd4 + emo_text_emd5 + emo_text_emd6 + emo_text_emd7 + emo_text_emd8 + emo_text_emd9 + emo_text_emd10) / 10
        prompt_emo_final = (pro_emo1 + pro_emo2 + pro_emo3 + pro_emo4 + pro_emo5 + pro_emo6 + pro_emo7 + pro_emo8 + pro_emo9 + pro_emo10) / 10
        emo_prompt_emb = (emo_prompt_emd1 + emo_prompt_emd2 + emo_prompt_emd3 + emo_prompt_emd4 + emo_prompt_emd5 + emo_prompt_emd6 + emo_prompt_emd7 + emo_prompt_emd8 + emo_prompt_emd9 + emo_prompt_emd10) / 10
        #text_emo_final = (pre_emo1 + pre_emo2 ) / 2
        #emo_text_emb = (emo_text_emd1 + emo_text_emd2 ) / 2
        #prompt_emo_final = (pro_emo1 + pro_emo2 ) / 2
        #emo_prompt_emb = (emo_prompt_emd1 + emo_prompt_emd2 ) / 2

        λ = 0.75
        emo_final = λ*text_emo_final+(1-λ)*prompt_emo_final

        #sim_emo = torch.div(torch.sum(emo_text_emb * emo_prompt_emb, 1),torch.sqrt(torch.sum(torch.pow(emo_text_emb, 2), 1)) * torch.sqrt(torch.sum(torch.pow(emo_prompt_emb, 2), 1)))

        emo_prime = self.uni_emo(torch.cat([λ*emo_text_emb, (1-λ)*emo_prompt_emb], 1))
        #sim_emo = sim_emo * self.w1 + self.b1
        #emoweight = sim_emo.unsqueeze(1)

        # Weighted cross-modal fusion
        #emo_prime = emo_prime * emoweight

        # 1正面，0负面
        emo_p = emo_final
        emo_n = 1 - emo_final

        # 真新闻正面a，负面1-a
        a = 0.45
        #real_p = self.real_p
        real_p = 0.45
        # 假新闻中正面b,负面1-b
        b = 0.3
        #fake_p = self.fake_p
        fake_p = 0.3

        # real_news_p = 0.5,real_news_n = 0.5,fake_news_p = a,fake_news_n = 1-a

        p_real = (real_p * 0.5) / (fake_p * 0.5 + real_p * 0.5)
        p_fake = 1 - p_real
        n_real = (0.5 * (1 - real_p)) / (0.5 * (1 - real_p) + 0.5 * (1 - fake_p))
        n_fake = 1 - n_real

        emo_real = (emo_p * p_real + emo_n * n_real).unsqueeze(1)
        emo_fake = (emo_p * p_fake + emo_n * n_fake).unsqueeze(1)

        emo_rumor_lable = torch.cat((emo_fake, emo_real), dim=1)
        emo_rumor_pre = emo_rumor_lable.argmax(1)

        # Compute CLIP similarity between text and image features
        sim = torch.div(torch.sum(text * image,1),torch.sqrt(torch.sum(torch.pow(text,2),1))* torch.sqrt(torch.sum(torch.pow(image,2),1)))
       
        # Apply learned weighting and bias to similarity
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)

        # Weighted cross-modal fusion
        correlation = correlation * mweight

        # Combine all features for final prediction

        final_feature = torch.cat([text_prime, image_prime, emo_prime, correlation], 1)


        # final prediction
        pre_label = self.classifier_corre(final_feature)

        return pre_label,emo_rumor_lable
