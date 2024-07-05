import torch
from torch import nn
from transformers import Trainer, TrainingArguments#, BertConfig, BertModel
from datasets import Dataset
import copy
import math

# 1. 定义自定义的Transformer模型，用于序列预测
class Tr_Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def __str__(self):
        return str(self.__dict__)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, size, eps = 1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeros(size))
        self.eps = eps
    def forward(self, x):
        # print(x.shape)
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a * (x - mean) / (std + self.eps) + self.b

def attention(Q, K, V, mask=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    return torch.matmul(p_attn, V)#, p_attn

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.WQ, self.WK, self.WV, self.Lx = clones(nn.Linear(config.hidden_size, config.hidden_size), 4)
        self.heads = config.num_attention_heads
        assert config.hidden_size % self.heads == 0
        self.d_k = config.hidden_size // self.heads
    def forward(self, Q, K, V, mask):
        if mask is not None:
            mask = mask.unsqueeze(1)
        B, L, D = Q.shape
        Q = self.WQ(Q).view(B, -1, L, self.d_k).transpose(1, 2)
        K = self.WK(K).view(B, -1, L, self.d_k).transpose(1, 2)
        V = self.WV(V).view(B, -1, L, self.d_k).transpose(1, 2)
        x = attention(Q, K, V, mask)
        del Q
        del K
        del V
        x = x.transpose(1, 2).contiguous().view(B, L, D)
        return self.Lx(x)
        

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.atten = MultiheadAttention(config)
        self.resi_layers = clones(nn.Sequential(LayerNorm(config.hidden_size), nn.Dropout(config.drop_out)),2)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
    def forward(self, x, mask = None):
        x = self.resi_layers[0](x + self.atten(x, x, x, mask)) 
        x = self.resi_layers[1](x + self.ffn(x))
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = clones(EncoderLayer(config), config.num_hidden_layers)
        self.norm = LayerNorm(config.hidden_size)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
        return self.norm(x) 
    
    
class ClickPredictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(config.hidden_size, config.hidden_size)
    def forward(self, input_ids, attention_mask = None, labels = None):
        B, L, D = input_ids.shape
        outputs = self.encoder(
            x = input_ids, 
            # mask = attention_mask
        ) # Attention_mask should be None in this code(simple)
        
        output = self.classifier(outputs)
        # print(output.shape)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(output.view(-1, D), labels.view(-1, D))
        return (loss, output) if loss is not None else output

# 2. 加载和准备数据集
def create_dataset_from_embeddings(embedding_file):
    # 加载嵌入数据
    embeddings = torch.load(embedding_file)

    # 创建输入和标签
    input_ids = embeddings[:, :-1, :]  # 所有序列，去掉最后一个作为输入
    labels = embeddings[:, 1:, :]  # 所有序列，去掉第一个作为标签

    return Dataset.from_dict({'input_ids': input_ids, 'labels': labels})

# 4. 初始化模型和Trainer
config = Tr_Config(
        hidden_size=108, 
        num_hidden_layers=12, 
        num_attention_heads=12, 
        intermediate_size=3072,
        drop_out = 0.1,
    )
model = ClickPredictionModel(config)


# 3. 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # evaluate each `epoch`
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=2,   # batch size for training
    per_device_eval_batch_size=2,    # batch size for evaluation
    num_train_epochs=3,              # total number of training epochs
    weight_decay=0.01,               # strength of weight decay
)

embedding_file = 'user_history_embedding.pt'
dataset = create_dataset_from_embeddings(embedding_file)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=dataset,               # training dataset
    eval_dataset=dataset                 # evaluation dataset (for simplicity, use the same dataset)
)

# 5. 训练模型
trainer.train()
