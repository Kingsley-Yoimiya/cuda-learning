import torch
from torch import nn
from transformers import Trainer, TrainingArguments#, BertConfig, BertModel
from datasets import Dataset
import copy
import math
from torch.utils.cpp_extension import load_inline

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Current device:", device)

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
        self.a = nn.Parameter(torch.ones(size, device=device))
        self.b = nn.Parameter(torch.zeros(size, device=device))
        self.eps = eps
    def forward(self, x):
        # print(x.shape)
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a * (x - mean) / (std + self.eps) + self.b

with open("transf.cpp", "r") as file:
    cpp_source = file.read()
    
transf_cpp = load_inline(
    name="transf_cpp",
    cpp_sources=cpp_source,
    functions=["forward"]
)

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.D = config.hidden_size
        self.WQ = torch.nn.Parameter(torch.empty((self.D, self.D), device=device))
        self.BQ = torch.nn.Parameter(torch.empty((self.D), device=device))
        self.WK = torch.nn.Parameter(torch.empty((self.D, self.D), device=device))
        self.BK = torch.nn.Parameter(torch.empty((self.D), device=device))
        self.WV = torch.nn.Parameter(torch.empty((self.D, self.D), device=device))
        self.BV = torch.nn.Parameter(torch.empty((self.D), device=device))
        self.WX = torch.nn.Parameter(torch.empty((self.D, self.D), device=device))
        self.BX = torch.nn.Parameter(torch.empty((self.D), device=device))
        self.WF1 = torch.nn.Parameter(torch.empty((self.D, config.intermediate_size), device=device))
        self.BF1 = torch.nn.Parameter(torch.empty((config.intermediate_size), device=device))
        self.WF2 = torch.nn.Parameter(torch.empty((config.intermediate_size, self.D), device=device))
        self.BF2 = torch.nn.Parameter(torch.empty((self.D), device=device))
        
        self.heads = config.num_attention_heads
        # assert config.hidden_size % self.heads == 0
        self.d_k = self.heads
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.D)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
            
    def forward(self, x):
        return transf_cpp.forward(
            x, self.d_k,
            self.WQ, self.BQ,
            self.WK, self.BK,
            self.WV, self.BV,
            self.WX, self.BX,
            self.WF1, self.BF1,
            self.WF2,  self.BF2,
        )

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
    input_ids = embeddings[:, :-8, :]  # 所有序列，去掉最后一个作为输入
    labels = embeddings[:, 8:, :]  # 所有序列，去掉第一个作为标签

    return Dataset.from_dict({'input_ids': input_ids, 'labels': labels})

# 4. 初始化模型和Trainer
config = Tr_Config(
        hidden_size=108, 
        num_hidden_layers=12, 
        num_attention_heads=8, 
        intermediate_size=3072,
        drop_out = 0.1,
    )
model = ClickPredictionModel(config)

def grad_check():
    B, L, D, K = 5, 32, 24, 10
    d_k = 8
    input_tensor = torch.randn(B, L, D, dtype=torch.double, requires_grad=True, device = device)
    WQ = torch.randn(D, D, dtype=torch.double, requires_grad=True, device = device)
    BQ = torch.randn(D, dtype=torch.double, requires_grad=True, device = device)
    WK = torch.randn(D, D, dtype=torch.double, requires_grad=True, device = device)
    BK = torch.randn(D, dtype=torch.double, requires_grad=True, device = device)
    WV = torch.randn(D, D, dtype=torch.double, requires_grad=True, device = device)
    BV = torch.randn(D, dtype=torch.double, requires_grad=True, device = device)
    WX = torch.randn(D, D, dtype=torch.double, requires_grad=True, device = device)
    BX = torch.randn(D, dtype=torch.double, requires_grad=True, device = device)
    WF1 = torch.randn(D, K, dtype=torch.double, requires_grad=True, device = device)
    BF1 = torch.randn(K, dtype=torch.double, requires_grad=True, device = device)
    WF2 = torch.randn(K, D, dtype=torch.double, requires_grad=True, device = device)
    BF2 = torch.randn(D, dtype=torch.double, requires_grad=True, device = device)
    
    if torch.autograd.gradcheck(transf_cpp.forward, (
        input_tensor, d_k,
        WQ, BQ,
        WK, BK,
        WV, BV, 
        WX, BX,
        WF1, BF1,
        WF2, BF2
    )):
        print("Custom transformer block - grad test Passed.")
    else:
        print("WA")
        assert 0

grad_check()

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
