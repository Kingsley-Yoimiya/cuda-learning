import torch
from torch import nn
from transformers import PreTrainedModel, Trainer, TrainingArguments, BertConfig, BertModel
from datasets import Dataset

# 1. 定义自定义的Transformer模型，用于序列预测
class ClickPredictionModel(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertModel(config).encoder  # 直接使用Transformer encoder层
        self.classifier = nn.Linear(config.hidden_size, config.hidden_size)  # 预测下一个嵌入

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # 将输入数据形状调整为 (batch_size, seq_length, hidden_size)
        batch_size, seq_length, hidden_size = input_ids.shape

        # 调用encoder部分的forward函数
        outputs = self.encoder(
            hidden_states=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]  # shape: (batch_size, seq_length, hidden_size)
        logits = self.classifier(sequence_output)  # shape: (batch_size, seq_length, hidden_size)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1, hidden_size), labels.view(-1, hidden_size))

        return (loss, logits) if loss is not None else logits

# 2. 加载和准备数据集
def create_dataset_from_embeddings(embedding_file):
    # 加载嵌入数据
    embeddings = torch.load(embedding_file)

    # 创建输入和标签
    input_ids = embeddings[:, :-1, :]  # 所有序列，去掉最后一个作为输入
    labels = embeddings[:, 1:, :]  # 所有序列，去掉第一个作为标签

    return Dataset.from_dict({'input_ids': input_ids, 'labels': labels})

embedding_file = 'user_history_embedding.pt'
dataset = create_dataset_from_embeddings(embedding_file)

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

# 4. 初始化模型和Trainer
config = BertConfig(vocab_size=1, hidden_size=108, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
model = ClickPredictionModel(config)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=dataset,               # training dataset
    eval_dataset=dataset                 # evaluation dataset (for simplicity, use the same dataset)
)

# 5. 训练模型
trainer.train()
