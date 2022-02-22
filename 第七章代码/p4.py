from transformers import TFBertForSequenceClassification, TFTrainer, TFTrainingArguments

model = TFBertForSequenceClassification.from_pretrained('bert-large-uncased')
training_args = TFTrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=64, warmup_steps=500, weight_decay=0.01, logging_dir='./logs')

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=tfds_train_dataset,
    eval_dataset=tfds_test_dataset
)

# 微调模型
trainer.train()
# 验证模型效果
trainer.evaluate()
