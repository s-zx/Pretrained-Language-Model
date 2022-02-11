# 第四章 GPT系列模型
在自然语言处理领域，最重要的预训练语言模型可以分为两个系列，一个是擅长**自然语言生成任务**的GPT(Generative Pre-Training,生成式预训练)系列模型，另一个是擅长**自然语言理解任务**的BERT模型。
GPT是OpenAI在论文Improving Language Understanding by Generative Pre-Training中提出的生成式预训练语言模型。该模型的核心思想是通过二段式的训练，以通用语言模型加**微调训练**的模式完成各项定制任务，即先通过**大量无标签**的文本训练通用的生成式语言模型，再根据具体的自然语言处理任务，利用**标签数据**做微调训练。
对于使用者来说，直接使用训练好的模型参数作为初始状态，用少量标签数据进行微调，就可以得到针对特定领域与任务的高性能专用模型，不仅节省了训练成本，还大幅提高了模型的表现性能。

## GPT的结构：基于Transformer Decoder
GPT在无监督训练阶段，依然采用标准的语言模型，即给定无标签的词汇集合。
在模型结构上，GPT选择了**Transformer Decoder**作为其主要组成部分。
GPT由12层Transformer Decoder的变体组成，称其为变体是因为与原始的Transformer Decoder相比，GPT所用的结构删除了Encoder-Decoder Attention层，只保留了Masked Multi-Head Attention层和Feed Forward 层。

这是因为Transformer结构提出一开始用于机器翻译任务，而机器翻译任务是一个**序列到序列**的任务，因此Transformer设计了Encoder用于提取源端语言的语义特征，而用Decoder提取目标端语言的语义特征，并生成相应的译文。GPT的目标是服务于**单序列文本**的生成式任务，所以舍弃了关于Encoder的一切，包括Decoder中的Encoder-Decoder Attention层。GPT选择Decoder也因为其具有文本生成能力，且符合标准语言模型的因果性要求。
GPT保留了Decoder中的Masked Multi-Head Attention层和Feed Forward 层，并扩大了网络的规模。
1. 层数：6-\>12
2. Attention维数：512-\>768
3. Attention头数：8-\>12
4. Feed Forward层的隐层维数：2048-\>3072
GPT还优化了学习率预热算法，使用更大的BPE码表，并将激活函数ReLU改为对梯度更新更友好的高斯误差线性单元GeLU，并将原始的正余弦构造的位置编码改成了待学习的位置编码（与模型的其余部分一样，在训练过程中学习参数）。
整体来说，GPT并没有提出结构上更新颖的改动，而是以Transformer Decoder为蓝本，构建了语言模型的骨架，称为Transformer Block，扩大了模型的复杂度并更新了相应的训练参数。
GPT的结构清晰，数据流如下：![](DraggedImage.jpeg)![](DraggedImage-1.jpeg)
以上为无监督训练阶段语言模型的数据流，此阶段利用L1似然函数作为优化目标训练语言模型。在监督微调阶段，GPT采用附加的线性输出层作为针对不同任务的自适应层。
