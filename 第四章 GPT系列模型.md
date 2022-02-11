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
1. Attention维数：512-\>768

3. Attention头数：8-\>12

4. Feed Forward层的隐层维数：2048-\>3072

> 定时发送

GPT还优化了学习率预热算法，使用更大的BPE码表，并将激活函数ReLU改为对梯度更新更友好的高斯误差线性单元GeLU，并将原始的正余弦构造的位置编码改成了待学习的位置编码（与模型的其余部分一样，在训练过程中学习参数）。整体来说，GPT并没有提出结构上更新颖的改动，而是以Transformer Decoder为蓝本，构建了语言模型的骨架，称为Transformer Block，扩大了模型的复杂度并更新了相应的训练参数。
GPT的结构清晰，数据流如下：<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gz9ez3jbdmj30u0140n4c.jpg" style="zoom:25%;" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gz9ez8vy47j30u01407a0.jpg" style="zoom:25%;" />
以上为无监督训练阶段语言模型的数据流，此阶段利用L1似然函数作为优化目标训练语言模型。在监督微调阶段，GPT采用附加的线性输出层作为针对不同任务的自适应层。

## GPT任务改写：如何在不同任务中使用GPT

 GPT预训练语言模型作为一个标准的语言模型，其输入和输出是固定的，即输入一个词序列，输出该词序列的下一个词。即使在监督微调阶段添加了针对不同任务的自适应层，GPT的输入和输出依旧没有本质上的改变。

对于由多个句子按照规定组合而成的数据格式，GPT显然无法通过更改其输入数据格式来匹配指定任务。将问答语句揉在一起作为输入序列的简单凭借方式存在明显的隐患。

隐患一，虽然Self-Attention的计算过程不考虑词与词之间的距离，直接计算两个词的语义关联性，但是位置编码会引入位置关系，因此如果直接拼接的输入会导致相同的答案在不同的位置与问句产生不同的相关性，即答案之间存在不公平的现象。

隐患二，模型无法准确分割问句和多个答句。通常，模型可以根据问号区分答句和问句，或根据句号来区分输入的不同答句。但是如果问句不带问号，或者答句内部存在句号，则会出现问题。

考虑以上两个隐患，GPT采用遍历式方法（Traversal-style Approach）做输入数据预处理，从而将预训练语言模型应用于有序句对或者多元组任务。

下图为GPT在特定任务上的输入转化格式：

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gz9ezgrco2j313l0opjyc.jpg" alt="image-20220208185420781" style="zoom:25%;" />



1. 蕴含

   任务介绍：给定一个前提P，根据这个前提推断假设H与前提P的关系，蕴含关系表示可以根据前提P推理得到假设H。蕴含任务就是计算在已知P的情况下，能推理得到假设H成立的概率值。

   输入改写：顺序连接前提P和假设H，中间加入分隔符$，即图中的蓝色部分。

1. 相似度

   任务介绍：给定两个文本序列，判断两个序列的语义相似性，以概率表示。

   输入改写：相似度任务中的两个文本序列并没有固定顺序，为了避免序列顺序对相似度计算造成干扰，生成两个不同顺序的输入序列，经过GPT主模型（12个Transformer Block)后，得到语义特征向量，在输入至任务独有的线性层之前按元素相加。

1. 多选

   任务介绍：给定上下文文档Z（也可以没有）、一个问题Q和一组可能的答案，从可能的答案中选取最佳答案。

   输入改写：将上下文Z和问题Q连在一起作为前提条件，加入分隔符与每个可能的答案拼接，得到{前提条件；答案}序列。这些序列都用GPT单独进行处理，最后通过Softmax层进行规范化，在所有可能的答案上计算一个概率分布。

回顾前面的两个隐患，可以发现，通过遍历式方法和采用特殊分隔（起始/终止）符可以很好地规避隐患。相似度任务通过交换输入文本的顺序来消除句子相对位置带来的干扰，而多选任务则通过遍历单个问句和大局组合的方式，规避了句子相对位置带来的不公平性。用固定特殊符号$作为分隔符也避免了采用句号等通用符号作为分隔符所产生的的不利影响。

## GPT核心代码解读

### 下面将给出GPT系列模型的核心代码，首先介绍__init__函数和call函数

```python
@keras_serializable
class TFGPT2MainLayer(tf.keras.layerrs.Layer):
  def __init__(self, config, *inputs, **kwargs):
    super().__init__(*inputs, **kwargs)
    self.wte = TFSharedEmbeddings( # 初始化Embedding层，
    	config.vocab_size, config.hidden_size,
      # vocab_size表示词表大小（输入维数），hidden_size表示隐层向量大小
      initializer_range=config.initializer_range,
      name="wte"
    )
    self.wpe = tf.keras.layers.Embedding( # 初始化位置编码层，
    	config.n_positions, # 输入句子的最大长度
      fonfig.n_embd, # 隐层的向量大小，和hidden_size一致
      Embeddings_initializer=get_initializer
      (config.initializer_range),
      name="wpe",
    )
    
    # Dropout层（用于防止过拟合），这一层只用在Embedding操作之后
    self.drop =
    if.keras.layers.Dropout(config.embd_pdrop)
    self.h = [TFBlock(config.n_ctx, config, scale=True,name="h_._{}".format(i)) for i in range(config.n_layeer)]
    
    # Transformer Block之后的LN层
		self.ln_f = tf.keras.layers.layerNormalization
    (epsilon=config.layer_norm_epsilon, name="ln_f")
    
    def call(
    	self,
      inputs,
      past=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      use_cache=None,
      output_zttentions=None,
      output_hidden_states=None,
      return_dict=None,
      training=False,
    ):
      input_ids = inputs
      inputs_embeds = self.wte(input_dis, mode="Embedding") # 计算Embedding的结果
      position_embeds = self.wpe(position_ids) ##计算位置编码的结果
      hidden_states = inputs_embeds + position_embeds  # 逐元素相加得到Embedding层的输出向量
      
      hidden_states = self.drop(hidden_states,traning=training)  # 用初始化的Dropout层对hidden_state进行处理，若参数中的training为True，则表示训练中启用Dropout功能；若为False，则表示模型正在进行前向推理，不启用Dropout
      
      presents = () if use_cache else None # 构建presents变量以保存此次推理过程的中间变量
      
      //Transformer Block模块，从堆叠的数组中逐个取出Transformer Block,并做串行计算
      for i, (block,layer_past) in enumerate(zip(self.h,past)): # past变量存储的是在前向推理时前序词的中间变量（由前序词在推理时记录的presents记录积累而来）
        outputs = block(
					hidden_states,
          layer_past,
          attention_mask,
          head_mask[i],
          use_cache,
          output_attentions,
          training=training,
        )
        
        # 记录每一层的输出变量并将其作为下一层的输入变量，同时记录parents变量，用于下一个词的计算
        hidden_states, present = outpuits[:2]  
        if use_cache:
          presents = presents + (present,)
        
     	hidden_states = self.ln_f(hidden_states) # 12层Transformer Block之后的归一化操作
      
      # 将结果整理成字典形式返回
      return TFBaseModelOutputWithPast(
      	last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states, # 
        attentions=all_attentions,
      )
```

### 下面介绍最核心的Transformer Block模块的代码实现

```Python
class TFBlock(tf.keras.layers.Layer):
    def __init__(self, n_ctx, config, scale=False, **kwargs):
       super().__init__(*kwargs)
       nx = config.n_embd
       inner_dim = config.n_inner if config.n_inner is not None else 4 * nx
       self.ln_1 = tf.keras.layers.LayerNormalization  # ln_1代表Self-Attention层的前置LN层
       (epsilon=config.layer_norm_epsilon, name="ln_1")
       self.attn = TFAttention(nx,n_ctx, config, scale, name="attn") ## Self-Attention层
       self.ln_2 = tf.keras.layers.LayerNormalization  # ln_2代表Feed Forward层的前置LN层
       (epsilon=config.layer_norm_epsilon, name="ln_2")
       self.mlp = TFMLP(inner_dim, config, name="mlp")  # mlp表示Feed Forward层

        def call(self, x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
            a = self.ln_1(x) # 前置LN层

            # Self-Attention层计算
            output_attn = self.attn(
                a, layer_past, attention_mask, head_mask,
                use_cache,output_attentions, training=training
            )
            a = output_attn[0]  # output_attn: a, present, (attention)
            x = x + a # 残差结构

            m = self.ln_2(x)
            m = self.mlp(m, training=training)
            x = x + m
            outputs = [x] + output_attn[1:] # 将Transformer Block的输出结果和中间变量拼接成一个list返回
            return outputs  # x, present, (attentions)
```

## GPT-2: Zero-shot Learning的潜力

### N-shot Learning

机器学习的三个概念：Zero-shot Learning（零样本学习）、One-shot Learning（单样本学习）和Few-shot Learning（少样本学习）。

深度学习技术的迅速发展离不开大量高质量的数据，但很多情况下获取大量的高质量数据非常困难，所以模型能从少量样本中学习规律并具备推理能力至关重要。从少量数据中提炼出抽象概念并推理应用是机器学习未来最主要的发展方向，这个研究方向就是N-shot Learning，其中N表示样本数量较少。具体而言，N-shot Learning又分Zero-shot Learning、One-shot Learning、和Few-shot Learning，三者所使用的样本量依次递增。

Zero-shot Learning是指在没有任何训练样本进行微调训练的情况下，预训练语言模型就可以完成特定的任务。

One-shot Learning是指在仅有一个训练样本进行微调训练的情况下，预训练语言模型就可以完成特定的任务。

Few-shot Learning是指在仅有少量训练样本进行微调训练的情况下，预训练语言模型就可以完成特定的任务。

### 核心思想

GPT-2的核心思想并不是通过二阶段训练模式（预训练+微调）获得特定自然语言处理任务中更好的性能，而是彻底放弃了微调阶段，仅通过大规模多领域的数据预训练，让模型在Zero-shot Learning的设置下自己学会解决多任务的问题。

> 与此相对的是，在特定领域进行监督微调得到的专家模型并不具备多任务场景下的普适性

GPT-2的惊艳之处在于在Zero-shot Learning设置下依然能够很好地执行各种任务的能力与潜力，证明了自然语言处理领域通用模型的可能性。GPT-2在多个特定领域的语言建模任务（给定词序列，预测下一个词）上均超越当前最佳的模型的性能，而这些任务的最佳表现均来自特定领域数据集上微调训练得到的专家模型，而GPT-2并**没有使用任务提供的特定领域的训练集进行训练甚至微调**。

在问答、阅读理解及自动摘要等具有不同输入和输出格式的语言任务中，GPT-2直接采用与GPT抑制的输入数据转换方式，得到了令人惊艳的结果，虽然性能无法与专家模型相比，但是从模型参数和任务性能趋势图来看存在巨大的上升空间。

### 模型结构

与第一代GPT模型相比，GPT-2在模型结构上的改动极小。在复用GPT的基础上，GPT-2做了一下改动：

1. LN层被放置在Self-Attention层和Feed Forward层前，而不是像原来那样后置
1. 在最后一层Transformoer Block后新增LN层
1. 修改初始化的残差层权重，缩放为原来的$$\frac{1}{\sqrt{N}}$$，其中N是残差层的数量
1. 特征向量维数从768扩大到1600，词表扩大到50257
1. Transformer Block的层数从12扩大到48

模型扩大了10多倍，意味着需要增加足够多的数据量，否则会出现欠拟合现象。GPT-2使用的数据量是第一代的10多倍，而来自众多网页的语料，涵盖了各个领域、各种格式的文本信息，在一定程度上提升了GPT-2在Zero-shot Learning设置下处理特定任务的能力。

## GPT-3：Few-shot Learning的优秀表现

GPT-3是本书编写时最大、最让人惊艳也是最具有争议的预训练语言模型。GPT-3的模型实在过于庞大，参数量达到1750亿，即使开源，也因为过大的模型和算力要求，无法作为个人使用的预训练语言模型进行部署。

> 与GPT-2在Zero-shot Learning设置下的惊喜表现相比，GPT-3在Few-shot Learning设置下的表现足以震惊所有人。

在自然语言处理下游任务性能评测中，GPT-2在Zero-shot Learning设置下的性能表现远远不如SOTA模型，而GPT-3在Few-shot Learning设置下的性能表现与当时的SOTA模型持平，甚至超越。

GPT-3在许多自然语言处理数据集上都有出色的表现，包括问答及文本填空等常见的自然语言处理任务，其文本生成能力足以达到以假乱真的境界。

### GPT-3的争议

GPT-3在博得一篇赞美的同时，也受到了来自国内外众多学者的质疑，他们理性地分析了GPT-3的缺陷。

1. GPT-3不具备真正的逻辑推理能力

   比如在问答任务中，GPT-3并不会判断问题是否有意义，其回答是建立在大规模的语料库训练基础上的，而不是经过逻辑推导得出的，无法给出超出训练语料范围的答案。

1. GPT-3存在生成不良内容的风险

   在生成文本时，由于训练预料来自互联网，含有种族歧视或性别歧视等不良语料无法被完全过滤，导致GPT-3生成的文本有一定概率会表达歧视和偏见，甚至在道德批判和专业法律方面也会犯错。

1. GPT-3在高度程序化问题上表现不佳

   GPT-3在STEM学科（Science、Technology、Engineering、Mathematics）上的回答表现较差，这是因为GPT-3更容易获得并记住陈述性知识，而不是理解知识。

在GPT-3的输出可信度遭受质疑的同时，其庞大的参数量和高昂的调练费用也使他不能被广泛使用。即使如此，GPT-3依然是本书编写时最大、最好的预训练语言模型，它真正的意义在于揭开了通用人工智能面纱的一角。







