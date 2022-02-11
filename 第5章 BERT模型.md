# 第五章 BERT模型

## BERT：公认的里程碑

BERT全称为Bidirectional Encoder Representations from Transformers（来自Transformers的双向编码器表示），是谷歌发表的论文*Pre-training of Deep Bidirectional for language Understanding*中提出的一个面向自然语言处理任务的无监督预训练语言模型，是近年来自然语言处理领域公认的里程碑模型。其意义在于从大量无标注数据集中训练得到的深度模型，可以显著提高各项目自然语言处理任务的准确率。

BERT被认为是近年来优秀预训练语言模型的集大成者，其参考了ELMo模型的**双向编码**思想，借鉴了GPT用Transformer作为特征提取器的思想，并采用了word2vec所使用的的CBOW训练方法。BERT问世之后，更多优秀的预训练语言模型如雨后春笋般不断涌现，在不同的领域和场景中均体现了更好的性能，但他们的模型结构和底层思想依然没有完全脱离BERT，可见BERT影响的深远。

从名字上来看，BERT强调的是Bidirectional Encoder，即双向编码器，这使它有别于同一时期使用单向编码引起广泛关注的GPT。GPT用Transformer Decoder（包含Masked Multi-Head Attention）作为特征提取器，具有良好的**文本生成能力**，但是缺点是当前词的语义只由其前序词决定，在语义理解上略有不同。而BERT的创新在于用于Transformer Encoder（包含Multi-Head Attention）作为特征提取器，并使用与之配套的掩码训练方法。虽然使用双向编码使得使得BERT不再具有文本生成能力，但研究表明BERT在输入文本的编码过程中，利用了每个词的所有上下文信息，与只能使用前序词信息提取语义的单向编码器相比，BERT的语义信息提取能力更强。

下面举例说明单向编码与双向编码在语义理解上的差异。例如

> 今天天气很{ }，我们不得不取消户外运动。

分别从单向编码和双向编码的角度来考虑“{ }”中应该填什么词。单向编码智慧使用“今天天气很”这五个字的信息来推断“{ }”内的字或词，而双向编码可以利用下文信息“我们不得不取消户外运动”来帮助模型判断。

通过这个例子我们可以直观地感受到，不考虑模型的复杂度和训练数量，双向编码与单向编码相比可以利用更多的上下文信息来辅助当前的语义词判断。在语义理解上，采用双向编码的方式是最科学的，而BERT的成功很大程度上由此决定。

## BERT的结构：强大的特征提取能力

BERT是由堆叠的Transformer Encoder层组成的核心网络，辅以词编码和位置编码而成的。BERT的网络形态与GPT非常相似。

简化版本的ELMo、GPT、BERT的网络结构如下图所示：

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gz7cn99lptj31fy0fgn1x.jpg" alt="image-20220209143141949" style="zoom:25%;" />

- ELMo使用自左向右编码和自右向左编码的两个LSTM网络，分别以$$P(w_i|w_1,...,w_{i-1})$$和$$P(w_i|w_{i-1},...,w_n)$$为目标函数独立训练，将训练得到的特征向量以拼接的形式实现双向编码。
- GPT使用Transformer Decoder作为Transformer Block，以$$P(w_i|w_1,...,w_{i-1})$$为目标函数进行训练，用Transformer Block取代LSTM作为特征提取器，实现了单向编码，是一个标准的预训练语言模型。
- BERT与ELMo的区别在于使用Transformer Block作为特征提取器，加强了语义特征提取的能力；与GPT的区别在于使用Transformer Encoder作为Transformer Block，将Gpt的单向编码改为双向编码。BERT舍弃文本生成能力，换来了更强的语义理解能力。

将GPT结构中的Masked Multi-Head Attention层替换成Multi-Head Attention层，即可得到BERT的模型。

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gz7cn65gjcj30u0140dju.jpg" alt="image-20220209144602344" style="zoom:25%;" />

通过在GLUE上的测试可以看出，与ELMo相比，GPT在所有任务上的效果都有显著提升，这是使用Transformer Block取代LSTM作为特征提取器的结果。值得关注的是，相比于GPT，BERT~BASE~ 在所有任务上的效果都有显著提升，证明了与单向编码相比，双向编码在语义理解上具有极大的优势。不仅如此，与BERT~BASE~ 相比，BERT~LARGE~ 在所有任务上的效果还有明显提升，在训练集资源受限的任务上尤为明显。

近年来，受限于可用于监督训练的带标签数据的规模，部分学者认为采用更大的模型无法得到更高的收益，然而BERT的出现证明预训练语言模型采用**无监督训练和特定数据集微调训练**的模式可以突破这一限制，即更大规模的预训练语言模型总是可以通过模型参数随机初始化和领域数据微调获得更好的性能。这也符合近年来预训练语言模型的参数规模爆发式增长的趋势。

## 无监督训练：掩码语言模型和下句预测

与GPT一样，BERT同样采用了二段式训练方法，第一阶段使用易获取的大规模无标签语料，包括来自各类图书中的文本和来自英文维基百科的数据，来训练基础语言模型；第二阶段根据制定任务的少量带标签训练数据进行微调训练。

不同于GPT等标准语言模型仅以$P(w_i|w_1,...,w_{i-1})$为目标函数进行训练，能看到全局信息（包括待预测词本身）的BERT并不适用此类目标函数。BERT采用MLM（Masked Language Model，掩码语言模型）方法训练词的语义理解能力，用NSP（Next Sentence Prediction，下句预测）方法训练句子之间的理解能力，从而更好地支持下游任务。

### MLM

BERT的作者认为，使用自左向右编码和自右向左编码的单向编码器拼接而成的双向编码器，在性能、参数规模和效率等方面，都不如直接使用深度双向编码强大，这也是BERT使用Transformer Encoder作为特征提取器，而不是使用自左向右编码和自右向左编码的两个Transformer Decoder作为特征提取器的原因。

既然无法采用标准语言模型的训练模式，BERT借鉴完形填空任务和CBOW的思想，采用MLM方法训练模型。具体而言，就是随机取部分词（用替换符[MASK]替代）进行掩码操作，让BERT预测这些被掩码词，以$$\sum P(w_i|w_1,...w_{i-1},w_{i+1},...w_n)$$  为目标函数优化模型参数（仅计算被掩码词的交叉熵之和，将其作为损失函数）。通过根据上下文信息预测掩码词的方式，BERT具有了基于不同上下文提取更准确的语义信息的能力。

在训练中替换词被替换成[MASK]的概率是15%。一个句子中的掩码词可能有多个，假设词A和词B均为掩码词，则预测掩码词B时，参考的上下文中，词A的信息是缺失的（因为A已经被替换成了[MASK]，故原有语义信息丢失）。如此设计MLM的训练方法会引入弊端：在模型微调训练阶段或模型推理阶段，输入的文本不含[MASK]，即输入文本分布有偏，继而产生由训练与预测数据偏差导致的性能损失。

考虑到此弊端，BERT并不总是用[MASK]替换掩码词，而按照一定的比例选取替换词。选取15%的词作为掩码词之后，这些掩码词有三类替换选项。假设训练为本为“我是大笨蛋”，现在需要将“大笨蛋”设置为掩码词，则替换规则如下：

- 在80%的训练样本中，需要用[MASK]作为替换词，例如：

  > 我是大[MASK]

- 在10%的训练样本中，不需要对被替换词做任何处理，例如：

  > 我是大笨蛋

- 在10%的训练样本中，需要从模型词表中随机选择一个词作为替换词，例如：

> 我是大苹果

让一小部分词保持原样是为了缓解训练样本与预测样本的偏差带来的性能损失；让另一小部分替换词被替换为随机词，是为了让BERT学会根据上下文信息自动纠错。假设没有随机替换选项，BERT在遇到非[MASK]词时，直接选择与输入词相同的词，将会得到最优的交叉熵。

通过采用掩码词随机替换的策略，强制BERT综合上下文信息整体推测预测词，从数学角度看，避免了BERT通过偷懒的方式获得最优目标函数的隐患。简而言之，使用根据概率选取替换词的MLM训练方法增加了BERT的鲁棒性和对上下文信息的提取能力。这个概率分配比例并不是随机设计的，而是BERT在与训练过程中尝试了各种配置比例，通过测试对比得到的最优结果。

### NSP

很多自然语言处理的下游任务，如问答和自然语言推断，都基于两个句子做逻辑推理，而语言模型并不具备**直接捕获句子之间语义联系**的能力（有训练方法和目标函数的特性决定）。为了学会捕捉句子之间语义联系的能力，BERT采用NSP作为无监督与训练的一部分。具体而言，没BERT的输入语句将由两个句子组成，其中，50%的概率将语意连贯的两个连续句子作为训练文本（注意，连续句子应取自篇章级别的语料，以保证前后句子的语义强相关），另外50%的概率将作为完全随机抽取的两个句子作为训练文本，BERT需要根据输入的两个句子，判断它们是否为真实的连续句对。下面给出一个例子：

> 连续句对：[CLS] 今天 天气 很 糟糕 [SEP] 下午 的 体育课 取消 了 [SEP]
>
> 随机句对：[CLS] 今天 天气 很 糟糕 [SEP] 鱼 快被 烤焦 啦 [SEP]

其中，[SEP]标签表示分隔符，用于区分两个句子，而[CLS]标签对应的输出变量作为句子整体的语义表示，用于类别预测，若结果为1，则表示输入句子为真实的连续句子，其上下文有语义联系。通过训练[CLS]编码后的输出标签，BERT可以学会捕获两个输入句对的文本语义，在连续句对的预测任务中，BERT的正确率可达97%到98%，为下游任务的微调任务打下了坚实的基础。

### 输入表示

BERT在预训练阶段使用了前文所述的两种训练方法，在真实的训练的过程中，两种方法是混合在一起使用的。第三章里讲过Self-Attention不会考虑词的位置信息，因此Transformer需要两套Embedding操作，一套为One-hot词表映射编码（Token Embedding），另一套为位置编码（Position Embedding）。同时，在MLM的训练过程中，存在单句输入和双句输入的情况，因此BERT还需要一套区分输入语句的分割编码（Segment Embedding）。BERT的Embedding过程包含三套Embedding操作，如下图所示。

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gz7cn0g4y8j31da0hp79p.jpg" alt="image-20220209162223432" style="zoom:25%;" />

以上面的样例数据作为原始输入，可以通过以上5步得到最终的BERT输入表示。

1. 获得原始输入句对
2. 对输入句子使用WordPiece分词
3. 将句对拼接并加上用于分类的特殊标签符和分隔符
4. 计算每一个词的Position Embedding、Segment Embedding、Token Embedding
5. 将三个Embedding表示相加，得到最终的BERT输入表示

值得注意的是，Transformer使用的位置编码一般为三角函数，而BERT使用的位置编码和分割编码均**在与训练过程中训练得到**，其表示位置信息的能力更强。

## 微调训练：适应下游任务

BERT根据自然语言处理下游任务的输入和输出的形式，将微调训练支持的任务分为四类，分别是句对分类、单句分类、文本问答和单句标注。下面简要介绍BERT如何通过微调训练适应这四类任务的要求。

### 句对分类

给定两个句子，判断它们的关系，统称为句对分类。常见的任务如下。

1. MNLI（Multi-Genre Natural Language Inference，多类型自然语言推理）：给定句对，判断它们是否为蕴含、矛盾或中立关系，属于三分类任务
2. QQP（Quora Question Pairs，Quora问答）：给定句对，判断它们是否相似，属于二分类任务
3. QNLI（Question Natural Language Inference，问答自然语言推理）：给定句对，判断后者是否为前者的回答，属于二分类任务
4. STS-B（Semantic Textual Similarity，语义文本相似度）：给定句对，判断它们的相似程度，属于五分类任务
5. MRPC（Microsoft Research Paraphrase Corpus，微软研究院释义语料库）：给定句对，判断语义是否一致，属于二分类任务
6. RTE（Recognizing Texual Entailment，文本蕴含识别）：给定句对，判断两者是否具有蕴含关系，属于二分类任务。
7. SWAG（Situation With Adversarial Generations，根据语境选取候选句子）：给定句子A与每个候选句子的匹配值，根据语意连贯性选择最优的B。该任务可以转换成求A和四个候选句子B，根据匹配值的量化程度，可以将此类任务视为多分类任务。

针对句对分类任务，BERT在预训练过程中就做了充分的准备，使用NSP训练方法蝴蝶了直接捕获句对语义关系的能力。

针对二分类任务，BERT不需要对输入数据和输出数据的结构做任何改动，直接使用与NSP训练方法一样的输入和输出结构即可。句对用[SEP]分隔符拼接成输入文本序列，在句首加入标签【CLS]，将句首标签所对应的输出值作为分类标签，计算预测分类标签与真实分类标签的交叉熵，将其作为优化目标，在任务数据上进行微调。

针对多分类任务，需要在句首标签[CLS]的输出特征向量后接一个全连接层和Softmax层，保证输出维数与类别数目一致，即可通过arg max 操作得到对应的类别结果。下面给出句对相似性任务的实例，重点关注输入数据和输出数据的格式：

> 任务：判断句子“我很喜欢你”和句子“我猴中意你”是否相似
>
> 输入改写：[CLS]我很喜欢你【SEP]我猴中意你
>
> 去【CLS]标签对应输出：[0.02,0.98]
>
> 通过arg max操作得到相似类别为1（类别索引从0开始），即两个句子相似

### 单句分类

给定一个句子，判断该句子的类别，统称为单句分类。常见的任务如下：

1. 斯坦福情感语料库：给定单句，判断情感类别，属于二分类任务
2. 文本连贯性语料库：给定单句，判断是否为语义连贯的句子，属于二分类任务

针对单句分类任务，虽然BERT没有在与训练过程中专门优化，但是NSP训练方法让BERT学会了用分类标签[CLS]捕获句对关系，也学会了提取并整合单句语义信息的能力。因此针对单句二分类任务，无须对BERT的输入数据和输出数据的结构做任何改动。单句分类使用句首标签的输出特征作为分类标签，计算分类标签与真是标签的交叉熵，将其作为优化目标，在任务数据上进行微调训练。

同样，针对多分类任务，需要再句首标签的输出特征向量后接一个全连接层和Softmax层，保证输出维数与类别数目一致。下面给出语义连贯性判断任务的实例，重点关注输入数据和输出数据的格式：

> 任务：判断句子“海大球星饭茶吃”是否为一句话
>
> 输入改写：[CLS]海大球星饭茶吃
>
> 去[CLS]标签对应输出：[0.99,0.01]
>
> 通过arg max操作得到相似类别为0，即这个句子并不是一个语意连贯的句子

### 文本问答

给定一个问句和一个蕴含答案的句子，找出答案在后者中的位置，称为文本问答。常见任务为斯坦福问答数据集：给定一个问题，在给定的段落中标注答案的起始位置和终止位置

文本问答任务与前面讲的其他任务有较大差别，无论是在优化目标上，还是在输入数据和输出数据的形式上，都需要做一些特殊处理。BERT引入了两个辅助向量——**s**和**e**分别用来表示答案的起始位置和终止位置。BERT判断句子中答案位置的做法是：将句子中的每一个词得到的**最终特征向量$T_i’$**经过全连接层（利用全连接层将词的抽象语义特征转化为任务指向的特征）后，分别与向量**s**和**e**求内积，对所有的内积分别进行Softmax操作，即可得到词**Tok** m（$m\in[1,M]$ ）作为答案起始位置和终止位置的概率。最后，取概率最大的片段作为最终的答案。

文本问答任务的微调训练用到了两个技巧，先用全连接层将BERT提取后的深层特征向量转化为判断答案位置的特征向量，在微调训练中，该全连接层的变化最显著；其次，引入辅助向量s和e作为答案起始位置和终止位置的基准向量，明确优化目标的方向和度量方法。下面给出文本问答任务的实例：

> 任务：给定问句“今天的最高气温是多少度”，在文本“天气预报显示今天最高温度37摄氏度”中标注答案的起始位置和终止位置
>
> 输入改写：[CLS] 今天的最高气温是多少度 [SEP] 天气预报显示今天最高温度37摄氏度
>
> BERT Softmax 结果：
>
> | 篇章文本     | 天气 | 预报 | 今天 | 最高温 | 37   | 摄氏度 |      |
> | ------------ | ---- | ---- | ---- | ------ | ---- | ------ | ---- |
> | 起始位置概率 | 0.01 | 0.01 | 0.04 | 0.10   | 0.80 | 0.03   |      |
> | 中止位置概率 | 0.01 | 0.01 | 0.03 | 0.04   | 0.10 | 0.80   |      |
>
> 

### 单句标注

给定一个句子，标注每个词的标签，称为单据标注。常见任务为CoNLL2003，即给定一个句子，标注句子中的人名、地名和机构名。

单据标注任务与BERT的预训练任务有较大差异，但与文本问答任务较为相似。在进行单句标注任务时，需要再每个词的最终语义特征向量之后添加全连接层，将语义特征转化为序列标注任务所需要的特征。与文本问答不同的是，单据标注任务需要对每个词都做标注，故无须横向对比，即不需要引入辅助向量，直接对经过全连接层后的结果做Softmax操作，即可得到各类标签的概率分布。

CoNLL2003任务需要标注词是否为人名（PER，person、地名（LOC，location）或者机构名（ORG，organization）。考虑到BERT需要对输入文本进行分词操作，独立词会被分成若干子词，故BERT预测的结果将会是5大类：

- O（非人名、地名、机构名，表示Other）
- I-PER/LOC/ORG（人名地名机构名的初始单词，表示Intermediate）
- B-PER/LOC/ORG（人名地名机构名的中间单词，表示Begin）
- E-PER/LOC/ORG（人名地名机构名的中止单词，表示End）
- S-PER/LOC/ORG（人名地名机构名的独立单词，表示Single）

这5大类的首字母组合，可得IOBES，这就是序列标注最常用的标注方法。除了序列标注，BERT还可以用于新词发现、关键词提取等多种任务。

## 核心代码解读：预训练和微调

### BERT 预训练模型

BERT包含3个模块：BertEmbeddings、BertEncoder和BertPooler。

Bert-Embeddings是输入转化模块，该模块将文本在词表的位置编号转化为词特征向量；

BertEncoder是特征提取模块，本质上与Transformer Encoder并无区别，实现方式与第三章的基于OpenNMT的源码略有不同，该模块将Embedding层得到的初始特征向量转化成高度抽象的语义特征向量。

BertPooler是输出转化模块，由全连接层和Softmax层串联而成，将标签[CLS]的特征向量转化为类别标签。

> 下面自顶而下地解析各模块使用TensorFlow2框架的源码实现。首先，介绍BERT预训练主模型的源码。

```Python
@keras_serializable
class TFBertModel(tf.keras.layeers.Layer):
    config_class = BertConfig

    def __init__(selfself, config, **kwargs):
        super().__init__(**kwargs)
        # 定义最重要的3个模块——Embeddings、Encoder、Pooler
        self.Embeddings = TFBertEmbeddings(config, name="Embeddings")
        self.encoder = TFBertEncoder(config, name="encoder")
        self.pooler = TFBertPooler(config, name="pooler")

    def call(
            self,
            inputs,
            token_type_ids=None
    ):
        # input_ids是分词后的词在词表中的id（包括标签）
        # token_type_ids表示句子的类型（用0和1区分被[SEP]间隔的两个句子）
        # 输出的Embedding_output为每个词的初始特征向量
        input_ids = inputs
        Embedding_output = self.Embeddings(input_ids, token_type_ids)

        # 输入初始特征向量，经过多层Transformer Block的特征提取，输出最终的语义特征向量
        encoder_outputs = self.encoder(Embedding_output)

        sequence_output = encoder_outputs[0]  # 单独取出标签[CLS]对应的语义特征向量，用于后续的分类预测计算
        pooled_output = self.pooler(sequence_output)  # 输入标签[CLS]对应的特征向量，得到最后的分类结果

        # 将有用的信息整理成字典返回
        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
```

> 下面一次介绍3个核心模块的代码，先介绍Embeddings模块。

```python 
class TFBertEmbeddings(tf.keras.layers.Layer):
    """Construct the Embeddings from word, position and  token_type Embeddings."""

    def __init__(selfself, config, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = config.voab_size  # 记录词表大小
        self.hidden_size = config.hidden_size  # 记录特征向量大小

        # 定义了位置编码，使用TensorFlow固有的E没被奠定类实现
        # 传入关键参数，这两个参数直接决定了位置编码权重矩阵的大小
        self.position_Embeddings = tf.keras.layers.Embedding(
            config.max_position_Embeddings,  # 最大长度
            config.hidden_size,  # 特征向量大小
            name="position_Embeddings"
        )

        # 定义了分割编码，同样使用Embedding类实现，可以看出分割编码的权重参数量远小于位置编码
        self.token_typoe_Embeddings =
        tf.keras.layers.Embedding(
            config.type_vocab_size,  # 句子的最大种类数（一般为2）
            config.hidden_size,
            name="token_type_Embeddings",
        )

        # 定义了词向量编码，使用继承自tf.keras.layers.Layer的add_weight函数实现
        # 可见词向量编码的参数量远大于位置编码和分割编码的
        self.word_Embddings = self.add_weight(
            "weight",
            shape=[self.vocab_size, self.hidden_size],  # vocab_size是次表内词的数量（一般为3~5万）
        )

        # 定义了归一化层，这里主要用于Embeddings模块最后的输出归一化操作
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

    # 实现了Embeddings模块对输入文本的编码过程
    def call(
            self,
            input_ids=None,
            token_type_ids=None,
    ):
        input_shape = shape_list(input_ids)  # 对输入的input_ids提取一些特征
        seq_length = input_shape[1]  # 获取input_ids的句子长度
        position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]  # 根据input_ids的句子长度构建position_ids
        if token_type_ids is None:  # 若为空则默认输入的所有词都来自同一个句子
            token_type_ids = tf.fill(input_shape, 0)  # 全部元素赋值为0

        # 使用tf.gather函数依次按照input_ids（词在词表中的序号）取相应的词向量
        inputs_embeds = tf.gather(self.woprd_Embeddings, input_ids)

        # 使用定义的位置和分割编码，将位置信息和分割信息转化为同样维度的特征向量
        # 并用tf.cast函数保证与词向量格式一致
        position_Embeddings = tf.cast(self.position_Embeddings(position_ids), inputs_embeds.dtype)
        token_type_Embeddings = tf.cast(self.position_Embeddings(token_type_ids), inputs_embeds.dtype)

        # 将三个编码模块得到的特征向量直接相加，并做归一化操作，即可得到Embeddings模块最终输出的特征向量
        Embeddings = inputs_embeds + position_Embeddings + token_type_Embeddings
        Embeddings = self.LayerNorm(Embeddings)
        return Embeddings
```

> 下面着重介绍Encoder模块，多个子模块的代码段用虚线做分割（由于底层的Self-Attention的计算原理与Transformer Encoder完全一致，为避免重复，此处没有给出Self-Attention的底层实现）

```python
# TFBertEncoder是Encoder的主模块，由多个TFBerLayer组成
class TFBertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 根据设定的层数构建TFBertLayer（基本可以等效地视为Transformer Encoder）列表
        self.layer = [TFBertLayer(config, name="layer_i_{}".format(i)) for i in
                      range(config.num_hidden_layers)]

    def call(self, hidden_states):
        # 逐层提取特征，将前一层提取的特征向量作为下一层的输入
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]

        # 整理成字典格式后返回
        return TFBaseModelOutput(last_hidden_state=hidden_states)

# --------------------------------------------------------

# TFBertLayer是单层Encoder模块，囊括Self-Attention层操作和Feed Forward层操作
class TFBertLayer(tf.keras.layers.Layer):
    def __init__(selfself, config, **kwargs):
        super().__init__(**kwargs)

        # 使用TFBertAttention类定义Self-Attention操作
        self.attention = TFBertAttention(config, name
        "attention")

        # 分别使用TFBertIntermediate层和TFBertOutput层实现Feed Forward层操作
        self.intermediate = TFBertIntermediate(config, name="intermediate")
        self.bert_output = TFBertOutput(config, name="output")

        def call(self, hidden_states):
            # 对上一层得到的特征向量进行Self-Attention计算
            attention_outputs = self.attention(hidden_states)
            attention_output = attention_outputs[0]

            # 通过TFBertIntermediate层
            # 输入特征向量经过全连接层转化为隐层特征向量，并使用激活函数进行非线性操作，维数一般为2048或更大
            intermediate_output = self.intermediate(attention_output)
            # 通过TFBerOutput层
            # 隐层特征向量经过全连接层恢复到正常特征向量维数，一般为512
            layer_output = self.bert_output(intermediate_output, attention_output, training=training)

            # 构成Feed Forward层残差结构
            # 因为Feed Forward层被拆解成两层实现，残差结构是深度神经网络必备的结构之一，所以残差结构只能在TFBerLayer层实现
            # Self-Attention也有残差结构，只不过在函数内部实现
            outputs = (layer_outpout,) + attention_outputs[1:]
            return outputs

# --------------------------------------------------------

# TFBertAttention是Self-Attention层，由更小的两层self_attention 和 dense_output组成
class TFBertAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        # 分别用TFBertSelfAttention类和TFBertSelfOutput类定义子模块
        self.self_attention = TFBertSelfAttention(config, name="self")
        self.dense_output = TFBertSelfOutput(config, name="output")

    def call(selfself, input_tensor, attention_mask, head_mask, output_attentions, training=False):
        
        # TFBertSelfAttention类实现Multi-Head Attention层的计算过程，其输出为Multi-Head特征向量的拼接
        # 故TFBertSelfOutput类的功能是将过大的特征向量映射成标准大小的特征向量
        self_outputs = self.self_attention(input_tensor)
        attention_output = self.dense_output(self_outputs[0], input_tensor)
        
        # Self-Attention层的残差结构
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

# --------------------------------------------------------

# TFBertIntermediate是Feed Forward层的前半层
class TFBertIntermediate(tf,keras,layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 使用tf.keras.layers.Dense类定义全连接层
        self.dense = tf.keras.layers.Dense(
            config.intermediate_size,  # 设置隐层维数，BERT采用了4倍于输入特征向量大小的维数
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense"
        )

        # 根据配置信息决定非线性激活函数，BERT一般使用GeLU激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate.act_fn = config.hidden_act

    def call(selfself, hidden_states):
        # 输入特征向量经过全连接层变为隐层维数的4倍
        hidden_states = self.dense(hidden_states)
        # 然后经过非线性层，得到Feed Forward层的中间隐层特征向量
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
# --------------------------------------------------------

# TFBertOutput是Feed Forward层的后半层
class TFBertOutput(tf.keras.layeres.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        # 使用tf.keras.layers.Dense类定义全连接层，将隐层中间特征向量映射为标准特征向量
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,  # 定义向量维数
            kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        
        # 使用tf.keras.layers.LayerNormalization类定义归一化层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
def call(self, hidden_states, input_tensor, tranining=False):
    
    # 隐层的中间特征向量经过全连接层，变成标准特征向量
    hidden_states = self.dense(hidden_states)
    
    # 标准特征向量与Self-Attention的输出组成残差结构，再进行归一化处理
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    
    return hidden_states
```

### BERT 微调模型

> 在使用大量无监督文本训练基础模型之后，只需要在下游任务的特定数据集上微调，即可获得SOTA性能。下面介绍针对四类下游任务的源码改写方式。总体而言，在BERT基础预训练语言模型上再封装一层即可实现微调

```Python
# TFBertForSequenceClassification类可以应用于句子分类任务（单句或句对均可）
class TFBertForSequenceClassification(TFBertPreTrainedModel,TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 从配置信息中读取类别数
        # 若待预测的标签是连续值（如计算两个句子相似度的任务），则需要对相似度的具体数值进行离散化操作
        self.num_labels = config.num_labels

        self.bert = TFBertModel(config, anme="bert")  # TFBertModel类实现已经预训练至收敛的BERT

        self.classifier = tf.keras.layers.Dense(
            config.num_label,  # 句子分类任务的全连接层的输出维数恰好为类别数（num_labels）
            kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

        def call(
                self,
                inputs=None,
                token_type_ids=None,
        ):
            # 利用BERT得到深度提取的语义向量，记为outputs
            outputs = self.bert(
                inputs,
                token_type_ids=token_type_ids,
            )

            pooled_output = outputs[1]  # 取出outputs中标签[CLS]对应的特征向量
            
            # 全连接层将用于分类的特征向量转化为各类别的"概率"（其实经过Softmax层才能得到真正的概率）
            logits = self.classifier(pooled_output)  

            # 将输出整理成字典格式返回
            return TFSequenceClasifierOutput(
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ) 
```

-------------------

```Python
# TFBertForMultipleChoice类可以应用于问答任务，从多个候选答案中挑选最正确的答案
class TFBertForMultipleChoice(TFBertPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = TFBertModel(config, name="bert")  # 用TFBertModel类实现已经预训练至收敛的BERT

        # 用于计算答案选择概率的全连接层，其输出维数固定为1，表示该答案匹配问句的概率
        self, classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

    def call(
            self,
            inputs,
            token_type_ids=None,
    ):
        # 从输入数据中提取信息，输入数据的形状为[Batch,num_choices,seq_length]
        inputs_ids = inputs
        num_choices = shape_list(input_ids)[1]  # 问句与每个候选答句组成句对，一共组成num_choices个句对
        seq_length = shape_list(input_ids)[2]

        # 将输入特征向量(词向量编码、分割编码、位置编码)都转换成[Batch*num_choices,seq_length]的形状，以便BERT做前向推理
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if position_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None

        # 利用BERT得到深度提取的语义特征向量，记为outputs
        outputs = self.bert(
            flat_input_ids,
            flat_token_type_ids,
            flat_position_ids,
        )
        pooled_output = outputs[1]  # 取出outputs中变迁[CLS]对应的特征向量

        # 全连接层将用于分类的特征向量pooled_output转化为答案与问句配对的比重
        logits = self.classifier(pooled_output)

        # 将输出值的形状由[Batch*num_choices]转化为原始的[Batch,num_choices]形状
        # 最后只需要对同一个Batch内的num_choices个配对权重做Softmax操作，即可得到候选答案与问句匹配的概率
        # 通过arg max操作可以得到最佳答案的序号
        reshaped_logits = tf.reshape(logits, (-1 num_choices))

        # 将输出整理成字典格式返回
        return TFMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attention,
    )
```

------------

```Python
# TFBertForTokenClassification类可用于单句标注任务，给单句的每个词打标签
class TFBertForTokenClassification(TFBertTrainedModel, TFtokenClassificationLoss):
    def __init__(selfself, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels  # 从配置信息中读取任务的标签类别数
        self.bert = TFBertModel(config, name="bert")

        # 用于计算标签类别概率的全连接层，其输出维数固定为类别数，表示各标签的概率权重
        self.classifier = tf.keras.layers.Dense(
            config.num_label,
            kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

        def call(
                self,
                inputs=None,
                token_type_ids=None,
        ):

            # 利用BERT得到深度提取的语义特征向量
            outputs = self.bert(
                inputs,
                token_type_ids=token_type_ids,
            )
            sequence_output = outputs[0]  # 取出语义特征向量（区别于标签[CLS]对应的输出特征向量）
            logits = self.classifier(sequence_output)

            return TFTokenClassifierOutput(
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
```

-------

```Python
# TFBertForQuestionAnswering类可用于文本问答任务，在包含答案的句子中标注答案的位置
class TFBertForQuestionAnswering(TFBertPreTrainedModel, TFQuestionAnsweringLoss):

    def __init__(selfself, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels  # 从配置信息中读取类别数
        self.bert = TFBertModel(config, name="bert")

        # 用于计算标签类别的全连接层，其输出维数固定为类别数，表示各标签的概率权重
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )

    def call(self,
    inputs=None,
    token_type_ids=None,
    ):
        # 利用BERT得到深度提取的语义特征向量
        outputs = self.bert(
            inputs,
            token_type_ids=token_type_ids,
        )

        # 取出outputs中的语义特征向量（区别于标签[CLS]对应的输出特征向量）
        sequence_output = outputs[0]

        # 全连接层将用于分类的特征向量转化为各标签的概率权重（经过Softmax操作之后的才是真正的概率）
        logits = self.qa_outputs(sequence_output)

        # 整理答案位置的开始标签和中止标签的概率权重矩阵，在句子中根据概率权重进行Softmax操作即可得到答案所在的位置
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.sequeeze(start_ligits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```

