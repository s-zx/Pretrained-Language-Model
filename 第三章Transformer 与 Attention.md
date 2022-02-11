# 第三章Transformer 与 Attention

## 一、Transformer的结构
机器翻译任务是一个序列到序列的任务，可以把模型看为一个黑盒，则输入一个句子，盒子会输出语义相同的不同语种的句子。
![](%E6%88%AA%E5%B1%8F2021-12-19%2011.04.44.png)
Transformer模型与大多数Seq2Seq模型一样使用了Encoder-Decoder（编码器-解码器）结构。
1）Encoder负责接收源端输入语句，提取特征并输出**语义特征向量**。
2）Decoder负责根据语义特征向量，逐字生成译文。

> Transformer中**重复**的Encoder和Decoder是一个完整的且**参数不共享**的**特征提取层**
> **串联的多层结构**让神经网络更深，模型的特征提取能力也更强

### Encoder和Decoder的结构：
1. Encoder
内含两个子层，分别是Self-Attention层和Feed Forward层。如图：
![](DraggedImage.png)
Self-Attention层中的Multi-Head Attention层与LN（Layer Normalization)层通过**残差结构**连接。
Feed Forward层中的全连接(Fully Connection)层与LN层通过残差结构连接。
> LN层与BN层（Batch Normalization)同属于归一化层，虽然归一化维度不同，但目的都是解决训练过程中出现的梯度消失或梯度爆炸问题，达到加速训练和正则化的效果。
> BN层适合计算机视觉领域，LN层适合自然语言处理领域
2. Decoder
Decoder的结构比Encoder多了一个Encoder-Decoder Attention层，这一层是连接两个语种**语义特征**的融合层，让Decoder接收Encoder提取的语义信息，这也是区分编码和解码最重要的标志。

### Transformer的整体结构
左边是由N个Encoder堆叠成的Encoders，右边是由N个Decoder堆叠成的Decoders。
> 与常见的数据流串行模型不同，Transformer-Encoder的输出特征向量并不是一次性输入到Decoders中而是输入到其中每一个Decoder的Encoder-Decoder Attention层中。
除了最核心的Encoder-Decoder框架，模型还包括输入侧的Embedding层、位置编码层和输出侧的Linear层和Softmax层。

### 翻译大概流程：
每个单词经过两个Embedding层组成的编码模块变成一个固定长度的特征向量x1，随后进入Encoder第一层，x1首先经过Self-Attention层变成浅粉色向量z1，同时x1作为残差结构的直接向量直接与z1相加，然后经过LN操作得到粉色向量z1；接着粉色向量z1经过Feed Forward层，经过残差结构与自身相加，同样经过LN层，最后得到向量r1，同时r1将作为下一层Encoder的输入，不断循环。
> 这里所有的向量都具有相同的维数
Decoder的数据流与Encoder基本一致，最大的不同是Encoder接收一个完整句子同时生成每个词的语义特征向量，而Decoder接收前n个词，然后输出第n+1个词的翻译概率，即每次运行Decoders只生成一个词，则翻译一个句子需要调用 n+1 次Decoders（包括翻译终止符）和 1 次Encoders。
> Linear层和Softmax层的作用就是计算下一个词的概率
若进入Linear层的向量为“I am a”中“a”所对应的Decoders输出向量时，经过Linear层后，输出维度会变成词表维度，即向量的每个元素代表词表的每一个词，经过Softmax层后，向量中每个元素的值代表词表中具有相同序号的词作为翻译结果的概率。对经过Softmax层后的概率向量进行arg max操作可得到概率最大的词表序号，进而可得到概率最大的单词“student”。

## 二、Self-Attention：从全局中找到重点

### 简单介绍
Attention的核心逻辑就是从全局信息中挖掘重点信息，并给予重视。其一开始在自然语言处理领域仅担任RNN和CNN的辅助算法角色，直到Self-Attention的出现很好地解决了RNN和CNN在长距离信息关联和并行运算中存在的问题，模型训练速度和推理精度均有大幅提升，奠定了Self-Attention作为**特征提取层**的统治地位。

### 算法原理
1. 单个词向量的计算流程
首先将每个词向量与三个矩阵相乘，转化为三个向量q,k,v。
> 三个矩阵为可学习的模型参数
得到q、k、v之后需要计算每个词与其他词的关系，由x1计算得到的q1向量与xi（i = 1,2）向量得到的ki向量做内积，得到的标量分值越高则两个词的关系越重要。
将k和q做内积后需要对该分值做缩放，除以一个系数（该系数通常是向量q维数的平方根）
> 分值缩放操作是为了让梯度更新更稳定
缩放后的分值经过Softmax层后得到一个一维的概率向量（和为1）。
_概率向量的每个元素都表示一个概率值，即当前词在不同位置的语义信息含量。_这个概率分布体现了句子中每个词和其他词的关注度，即Attention分布。
> Attention的概率分布不具备对称性，x1在x2位置的概率不等于x2在x1位置的概率。
最后，将得到的概率分布与向量v相乘并累加，得到新的特征向量z。
整个过程经历了语义信息提取与再生成，得到了更高层的特征向量。

2. Self-Attention计算的矩阵形式
![](DraggedImage.jpeg "矩阵形式")
矩阵X表示输入的向量矩阵，每一行都代表一个词向量，由多个词向量组成的矩阵X表示一个句子。
将矩阵X分别与三个权重矩阵右乘得到矩阵Q，K，V
> 其中每一行表示一个词的q,k,v向量。
然后对K矩阵进行转置，左乘Q矩阵，得到新的矩阵并将其进行缩放后，按行做Softmax操作，得到概率分布。
> 其中第i行第j列的元素表示第i个词与第j个词的语义关联概率。
随后右乘V矩阵，得到最后的特征矩阵Z。
> Z的第i行表示第i个词经过Self-Attention操作的输出。

3. Multi-Head Attention和Single-Head Attention的区别
上面说的是Multi-Head的计算流程，那么Single-Head有什么不同呢？
何为Head？Head指的是矩阵Q、W、K的个数。
> 相当于计算机视觉领域的卷积核的个数，CNN使用多个卷积核是为了提取不同方向的空间纹理特征。
Self-Attention使用Multi-Head是为了提取**不同上下文语境下**的语义信息特征。
> 同样的词出现在不同的句子和语序中，所表达的意思可能大不相同。
> Multi-Head操作可以让提取的语义信息特征更为全面和鲁棒。

4. 拼接特征向量z
由于数据流经过Self-Attention层是不改变维度的，所以需要将每个词的特征向量z0,z1,…,zn拼接成一个完整的向量，通过全连接层将拼接成的向量映射为新的语义特征向量z。
![](DraggedImage-1.jpeg)

5. 完整的Self-Attention流程图
![](DraggedImage-2.jpeg)

### 代码分析
接下来通过OpenNMT-tf（由哈佛大学自然语言处理研究组开源）项目的TensorFlow2代码，更深入地介绍Self-Attention的实现与计算技巧。
```python
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        num_units,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_units_per_head = num_units // num_heads
        self.linear_queries = common.Dense(num_units)
        self.linear_keys = common.Dense(num_units)
        self.linear_values = common.Dense(num_units)
        self.linear_output = common.Dense(num_units)
```
定义一个MultiHeadAttention类只需要传入两个参数，numheads指的是头数，num units指的是输入向量x和最后输出向量z的维数（默认相同）。
读入两个参数后，该类执行了三个操作：
1. 保存numheads为本实例变量。
2. 计算每个头的Attention维数，即向量q，k，v的维数。
3. 构造四个**全连接层**，分别用于计算向量q、k、v和输出向量z。
代码中的common.Dense是由 tf.keras.layers.Dense 封装而成的全连接层，其核心功能如下：
```python
output = self.activation(tf.matmul(inputs, self.kernel))
```
inputs是输入向量x，self.kernel是权重矩阵WQ,WK,WV,WO，输入向量与权重矩阵通过tf.matmul函数进行矩阵乘法运算后，再经过activation函数（即激活函数），outputs为q,k,v,z向量。
定义完MultiHeadAttention类后，下面的代码实现了Attention的计算。
```python
def split_heads(inputs, num_heads):
    """Splits a tensor in depth.

    Args:
      inputs: A ``tf.Tensor`` of shape :math:`[B, T, D]`.
      num_heads: The number of heads :math:`H`.

    Returns:
      A ``tf.Tensor`` of shape :math:`[B, H, T, D / H]`.
    """
    shape = misc.shape_list(inputs)
    outputs = tf.reshape(inputs, [shape[0], shape[1], num_heads, shape[2] // num_heads]) # reshape实现矩阵变换
    outputs = tf.transpose(outputs, perm=[0, 2, 1, 3]) 
	# transpose实现矩阵转置
    return outputs

def call(self, inputs, memory=None, mask=None, cache=None, training=None):

        def _compute_kv(x):
            keys = self.linear_keys(x)
            keys = split_heads(keys, self.num_heads)
            values = self.linear_values(x)
            values = split_heads(values, self.num_heads)
            return keys, values

        # Compute queries.
        queries = self.linear_queries(inputs)
        queries = split_heads(queries, self.num_heads)
        queries *= self.num_units_per_head ** -0.5 # 缩放操作

        # Compute keys and values.
        if memory is None: # 当前层为Self-Attention层,根据inputs计算k和v
            keys, values = _compute_kv(inputs)
            if cache:
                keys = tf.concat([cache[0], keys], axis=2)
                values = tf.concat([cache[1], values], axis=2)
        else: # 当前层为Encoder-Decoder层,根据memory计算k和v
            if cache:
                keys, values = tf.cond(
                    tf.equal(tf.shape(cache[0])[2], 0),
                    true_fn=lambda: _compute_kv(memory),
                    false_fn=lambda: cache,
                )
            else:
                keys, values = _compute_kv(memory)

        # Dot product attention.
        dot = tf.matmul(queries, keys, transpose_b=True)
        
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, 1)  # Broadcast on head dimension.
            dot = tf.cast(
                tf.cast(dot, tf.float32) * mask + ((1.0 - mask) * tf.float32.min),
                dot.dtype,
            )

        attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)
        heads = tf.matmul(drop_attn, values)

        # Concatenate all heads output.
        combined = combine_heads(heads)
        outputs = self.linear_output(combined)
        if self.return_attention:
            return outputs, cache, attn
        return outputs, cache
```
首先观察函数的输入：
1. inputs：输入一个三维矩阵x，维度分别为【 B,T,D】；B表示Batch size（一个Batch包含多少个句子），T代表lengTh（一个句子有几个单词，由batch内最长的句子决定），D代表Dimension（每个词向量的维数）。
2. memory：Encoders模块最终输出的特征向量，用于Decoders模块的Encoder-Decoder Attention计算。
3. cache：生成第n+1个译文时，Decoders保留的前n个词的所有层的中间量（k，v向量）。

compute kv函数在计算矩阵K、V的过程中调用了一个split heads函数，其目的是将合并计算的K和V矩阵分成8个子矩阵，即8头分割。
在计算完矩阵Q后，这一步实现了对矩阵Q的缩放：
```python
queries *= self.num_units_per_head ** -0.5
```
随后，通过判断memory是否为空的操作，Self-Attention层和Encoder-Decoder层都得到了由向量q，k，v组成的矩阵。
于是，将矩阵Q左乘上矩阵K的转置，得到了分值矩阵Dot。
```python
dot = tf.matmul(queries, keys, transpose_b=True)
```
> Dot(i,j)表示第i个词的向量q与第j个词的向量k的内积，即第i个词和第j个词的相关性分值。
将不同长度的句子合成一个Batch时，必然会出现**句子长短不一**的情况，但是将文字变成特征向量矩阵存储时必须保证每个句子的长度一致（由代码实现的逻辑决定）。
为了解决这个问题，Padding操作会在短句后面补充相应默认数量的标识符，表示为空）。
而为了消除Padding字符带来的干扰，根据原始输入矩阵生成Mask矩阵（即掩码矩阵，大小与原始矩阵一致，Padding字符所在位置为0，其余位置为1）。那么Mask矩阵对应为1的位置保留Dot矩阵元素，对应0的位置的元素被强行赋值为负无穷（计算机fp32精度最小值为tf.float.min）。
对应代码如下：
```python
dot = tf.cast(tf.cast(dot, tf.float32) * mask + ((1.0 - mask) * tf.float32.min),dot.dtype,)
```

随后用tf.nn.Softmax对Dot矩阵求概率分布
```python
attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)
```
公式为![](DraggedImage-3.jpeg) attn(i,j)表示第i个词与其他所有位置的词的关联概率。
将求得的概率矩阵与矩阵V相乘，即可得到每一个Single-Head Attention的输出矩阵Z，即![](DraggedImage-4.jpeg)
> 在整个计算过程中，Multi-Head的V计算已经被融合在矩阵运算中，即同时进行8-head Attention的计算，这种处理方式能大大加快Self-Attention的计算速度
```python
combined = combine_heads(heads)
outputs = self.linear_output(combined)
```
故这段代码将不同维度的Multi-Head的计算结果拼接在一起。
随后，将拼接后的向量经过全连接层（权重矩阵WO），得到最终的输出Z。
_在作者看来，Self-Attention的核心思想就是将句子不同位置的语义信息关联到每个词的语义信息中，从而尽可能地消除词本身的歧义_

## 三、位置编码：为什么有效
首先看下面两句话：
> 他说，你欠我那一百万元不用还了。
> 他说，我欠你那一百万元不用还了。
回忆Self-Attention的计算过程，_如果没有位置编码_，则在这两句话中，虽然相同的词对应的语义特征向量完全相同，但两句话意思完全不同，所以Transformer必须要有位置编码。
> CNN中的卷积操作本身带有空间信息，RNN的文本输入顺序也包含了位置信息，所以他们都不需要额外的位置编码模块
Self-Attention具有_无视距离_同时提取词语间关联信息的能力，故需要额外引入位置信息模块。
论文《Attention is all you need》中提出了一种基于三角函数的位置编码算法，公式如下：![](DraggedImage-5.jpeg)
> PE表示位置编码后得到的向量，pos表示词的词序，2i和2i+1分别表示词特征向量的偶数位置和奇数位置，dmodel表示特征向量的维数
简而言之，特征向量的偶数位置用正弦函数计算，奇数位置用余弦函数计算。
经过证明，使用三角函数的绝对编码是可以学习到相对位置的，证明如下：![](DraggedImage-6.jpeg)

## 四、单向掩码：另一种掩码机制
前面在Self-Attention代码分析那里提到了Mask操作，其目的是消除由句子长度不一致而衍生的Padding字符对特征提取过程的噪声干扰。
在实际训练过程中，Decoders接收的是多个完整的句子组成的、带有Padding字符的矩阵，而添加Mask矩阵的目的是：在输入全量的目标端译文时，利用Mask矩阵实现译文目标端的部分可见效果，保证逻辑推导的因果性。
Mask矩阵的生成过程：![](DraggedImage-7.jpeg)

## 五、代码解读：模型训练技巧
接下来介绍两个提升_模型训练速度_和_收敛稳定性_的训练技巧，不仅适用于Transformer，也可用在结构相似的模型上。



