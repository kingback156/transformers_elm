# TODO，按照优先级：
1. + 1/c
2. 可视化loss，包括valid集
3. Singular Value Recognition

4. 在纯的ELM上实验householder

# repo结果汇报
**目前已有的实验效果:** 

(使用本代码跑出来的transformers的复现内容,BLEU指标数）

(因为如果设置的全部和transformers的参数配置全部一样，那么将要非常长的时间，在原文中使用**base模型**单次训练就跑了**12小时**，使用**big模型**单次训练便**3.5天**，**复现的时间消耗太大**，因此为了尽快的看到所有的复现结果，我调整了参数设置，这样便可以尽快的看到不同的设置所导致的bleu结果趋势)

| transformrs | transformrs+ELM | transformrs+ELM<br>(node*1.2) |  transformrs+ELM<br>(node*1.5)|layer_1|layer_2|layer_3|layer_4|
| :----: | :----: | :----: |:----: |:----:|:----:|:----:|:----:|
| 27.9 | 21.49 | 22.06 |22.43|||||
|33.37|  |  |||||

**在/Transformer/modules下面发布主要的结构修改内容:**

transformer_layer_1.py(随机初始化-->SVD分解-->中间的进行BP更新，左右两边固定)

transformer_layer_2.py(随机初始化-->SVD分解-->中间的赋值为1固定，左右两边固定-->三个都固定)

transformer_layer_3.py(随机初始化-->SVD分解-->中间的赋值为1，左右两边固定-->中间的从1进行BP更新)

transformer_layer_4.py(直接去掉一层)

**在/translation下面发布不同的结构对应的翻译:**

translation(0-4)

其中translation_0对应的最原始的transformers,其余的标号都各自对应上面的结构更改。

# 主要参数配置
使用的是transformers（ltl）模型，这里使用的参数配置如下：
```
model_dim: 512
ffn_dim: 1024
head_num: 4
encoder_layers: 6
decoder_layers: 6
droprate=0.1
epoch=10
```
# 主要修改部分
1.在encoder和decoder中都加了一个svdchange函数，专门做svd分解。

2.在这个里面的其他部分都是常规操作，主要是做的self.diag_param = nn.Parameter(torch.diag(S))，分解出来的对角矩阵提取他的对角，这样确保不是更改整个矩阵，而是更改这个向量

3.在这个forward部分，把E通过对角的引入重建出来
```
E = torch.zeros(self.U.size(0), self.V.size(1), device=net_input.device)
min_dim = min(E.size(0), E.size(1))
E[torch.arange(min_dim), torch.arange(min_dim)] = self.diag_param
```
4.其他的都没有太大的变化
