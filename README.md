# transformers_elm
Hello, I put the .py files and data in the repository.

The main modification operations are in "/Transformer/modules/transformer_layer.py"

**在/Transformer/modules下面发布:**

transformer_layer_1.py(随机初始化-->SVD分解-->中间的进行BP更新，左右两边固定)

transformer_layer_2.py(随机初始化-->SVD分解-->中间的赋值为1固定，左右两边固定-->三个都固定)

transformer_layer_3.py(随机初始化-->SVD分解-->中间的赋值为1，左右两边固定-->中间的从1进行BP更新)

**目前已有的实验效果:** （使用本代码跑出来的transformers的复现内容,BLEU指标数）

| transformrs | transformrs+ELM | transformrs+ELM(node*1.2) |  transformrs+ELM(node*1.5)|
| :----: | :----: | :----: |:----: |
| 27.9 | 21.49 | 22.06 |22.43|

# 主要修改部分
1.在encoder和decoder中都加了一个svdchange函数，专门做svd分解。

2.在这个里面的其他部分都是常规操作，主要是做的self.diag_param = nn.Parameter(torch.diag(S))，分解出来的对角矩阵提取他的对角，这样确保不是更改整个矩阵，而是更改这个向量

3.在这个forward部分，把E通过对角的引入重建出来
![image](https://github.com/kingback156/transformers_elm/assets/146167978/74eead85-f32a-44a1-8ac0-80bf33332032)

4.其他的都没有太大的变化
