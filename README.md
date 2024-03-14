# transformers_elm
Hello, I put the .py files and data in the repository.

The main modification operations are in "/Transformer/modules/transformer_layer.py"

**在/Transformer/modules下面发布:**

transformer_layer_1.py(随机初始化-->SVD分解-->中间的进行BP更新，左右两边固定)

transformer_layer_2.py(随机初始化-->SVD分解-->中间的赋值为1固定，左右两边固定-->三个都固定)

transformer_layer_3.py(随机初始化-->SVD分解-->中间的赋值为1，左右两边固定-->中间的从1进行BP更新)
