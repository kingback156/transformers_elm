# repo结果汇报
## STAGE2:全部在做transELM[表4]

{**case1-3，5**:model_dim=512 ; ffn_dim=1024;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**case4**:model_dim=1024;ffn_dim=4096;}

{**case1-4**:drop_out=0.1;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**case5**:drop_out=0.3}

|case1<br>other_weightdecay=0<br>fc2_weightdecay=0.0001|case2<br>all_weightdecay=0 |case3<br>all_weightdecay=0.0001 |case4<br>other_weightdecay=0<br>fc2_weightdecay=0.0001<BIG model>|case5<br>other_weightdecay=0<br>fc2_weightdecay=0.0001<br>drop_out=0.3|
| :----: | :----: | :----: |:----: |:----: |
|30.80|30.70| 28.84 |23.82|27.55|

||ffn_dim*1.05|ffn_dim*1.10|ffn_dim*1.15|ffn_dim*1.20|ffn_dim*1.25|ffn_dim*1.30|
|:----:| :----: | :----: | :----: |:----: |:----: |:----: |
|参数量|other=35963400<br>fc2=6610944 | other=36277356<br>fc2=6924288 | other=36591312<br>fc2=7237632 |other=36905268<br>fc2=7550976 |other=37225380<br>fc2=7870464 |other=37539336<br>fc2=8183808 |
|BLEU|31.16|31.76|31.69|31.51|31.93|31.60|

## 目前已有的实验效果:[表1]

(使用本代码跑出来的transformers的复现内容,BLEU指标数）

case_1:casetransformer_layer_1.py(随机初始化-->SVD分解-->中间的进行BP更新，左右两边固定)

case_2:transformer_layer_2.py(随机初始化-->SVD分解-->中间的赋值为1固定，左右两边固定-->三个都固定)

case_3:transformer_layer_3.py(随机初始化-->SVD分解-->中间的赋值为1，左右两边固定-->中间的从1进行BP更新)

case_4:transformer_layer_4.py(直接去掉一层)

| transformrs | transformrs<br>ELM(fixed) | transformrs+ELM<br>(fixed+node*1.2) |  transformrs+ELM<br>(fixed+node*1.5)|case_1|case_2|case_3|case_4|
| :----: | :----: | :----: |:----: |:----:|:----:|:----:|:----:|
|33.37|29.25| 27.98 |29.51|8.18|30.67|30.82|29.04|

## bias结果挑选:[表2]

case_5:transformer_case5.py(使用QR分解，Q作为权重在后面不进行更新，bias从0开始更新)

case_6:transformer_case5.py(使用QR分解，Q作为权重在后面不进行更新，bias从randn开始更新)

case_7:transformer_case5.py(使用QR分解，Q作为权重在后面不进行更新，bias从0.1*randn开始更新)

case_8:transformer_case5.py(使用QR分解，Q作为权重在后面不进行更新，bias为0，且不更新)

|case_5|case_6|case_7|case_8|
|:----:|:----:|:----:|:----:|
|29.87 |29.21 |29.93 |27.97 |

## FC1的Bias在高斯初始化的情况下调整FC2的weight_decay:[表2.5]

case_9:transformer_case9.py(高斯初始化+FC2的weight_decay调整)

因为修改较多，提交了

(1)"transformer_case9.py"; [Transformer/modules/transformer_case9.py]

(2)"train2.py"; [main folder/train2.py]

(3)"transformer2.py" [Transformer/models/transformer2.py]

|只有高斯初始化|高斯+weight_decay<br>=0.0001|高斯+weight_decay<br>=0.00005|高斯+weight_decay<br>=0.001|
|:----:|:----:|:----:|:----:|
| | 31.42| | |

## 参数搭配[表3]:
|             |model_dim=512<br>ffn_dim=1024<br>FFN_dropout=attention_dropout=0.1|model_dim=512<br>ffn_dim=2048<br>FFN_dropout=attention_dropout=0.1<br>(base)|model_dim=1024<br>ffn_dim=4096<br>FFN_dropout=attention_dropout=0.3<br>(big)|
| :----:      | :----: |:----:|:----:|
| transformrs | 33.37  |32.79|29.74|
|case_2      |30.67   |27.96|16.59|
|case_4      |29.04   |28.20|25.23|

# [表1]训练过程可视化：
<table>
  <tr>
    <td><img width="1044" alt="Snipaste_2024-03-17_15-46-47" src="https://github.com/kingback156/transformers_elm/assets/146167978/9f80aa5a-6250-4898-8b95-b7f25fc1987a" scale=0.5></td>
    <td><img width="1057" alt="Snipaste_2024-03-17_15-47-37" src="https://github.com/kingback156/transformers_elm/assets/146167978/79569935-3069-4c62-ade3-c67d4a7be19b" scale=0.5></td>
  </tr>
</table>

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
