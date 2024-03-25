# repo结果汇报
**目前已有的实验效果:[表1]** 

(使用本代码跑出来的transformers的复现内容,BLEU指标数）

case_1:casetransformer_layer_1.py(随机初始化-->SVD分解-->中间的进行BP更新，左右两边固定)

case_2:transformer_layer_2.py(随机初始化-->SVD分解-->中间的赋值为1固定，左右两边固定-->三个都固定)

case_3:transformer_layer_3.py(随机初始化-->SVD分解-->中间的赋值为1，左右两边固定-->中间的从1进行BP更新)

case_4:transformer_layer_4.py(直接去掉一层)

| transformrs | transformrs<br>ELM(fixed) | transformrs+ELM<br>(fixed+node*1.2) |  transformrs+ELM<br>(fixed+node*1.5)|case_1|case_2|case_3|case_4|
| :----: | :----: | :----: |:----: |:----:|:----:|:----:|:----:|
|33.37|29.25| 27.98 |29.51|8.18|30.67|30.82|29.04|

**bias结果挑选:[表2]** 

case_5:transformer_case5.py(使用QR分解，Q作为权重在后面不进行更新，bias从0开始更新)

case_6:transformer_case5.py(使用QR分解，Q作为权重在后面不进行更新，bias从randn开始更新)

case_7:transformer_case5.py(使用QR分解，Q作为权重在后面不进行更新，bias从0.1*randn开始更新)

case_8:transformer_case5.py(使用QR分解，Q作为权重在后面不进行更新，bias为0，且不更新)

|case_5|case_6|case_7|case_8|
|:----:|:----:|:----:|:----:|
|29.87 |29.21 |29.93 ||

**参数搭配[表3]:**
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
