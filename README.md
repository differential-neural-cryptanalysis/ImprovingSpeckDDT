# ND 可解释性代码：是对基于DDT的区分器的改进的区分器的生成和测试代码。

## 功能描述：
    生成和测试5~8轮Speck32/64的DD、AD_YD、ND。
## 测试方法：
    生成2^19个随机数据，测试准确率（Accuracy, ACC）、真正率(True Positive Rate, TPR)、真负率(True Negative Rate,TNR)、均方误差（Mean square error, MSE）, 和运行时间。

# ND 可解释性代码描述： 改进对缩减轮 Speck32/64 的基于 DDT 的区分器

- `ImprovingSpeckDDT`：该目录包含改进基于 DDT 的对缩减轮 Speck32/64 的区分器的实现代码、对Gohr给出的Speck32/64的5~8轮神经区分器的准确率测试代码：

     - `speck_improvedDD.cpp`：区分器 **DD**、**AD_YD** 的 C++ 实现和测试代码。 可以使用以下命令进行编译：`g++ -o D -O3 ./speck_improvedDD.cpp`

     - `speck_ND.py`：对Gohr给出的Speck32/64的5~8轮神经区分器的准确率测试代码

     - `speck_ddt.cpp`：计算缩减轮 Speck32/64 的 DDT 的程序，生成名为 **DD** 的区分器。 可以使用 makefile 对其进行编译。 它是从 GitHub 存储库 [deep_speck](https://github.com/agohr/deep_speck) 下载而来。

     - `speck.py`：Speck32/64 的实现

     - `Results`：各种区分器的准确性和性能的测试结果。
         - 各种区分器的测试结果
           - `5R_8R_speck_eval_DDT_TND19.log`：DD
           - `5R_8R_speck_eval_conditional_DDT_withPr_carry_TND19.log`：AD_YD
           - `5R_8R_speck_ND.log`: ND

## 参考文献

[BLYZ23] Zhenzhen Bao, Jinyu Lu, Yiran Yao, Liu Zhang: More Insight on Deep Learning-aided Cryptanalysis. IACR Cryptol. ePrint Arch. 2023: 1391 (2023), ASIACRYPT 2023.