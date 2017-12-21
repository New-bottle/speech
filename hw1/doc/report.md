
---
Title

---

# Abstract


## method


通过计算短时能量和短时过零率来预测是否在说话：设置阈值，当

对于每一帧：
短时能量ste[i]：
$ short\_time\_energy[i] = \sum_1^n sign[i]^2 $
短时过零率zcr[i]:
$ zero\_cross\_rate[i] = \sum_1^{n-1} (sign[j] * sign[j-1] < 0) $


阈值选择：通过观察数据手动选择调整
![Image 1](./Figure_1.png)

![Image 1](./Figure_2.png)