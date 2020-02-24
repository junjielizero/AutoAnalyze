# Fast Select Important Features

## 业务场景

* 了解哪些特征（变量）影响如流失/非流失，活跃/非活跃，高价值/低价值等yes-no目标
* 快速、能分析多个特征，并能量化与可视化说明

## 解决方案

### 原理

* woe 技术分箱切割，iv 筛选变量，[具体参见](https://cloud.tencent.com/developer/news/32692)
* 自动化切割实现，[参考 scorecardpy](https://github.com/ShichenXie/scorecardpy)

### 优势

* 对连续变量等箱自动化切割为分类变量，避免正太分布要求，且能较好可视化
* 无需考虑目标变量（如流失/非流失）是否平衡
* 可自动化将差别不大的连续变量区间压缩

## 使用方式

* 在 data 下放入数据，本示例以 titanic 作为示范，不建议数据含中文
* 预先安装 pandas/numpy/matplotlib/scipy/seaborn/scorecardpy
* 运行 notebook ，修改文件名与目标值即可
 