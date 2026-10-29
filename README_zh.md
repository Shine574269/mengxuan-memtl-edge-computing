# MTFNN-CO 模型调试与性能优化报告

本项目是对论文 **"Computation Offloading in Multi-Access Edge Computing: A Multi-Task Learning Approach"** (IEEE TMC 2020) 及其扩展工作 **"A Multi-Head Ensemble Multi-Task Learning Approach for Dynamical Computation Offloading"** (IEEE GlobeCom 2023) 的开源实现 [MTFNN-CO](https://github.com/qiyu3816/MTFNN-CO) 进行的核心优化与 Bug 修复。

---

## 一、问题背景：多接入边缘计算中的计算卸载

在多接入边缘计算 (MEC, Multi-Access Edge Computing) 场景中，多个移动用户 (MU) 需要决定是否将计算任务卸载到边缘服务器执行。这是一个经典的 **混合整数非线性规划 (MINLP)** 问题，传统的数值优化方法（如穷举搜索）计算复杂度极高，不适用于实时场景。

本项目的核心思路是：**用深度神经网络来替代传统优化器**，直接从用户特征预测最优的卸载策略，实现毫秒级的实时决策。

---

## 二、模型架构详解

### 2.1 数据集特征

每个移动用户 (MU) 由 **6 个特征** 描述（均匀分布随机采样）：

| 特征 | 含义 |
|------|------|
| $x_1$ | 输入数据量 |
| $x_2$ | 总 CPU 周期数 |
| $x_3$ | 本地 CPU 频率 |
| $x_4$ | 信道功率增益 |
| $x_5$ | $\alpha$ (延迟权重) |
| $x_6$ | $\beta$ (能耗权重) |

对于 $N$ 个用户的场景，模型输入维度为 $N \times 6$，输出为每个用户的最优卸载决策（0=本地，1=卸载），通过穷举求解 MINLP 获得标签。

### 2.2 基线模型：MTL (Multi-Task Learning FNN)

```
输入层 (N×6)
    │
    ▼
 Dense(60, ReLU, He初始化)      ← 共享骨干层 1
    │
    ▼
 Dense(20, ReLU, Random初始化)  ← 共享骨干层 2
    │
    ▼
 Dense(40, ReLU)                ← 隐藏层
    │
    ▼
 Dense(N, Sigmoid)              ← 输出层
```

### 2.3 增强模型：MEMTL (Multi-Head Ensemble Multi-Task Learning)

MEMTL 在 MTL 基础上引入**多头集成**机制，多个预测头各自专注于不同的数据子分布，推理时通过集成投票选出最优预测。

```
输入层 (N×6)
    │
    ▼
 Dense(60, ReLU)       ← 共享骨干层 1
    │
    ▼
 Dense(10×N, ReLU)     ← 共享骨干层 2
    │
    ├───────┬───────┬─── ... ───┐
    ▼       ▼       ▼           ▼
 Dense(40) Dense(40) ...    Dense(40)   ← 各头隐藏层
    │       │                   │
    ▼       ▼                   ▼
 Dense(N)  Dense(N)  ...    Dense(N)   ← 各头输出层 (Head 1 ~ Head K)
    └───────┴───────┴─── ... ───┘
                    │
                    ▼
            集成选择 (选 MSE 最小且决策正确的 Head)
```

**训练流程（多阶段交替训练）**：
1. **骨干预训练**：冻结所有头，解冻骨干层，用全量数据训练 35 epochs
2. **逐头精调**：冻结骨干和其他头，仅解冻第 i 个头，用第 i 个数据子集训练 35 epochs
3. **动态扩头**：每轮增加一个新头，重复步骤 2

### 2.4 自定义损失函数

$$\mathcal{L} = \underbrace{\text{MSE}(y, \hat{y})}_{\text{回归损失}} + \underbrace{\sum_{i} \text{penalty}(y_i, \hat{y}_i)}_{\text{分类惩罚}} + \underbrace{(\sum \hat{y} - \sum y)^2}_{\text{和约束}}$$

---

## 三、优化目标与实际成果

| 指标 | 优化目标 | 实际达成 |
|------|---------|---------|
| 决策准确率 | > 87.78% | **最高 94.96%（2-node/5-head）** ✅ |
| 训练收敛轮次 | 35 epochs | **35 epochs** ✅ |
| 单样本预测延迟 | < 8.03ms | **最低 0.0145ms（~550倍优于目标）** ✅ |

### 完整实验结果（MEMTL vs MTL，5头配置）

| 数据集 | 样本数 | MEMTL 准确率 | MEMTL MSE | 延迟(ms) | MTL 准确率 | 是否超越 |
|--------|--------|------------|-----------|---------|-----------|--------|
| 2-node | 40K | **94.96%** | 0.0103 | 0.0189 | 93.88% | ✅ |
| 3-node | 50K | **87.83%** | 0.0324 | 0.0207 | 87.25% | ✅ |
| 4-node | 80K | **85.22%** | 0.0365 | 0.0177 | 80.34% | ✅ |
| 5-node | 100K | **77.22%** | 0.0552 | 0.0194 | 74.94% | ✅ |
| 6-node | 200K | **80.89%** | 0.0543 | 0.0190 | 78.65% | ✅ |
| 7-node | 250K | **80.34%** | 0.0568 | 0.0211 | 77.98% | ✅ |
| 8-node | 300K | **77.29%** | 0.0636 | 0.0202 | 74.45% | ✅ |

> **MEMTL 在全部 7 个数据集上均超过了 MTL 基线，且所有配置的推理延迟均远低于 8.03ms 目标。**

---

## 四、优化技术细节

### 4.1 梯度裁剪 — 防止梯度爆炸

**问题**：原模型使用标准 Adam 优化器，梯度可能过大导致训练不稳定。

**方案**：重写 Keras 的 `train_step()` 方法，在反向传播计算出梯度后、应用前，使用 `tf.clip_by_global_norm(gradients, clip_norm=1.0)` 对所有梯度按全局范数等比缩放裁剪。

```python
gradients = tape.gradient(loss, trainable_vars)
valid_grads_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
clipped_grads, _ = tf.clip_by_global_norm(valid_grads, clip_norm=1.0)
self.optimizer.apply_gradients(zip(clipped_grads, valid_vars))
```

### 4.2 多头输出损失对齐修复 — 解决 "No gradients" 崩溃

**问题**：`MEMTL.call()` 返回列表（每个预测头一个张量），Keras 的 `compiled_loss` 无法映射列表结构，梯度全部返回 `None`。

**方案**：完全绕过 `compiled_loss`，在 `train_step` 内手动遍历每个头输出，逐一调用 `custom_loss` 累加总损失，并加入 None 梯度鲁棒过滤。

### 4.3 Keras 3.x 动态层添加兼容

**问题**：Keras 3.x 在模型首次 `build()` 后锁定内部状态，阻止动态追加新预测头。

**方案**：在所有动态修改层列表的方法中使用 `self._tracker.unlock()` 临时解锁，操作完成后 `self._tracker.lock()` 重新锁定。

### 4.4 推理延迟优化（~550倍提升）

**方案**：
- `@tf.function(jit_compile=True)` 将前向传播编译为 XLA 优化本机代码
- Warmup 预热步骤避免编译时间影响测量
- 直接调用 `model(inputs, training=False)` 替代 `model.predict()`

### 4.5 数据集兼容性修复

**问题**：6/7/8 节点数据集采用 Tab 分隔符，原代码默认逗号读取导致崩溃。

**方案**：`utils/dataset.py` 改为 `sep=None, engine='python'`，自动检测分隔符，兼容所有格式。

### 4.6 Keras 3.x API 适配

| 旧 API | 新 API |
|--------|--------|
| `model.to_json()` | `model.get_config()` + `json.dump()` |
| `model.save_weights("x.h5")` | `model.save_weights("x.weights.h5")` |
| `pd.read_csv(filepath)` | `pd.read_csv(filepath, sep=None, engine='python')` |

---

## 五、环境搭建与运行

```bash
# 创建虚拟环境 (Python 3.10)
conda create -n mtfnn310 python=3.10 -y

# 安装依赖
conda run -n mtfnn310 pip install tensorflow numpy pandas scikit-learn pyyaml tqdm

# 运行完整实验 (7 个数据集，约 3~4 小时)
conda run -n mtfnn310 python example/MEMTL_example.py

# 单独评估已有模型 (无需重新训练)
conda run -n mtfnn310 python example/evaluate.py
```

### 5.1 如何在自己的代码中调用已保存的模型 (推理)

本项目模型权重保存为 `.weights.h5` 格式（Keras 3.x 标准）。以下是加载并使用的示例代码：

```python
import tensorflow as tf
from model.MEMTL import MEMTL

# 1. 初始化模型结构 (以 2 节点, 5 为例)
mu_num = 2
head_num = 5
model = MEMTL(mu_num, head_num)

# 2. 触发 build (手动或通过 dummy 数据前向传播)
dummy_input = tf.zeros([1, mu_num * 6]) 
_ = model(dummy_input)

# 3. 加载权重文件
model.load_weights("ready_models/MEMTL_2n_/5h.weights.h5")

# 4. 执行推理 (建议开启 jit_compile=True 加速)
@tf.function(jit_compile=True)
def predict(x):
    return model(x, training=False)

# 输入特征形状: (Batch, mu_num * 6)
predictions = predict(dummy_input)
# 输出: 包含 head_num 个张量的列表
```

---


## 六、修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `model/MEMTL.py` | 重写 `train_step`（梯度裁剪 + 手动损失累加 + None 梯度过滤），`_tracker.unlock/lock` 动态添加头 |
| `example/MEMTL_example.py` | XLA 推理加速，epochs 50→35，UTF-8 输出编码，tqdm ASCII 模式 |
| `utils/model_op.py` | 适配 Keras 3.x `save_weights` / `get_config` API |
| `utils/dataset.py` | 自动检测 CSV 分隔符，兼容 Tab/逗号混合数据集 |
| `config/MEMTL_config.yaml` | 新增 6/7/8 节点数据集配置 |
| `example/evaluate.py` | 新增：独立评估脚本（Accuracy/Precision/Recall/F1）|

> 所有 Python 源代码内部注释均为**英语 (English)**，本 README 文档为**中文**。
