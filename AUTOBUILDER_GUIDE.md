# 自动构建脚本通用架构指南

为了回答“有没有通用的架构，可以随意调取已有模块构造模型，并进行训练”这个问题，本仓库新增了一个轻量的 `autobuilder/` 工具包，用来把 PlugNPlay 模块动态拼装成可训练网络。该工具包围绕三个核心组件展开：

1. **模块注册器（`autobuilder/registry.py`）**：支持从仓库任意 `.py` 文件中动态导入 `nn.Module` 子类，并为其注册一个易记名称；同时兼容标准库/第三方库的全限定导入路径，例如 `torch.nn.Conv2d`。
2. **计算图构建器（`autobuilder/graph.py`）**：读取声明式配置，把节点列表（模块或内置算子）解析成一个可执行的 `DynamicGraphModel`。节点既可以是仓库模块，也可以是加法、拼接、展平等轻量操作。
3. **训练循环（`autobuilder/trainer.py`）**：提供一个最小可用的训练器，支持自动选择设备、AMP 混合精度、学习率调度器，并兼容 `DataLoader` 产出的常见批数据格式。

结合以上组件，用户可以通过一个 YAML 配置文件完成“注册模块 → 构建模型 → 训练测试”的全流程。下面以 `autobuilder/configs/example_classification.yaml` 为例，展示完整的调用方式。在运行示例脚本前，请确保已经安装 PyTorch（例如 `pip install torch torchvision`）。

## 1. 在配置中注册模块

```yaml
registry:
  - file: DA_Block.py        # 仓库中的文件路径
    class_name: DA_Block     # 文件内的类名
    alias: DA_Block          # 注册时使用的别名，可选
```

注册器会把 `DA_Block.py` 中的 `DA_Block` 类加载进来，以后在模型节点中即可直接写 `target: DA_Block`。如果需要一次性注册整个文件中的所有模块，可以去掉 `class_name` 字段。

## 2. 声明模型拓扑

```yaml
model:
  inputs: [x]
  outputs: [head]
  nodes:
    - name: stem
      target: torch.nn.Conv2d
      init: {in_channels: 3, out_channels: 32, kernel_size: 3, padding: 1}
      inputs: [x]
    - name: stem_bn
      target: torch.nn.BatchNorm2d
      init: {num_features: 32}
      inputs: [stem]
    - name: stem_act
      target: torch.nn.ReLU
      init: {inplace: true}
      inputs: [stem_bn]
    - name: dab
      target: DA_Block         # 调用上一节注册的仓库模块
      init: {in_channels: 32}
      inputs: [stem_act]
    - name: pool
      target: torch.nn.AdaptiveAvgPool2d
      init: {output_size: 1}
      inputs: [dab]
    - name: flatten
      op: flatten              # 内置算子示例
      call: {start_dim: 1}
      inputs: [pool]
    - name: head
      target: torch.nn.Linear
      init: {in_features: 32, out_features: 10}
      inputs: [flatten]
```

- `target` 表示需要实例化的 `nn.Module` 类名，可以是注册器中的别名，也可以是 `torch.nn.*` 之类的全路径；`init` 字段对应构造函数参数。
- `op` 用于声明无需参数的基础算子（加法、拼接、乘法、展平等），`call` 字段控制前向调用时的额外参数。
- `inputs` 指定该节点的前驱名称，构建器会按拓扑顺序依次执行。
- `inputs`/`outputs` 顶层字段定义了整个模型的输入变量名和输出节点。

## 3. 声明训练组件

```yaml
loss:
  target: torch.nn.CrossEntropyLoss

optimizer:
  target: torch.optim.Adam
  params: {lr: 0.001}

trainer:
  epochs: 2
  log_every: 5
```

`Trainer.from_config` 会自动按照 `loss`、`optimizer`、`scheduler`（可选）等字段实例化训练所需对象，并且支持 `device: auto`、`mixed_precision: true` 等扩展参数。

## 4. 运行示例脚本

```bash
python -m autobuilder.example_usage \
  --config autobuilder/configs/example_classification.yaml
```

`example_usage.py` 会完成以下流程：
1. 读取配置并注册所需模块；
2. 基于配置构建 `DynamicGraphModel`；
3. 生成一个随机图像分类数据集（512 张 32×32 彩图，10 类）以验证训练链路；
4. 调用 `Trainer` 完成两轮训练，并在控制台打印训练/验证损失。

你可以把 `registry` 和 `model.nodes` 中的 `target` 替换成任何仓库已有模块（例如 `GFM.py`、`AFPN.py` 等），并结合自定义的骨干或检测头节点，即可快速尝试不同的组合架构；同时也可以扩展配置 schema，以满足更复杂的并联/跨层连接需求。

## 5. 扩展建议

- **自动扫描模块**：`ModuleRegistry` 暴露的 `from_files` / `register_from_file` 接口可以嵌入你的自动脚本，根据文件命名或目录策略批量注册模块。
- **自定义算子**：如果需要更多图操作（例如 residual、注意力权重计算等），可在 `DynamicGraphModel._execute_op` 中按需扩展。
- **数据/训练策略**：`Trainer` 采用最小实现，便于根据项目需求加入评价指标、梯度裁剪、模型保存等功能。
- **YAML 复用**：将常用的骨干、neck、head 配置拆分成片段，脚本层面再做组合，可以快速枚举多种候选结构。

通过上述通用架构，可以将仓库中丰富的即插即用模块纳入统一的配置化构建体系，既方便实验管理，也便于脚本自动搜索和训练。具体实现细节可参考 `autobuilder/` 目录中的代码。
