# 顶层未分类模块梳理

为方便模型自动构建脚本查找组件，以下按功能类别对仓库根目录的 108 个独立 Python 模块逐一做简要介绍。

## 目标检测增强

- **AFPN.py**：实现非对称特征金字塔网络的上下采样与尺度自适应融合单元，可作为检测颈部提升多尺度响应。
- **CPAM.py**：复现 ASF-YOLO 的通道与位置注意力分支，对多路特征进行加权融合以增强检测头。
- **MDCR.py**：提供多尺度空洞卷积残差块，将特征拆分并行感受野后重组，用于红外小目标突出。
- **MCM.py**：包含多尺度上下文调制与预测分支，服务于 RGB-D 显著性检测的模态融合与增强。
- **UCDC.py**：实现空洞卷积堆叠的 U 型特征提取结构，用于小目标检测的细粒度对比增强。

## 卷积与多尺度结构

- **(BMVC 2023)CoordGate.py**：通过坐标编码控制通道门控，按位置调节卷积响应，实现空间可变卷积。
- **(CVPR 2019) DCNv2.py**：封装可学习偏移与调制的二代可变形卷积算子，兼容 PyTorch 的 deform_conv2d。
- **(CVPR 2024)IDC.py**：InceptionNeXt 提出的分支深度可分离卷积块，将输入拆成四路并融合长短条卷积。
- **(CVPR 2024)PKIBlock.py**：实现 PKIBlock 中的多路卷积瓶颈与通道注意力组合，用于轻量 Inception 模块。
- **(CVPR2020)strip_pooling.py**：StripPooling 模块以长条池化捕获长程依赖，并与局部卷积耦合进行上下文聚合。
- **DFF2d.py**：Dynamic Feature Fusion 块使用并行卷积支路和注意力加权融合多尺度纹理。
- **FEM.py**：FFCA-YOLO 的特征增强模块，集成多尺度分支和膨胀卷积以强化小目标感受野。
- **FMS.py**：SFFNet 的波小波与卷积分支结合结构，可在空间与频域间交互并保留多尺度细节。
- **GhostModule.py**：实现 GhostNet 提出的廉价线性算子与逐点卷积组合，生成幽灵特征减少计算。
- **GlobalPMFSBlock.py**：PMFSNet 中的全局极化多尺度特征块，结合深度可分离卷积与自注意强化全局上下文。
- **HFF.py**：HiFuse 网络的层级融合骨干，交替使用深度可分离卷积与残差连接堆叠多尺度特征。
- **LDConv.py**：线性可形变卷积通过预测稀疏偏移参数实现坐标采样，以降低可变形卷积计算。
- **RFAConv.py**：RFAConv 在常规卷积基础上引入注意力权重，动态重构空间核并保留轻量性。
- **SPConv.py**：将输入分成 3×3 与 1×1 分支的分离并行卷积，辅以自适应池化实现轻量混合感受野。
- **UIB.py**：MobileNetV4 的通用倒残差块，支持多阶段深度卷积与可调扩张比。
- **dynamic_conv.py**：Dynamic Convolution 框架，根据输入生成多组卷积核权重，实现输入自适应卷积。

## 注意力与Transformer

- **(ACCV 2022)CSCA.py**：跨模态人群计数的空间通道注意力块，融合多模态特征并执行尺度自适应。
- **(ACCV 2024) LIA.py**：Local Interaction Attention 结合软池化与局部注意，实现轻量图像复原增强。
- **(Arxiv2024)MDAF.py**：MDAF 模块提供无偏与有偏 LayerNorm 以及多头动态融合结构，适配频域注意。
- **(CVPR 2022) DAT.py**：Deformable Attention Transformer 的注意力实现，支持多尺度采样点偏移。
- **(CVPR 2024)RAMiT.py**：RAMiT 引入互反注意力混合机制，包含 QKV 投影与空间自注意层。
- **(ECCV 2024)HTB.py**：Histogram Transformer Block 提供直方图引导的注意力与归一化组合应对恶劣天气。
- **(ECCV2024)SMFA.py**：SMFA 模块以深度 MLP 与空间调制提升超分辨率网络的适应性。
- **(ICCV 2021) RA.py**：Residual Attention 块结合通道和空间权重，以残差方式提升多标签识别。
- **(ICCV2023)SAFM.py**：SAFM 使用多尺度滤波与加权聚合，为轻量图像复原提供语义自适应。
- **(ICLR 2022) Crossformer.py**：CrossFormer 的跨尺度注意力与动态位置偏置实现，可用于多尺度 Transformer。
- **(ICLR 2022) MobileViTAttention.py**：MobileViT 注意力堆叠前置归一化与前馈，适配移动端视觉 Transformer。
- **(ICLR 2024)CA Block.py**：MogaNet 的通道聚合块，使用元素缩放与双层 FFN 聚合多阶信息。
- **(ICPR 2022) MOATransformer.py**：MOA Transformer 提供窗口与全局注意组合，支持多尺度聚合。
- **(NeurIPS 2021) CoAtNet.py**：CoAtNet 模块融合卷积与注意力，包含高效缩放点积注意实现。
- **(NeurIPS 2021) GFNet.py**：Global Filter Networks 的频域滤波注意力，配合 PatchEmbed 与 MLP。
- **(TPAMI 2022) ViP.py**：Vision Permutator 的加权通道置换 MLP，重排特征以模拟注意力。
- **(arXiv 2019) ECA.py**：ECA 注意力以局部一维卷积建模通道间交互，避免维度压缩。
- **(arXiv 2020 ) SSAN.py**：简化的缩放点积注意力，通过共享投影实现高效自注意。
- **(arXiv 2021) AFT.py**：Axial Fusion Transformer 使用全局权重与位置偏置构建轻量注意力。
- **(arXiv 2021) EA.py**：External Attention 通过外部记忆键值构建稀疏注意力响应。
- **(arXiv 2021) MobileViTv2.py**：MobileViTv2 模块结合线性注意力与卷积分支，用于轻量视觉骨干。
- **(arXiv 2021) PSA.py**：Pixel Shuffle Attention 通过像素重排加强空间上下文聚合。
- **(arXiv 2021) PSAN.py**：Polarized Self-Attention 提供并行与串行分支以分离水平/垂直注意。
- **(arXiv 2021) S2Attention.py**：S2Attention 拆分注意力头，结合通道重排实现分层聚合。
- **(arXiv 2023) ScaledDotProductAttention.py**：封装标准缩放点积注意力的查询、键值与缩放流程。
- **(arXiv 2024) MoHAttention.py**：MoHAttention 模块混合多阶 Hadamard 注意力以增强图像复原。
- **CPCA2d.py**：通道注意力块结合多尺度池化与卷积，输出逐通道重标系数。
- **CRMSA.py**：交叉区域多头注意通过内外部注意力桥接，服务遥感影像理解。
- **DA.py**：Double Attention Layer 将压缩特征映射成注意力图，再与值映射重分配。
- **DASI.py**：DASI 结构通过并行卷积与注意力袋，实现多尺度信息融合。
- **DA_Block.py**：Dual Attention Block 同时计算位置与通道注意并与深度可分离卷积结合。
- **DICAM.py**：DICAM 引入多尺度卷积分支与通道注意融合用于上下文增强。
- **DSAM.py**：DSAM 结合立方注意与条带注意，针对多方向特征赋权。
- **EFF2d.py**：Efficient Attention Gate 同时建模空间注意与通道门控，实现轻量特征筛选。
- **ENLTB.py**：ENLTB 块堆叠归一化、线性投影与跨层注意，实现轻量非局部交互。
- **FADConv.py**：频率自适应空洞卷积结合频域选择器与多尺度空洞率，统一卷积与注意力。
- **FCA.py**：FCAttention 模块混合卷积与注意力分支，对特征进行频域重加权。
- **FCHiLo.py**：FCHiLo 使用频域位置编码与多分辨卷积分支加强高低频协同。
- **FECAttention.py**：FEC 注意力通过离散余弦变换抽取通道频率信息并生成权重。
- **FFA.py**：FFA 模块含位置与通道注意残块，专注图像去雾去噪场景。
- **FMB.py**：FMB 堆叠动态 MLP 与并行卷积分支，提供频域调制能力。
- **FMM.py**：FMM 结合通道压缩、门控与自适应注意，强化多模态特征融合。
- **GAB.py**：Group Aggregation Bridge 使用多组卷积和注意融合连接多尺度块。
- **GAU.py**：GAU 模块包含时序和通道注意，用于视频或序列特征融合。
- **GCTattention.py**：GCT 注意力以全局上下文生成门控系数，重加权输入通道。
- **GHPA.py**：GHPA 采用分组多轴 Hadamard 乘积，捕获方向性上下文。
- **GLSA.py**：GLSA 上下文块结合多分支卷积与注意力，构建轻量语义捕获。
- **HAAM.py**：HAAM 包含通道与空间双支路注意力，用于多光谱图像。
- **HWAB.py**：HWAB 结合小波变换与自适应注意，实现频域-空间双重增强。
- **LAE.py**：LAE 模块串联卷积和注意力，通过局部注意强化边缘细节。
- **LGAG.py**：LGAG 块用局部全局注意桥接多尺度特征。
- **LPA.py**：LPA 将通道与空间注意解耦建模，用于低光图像增强。
- **MDTA.py**：多头深度可分离转置注意力（MDTA）结合深度卷积与自注意。
- **MHIASA.py**：MHIASA 在高光谱分类中结合多头交互注意与自适应融合。
- **MLA.py**：LiteMLA 模块以多尺度线性注意和卷积层提升高分辨率预测。
- **MLAttention.py**：多层级注意模块提供快速与精确两种分支应对多尺度依赖。
- **MixStructure.py**：MixStructureBlock 将 CNN 与 Transformer 结构并行融合。
- **NAF.py**：NAF 块以简化的门控与残差结构构建非注意力型 Transformer。
- **PCBAM.py**：PCBAM 整合通道与空间注意模块，并叠加残差短接。
- **RAB&HDRAB.py**：残差注意块组合通道池化与空间注意，用于图像去噪。
- **RSSG.py**：RSSG 引入通道注意与状态空间卷积，用于图像恢复。
- **SWA.py**：SWA 空间权重注意力模块用于医学图像分割中的特征筛选。
- **TIF.py**：TIF 结构结合 Transformer 编码器与局部注意增强医学分割。
- **ULSAM.py**：ULSAM 以子空间注意构建极轻量注意力，用于嵌入式网络。
- **axial.py**：实现轴向注意力的 qkv 投影与行列解耦扫描。
- **scSE.py**：scSE 将通道与空间挤压-激励并联，用于细粒度注意力增强。

## 归一化与激活

- **(ICCV 2021)Crossnorm-Selfnorm(领域泛化).py**：实现 CrossNorm 与 SelfNorm 归一化策略以缓解分布偏移。
- **(ICLR 2023)ContraNorm(对比归一化层).py**：ContraNorm 通过对比约束调整归一化统计抑制过平滑。
- **(arxiv 2023)BCN.py**：BCN 提供批归一化的多变体及其在视觉 Transformer 中的适配。
- **(arxiv)Arelu.py**：自适应 ReLU 激活根据输入分段调整斜率，改善梯度流。

## 时序与频域建模

- **(AAAI 2019)AGCRN.py**：AGCRN 提供自适应图卷积与循环单元，用于交通预测时序建模。
- **TSConformerBlock.py**：TS-Conformer 模块结合前馈、卷积与自注意，面向语音增强。
- **cleegn.py**：CLEEGN 网络包含特征重排与卷积编码器，用于 EEG 重建。
- **f_sampling.py**：提供频域下采样与上采样单元，面向时频卷积网络。
- **phase_encoder.py**：PhaseEncoder 组合复数卷积与线性映射，编码音频相位。
- **tfcm.py**：TFCM 模块堆叠时频卷积与注意力，用于语音增强。

## 训练策略与损失

- **(CVPR 2020)CBDE.py**：复现 MoCo 的对比学习数据增强与编码器，支持自监督预训练。
- **(Elsevier 2024)CF_loss.py**：CF-Loss 提供针对视网膜多分类分割的临床友好损失计算。
- **LMFLoss.py**：实现 Focal、LDAM 与结合权重的 LMFLoss，多用于长尾分类。

## 任务特定/领域组件

- **(ACM MM 2023)Deepfake(深度伪造检测).py**：深度伪造检测网络的 CMCE 与 LFGA 模块，强调局部与全局特征融合。
- **(ECCV 2024)RCM语义分割.py**：RCM 语义分割块通过上下文重建与注意力提升高效分割表现。
- **(ICPR 2021)CAN(人群计数,CV2维任务通用).py**：CAN 模块为人群计数提供多尺度上下文编码与解码结构。
- **BFAM.py**：BFAM 面向遥感变化检测，融合 SIMAM 注意与多尺度卷积。
- **DPTAM.py**：DPTAM 结合双路 Transformer 注意力检测遥感变化。
- **PGM.py**：Prompt Generation Module 按任务生成提示向量，便于多任务调优。
- **TIAM.py**：时空交互注意模块在遥感变化检测中建模跨时间依赖。

## 工具与脚手架

- **特征维度转换.py**：提供二维与三维张量的尺寸转换工具，便于模块适配不同输入。
