# 顶层未分类模块梳理

本节将仓库根目录下未归入子文件夹的 108 个 Python 模块按功能归纳为六大类，列出文件名、主要类定义以及源码注释中提供的论文或用途说明，便于快速检索。

## 卷积与多尺度结构

| 文件 | 主要类 | 源注释摘要 |
| --- | --- | --- |
| (BMVC 2023)CoordGate.py | CoordGate 【F:(BMVC 2023)CoordGate.py†L5-L80】 | — |
| (CVPR 2019) DCNv2.py | DCNv2 【F:(CVPR 2019) DCNv2.py†L5-L142】 | — |
| (CVPR 2024)IDC.py | InceptionDWConv2d 【F:(CVPR 2024)IDC.py†L5-L47】 | — |
| (CVPR 2024)PKIBlock.py | ConvModule, InceptionBottleneck, CAA 【F:(CVPR 2024)PKIBlock.py†L11-L82】 | — |
| (CVPR2020)strip_pooling.py | StripPooling 【F:(CVPR2020)strip_pooling.py†L5-L86】 | ---------------------------------------；论文: Strip Pooling: Rethinking spatial pooling for scene parsing  (CVPR2020)；Github地址: https://github.com/houqb/SPNet；---------------------------------------【F:(CVPR2020)strip_pooling.py†L2-L4】 |
| DFF2d.py | DFF 【F:DFF2d.py†L5-L46】 | — |
| FEM.py | FEM, BasicConv 【F:FEM.py†L7-L55】 | — |
| FMS.py | Bconv, SppCSPC, ConvBNReLU 【F:FMS.py†L6-L70】 | — |
| GhostModule.py | GhostModule 【F:GhostModule.py†L5-L40】 | — |
| GlobalPMFSBlock.py | DepthWiseSeparateConvBlock, GlobalPMFSBlock_AP_Separate 【F:GlobalPMFSBlock.py†L9-L90】 | 论文：PMFSNet: Polarized Multi-scale Feature Self-attention Network For Lightweight Medical Image Segmentation；论文地址：https://arxiv.org/pdf/2401.07579；github地址：https://github.com/yykzjh/PMFSNet【F:GlobalPMFSBlock.py†L1-L3】 |
| HFF.py | LayerNorm, DropPath, Conv 【F:HFF.py†L7-L122】 | 论文：HiFuse: Hierarchical multi-scale feature fusion network for medical image classification；论文地址：https://www.sciencedirect.com/science/article/abs/pii/S1746809423009679【F:HFF.py†L4-L5】 |
| LDConv.py | LDConv 【F:LDConv.py†L5-L44】 | — |
| RFAConv.py | Conv, h_sigmoid, h_swish 【F:RFAConv.py†L7-L98】 | — |
| SPConv.py | SPConv_3x3 【F:SPConv.py†L5-L69】 | — |
| UIB.py | UniversalInvertedBottleneckBlock 【F:UIB.py†L7-L90】 | — |
| dynamic_conv.py | attention1d, Dynamic_conv1d, attention2d 【F:dynamic_conv.py†L7-L123】 | 论文：Dynamic Convolution: Attention over Convolution Kernels；论文地址：https://zhuanlan.zhihu.com/p/208519425【F:dynamic_conv.py†L5-L6】 |

## 工具/辅助

| 文件 | 主要类 | 源注释摘要 |
| --- | --- | --- |
| 特征维度转换.py |  【F:特征维度转换.py†L1-L9】 | — |

## 归一化/激活

| 文件 | 主要类 | 源注释摘要 |
| --- | --- | --- |
| (ICCV 2021)Crossnorm-Selfnorm(领域泛化).py | CrossNorm, SelfNorm, CNSN 【F:(ICCV 2021)Crossnorm-Selfnorm(领域泛化).py†L6-L80】 | 论文：CrossNorm and SelfNorm for Generalization under Distribution Shifts；论文地址；https://arxiv.org/pdf/2102.02811【F:(ICCV 2021)Crossnorm-Selfnorm(领域泛化).py†L6-L7】 |
| (ICLR 2023)ContraNorm(对比归一化层).py | ContraNorm 【F:(ICLR 2023)ContraNorm(对比归一化层).py†L6-L49】 | 论文：ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond；论文地址：https://ar5iv.labs.arxiv.org/html/2303.06562【F:(ICLR 2023)ContraNorm(对比归一化层).py†L3-L4】 |
| (arxiv 2023)BCN.py | BatchNorm2D, BatchNormm2D, BatchNormm2DViiT 【F:(arxiv 2023)BCN.py†L6-L88】 | — |
| (arxiv)Arelu.py | AReLU 【F:(arxiv)Arelu.py†L8-L18】 | — |

## 时序/频域建模

| 文件 | 主要类 | 源注释摘要 |
| --- | --- | --- |
| (AAAI 2019)AGCRN.py | AVWGCN, AGCRNCell, AVWDCRNN 【F:(AAAI 2019)AGCRN.py†L7-L51】 | 论文：Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting；论文地址：https://arxiv.org/pdf/2007.02842【F:(AAAI 2019)AGCRN.py†L5-L6】 |
| TSConformerBlock.py | FeedForwardModule, ConformerConvModule, AttentionModule 【F:TSConformerBlock.py†L10-L49】 | 论文：CMGAN: Conformer-Based Metric-GAN for Monaural Speech Enhancement；论文地址：https://arxiv.org/pdf/2209.11112v3【F:TSConformerBlock.py†L4-L6】 |
| cleegn.py | Permute2d, CLEEGN 【F:cleegn.py†L6-L50】 | 论文：CLEEGN: A Convolutional Neural Network for Plug-and-Play Automatic EEG Reconstruction；论文地址：https://arxiv.org/pdf/2210.05988v2.pdf【F:cleegn.py†L4-L5】 |
| f_sampling.py | FD, FU 【F:f_sampling.py†L6-L52】 | 论文：Multi-Scale Temporal Frequency Convolutional Network With Axial Attention for Speech Enhancement (ICASSP 2022)；论文地址：https://ieeexplore.ieee.org/document/9746610【F:f_sampling.py†L3-L4】 |
| phase_encoder.py | ComplexConv2d, ComplexLinearProjection, PhaseEncoder 【F:phase_encoder.py†L7-L92】 | 论文：Multi-Scale Temporal Frequency Convolutional Network With Axial Attention for Speech Enhancement (ICASSP 2022)；论文地址：https://ieeexplore.ieee.org/document/9746610【F:phase_encoder.py†L4-L5】 |
| tfcm.py | TFCM_Block, TFCM 【F:tfcm.py†L6-L70】 | 论文：Multi-Scale Temporal Frequency Convolutional Network With Axial Attention for Speech Enhancement(ICASSP 2022)；论文地址：https://ieeexplore.ieee.org/document/9746610【F:tfcm.py†L4-L5】 |

## 注意力/Transformer模块

| 文件 | 主要类 | 源注释摘要 |
| --- | --- | --- |
| (ACCV 2022)CSCA.py | FusionModel, Block, MSC 【F:(ACCV 2022)CSCA.py†L8-L182】 | 论文：Spatio-channel Attention Blocks for Cross-modal Crowd Counting；论文地址：https://arxiv.org/pdf/2210.10392【F:(ACCV 2022)CSCA.py†L6-L7】 |
| (ACCV 2024) LIA.py | SoftPooling2D, LocalAttention 【F:(ACCV 2024) LIA.py†L8-L40】 | 论文题目：PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution；论文地址：https://openaccess.thecvf.com/content/ACCV2024/papers/Wang_PlainUSR_Chasing_Faster_ConvNet_for_Efficient_Super-Resolution_ACCV_2024_paper.pdf【F:(ACCV 2024) LIA.py†L6-L7】 |
| (Arxiv2024)MDAF.py | BiasFree_LayerNorm, WithBias_LayerNorm, LayerNorm 【F:(Arxiv2024)MDAF.py†L15-L48】 | 论文地址 https://arxiv.org/pdf/2405.01992【F:(Arxiv2024)MDAF.py†L1-L1】 |
| (CVPR 2022) DAT.py | LayerNormProxy, DAttention 【F:(CVPR 2022) DAT.py†L10-L223】 | 论文题目：Efficient Attention: Attention with Linear Complexities；论文链接：https://openaccess.thecvf.com/content/CVPR2022/papers/Xia_Vision_Transformer_With_Deformable_Attention_CVPR_2022_paper.pdf【F:(CVPR 2022) DAT.py†L1-L2】 |
| (CVPR 2024)RAMiT.py | DropPath, QKVProjection, SpatialSelfAttention 【F:(CVPR 2024)RAMiT.py†L26-L53】 | 论文：Reciprocal Attention Mixing Transformer for Lightweight Image Restoration(CVPR 2024 Workshop)；论文地址：https://arxiv.org/abs/2305.11474【F:(CVPR 2024)RAMiT.py†L6-L9】 |
| (ECCV 2024)HTB.py | BiasFree_LayerNorm, WithBias_LayerNorm, LayerNorm 【F:(ECCV 2024)HTB.py†L24-L53】 | 论文：Restoring Images in Adverse Weather Conditions via Histogram Transformer (ECCV 2024)；论文地址：https://arxiv.org/pdf/2407.10172；全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules【F:(ECCV 2024)HTB.py†L1-L3】 |
| (ECCV2024)SMFA.py | DMlp, SMFA 【F:(ECCV2024)SMFA.py†L6-L49】 | 论文地址：https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Spatially-Adaptive_Feature_Modulation_for_Efficient_Image_Super-Resolution_ICCV_2023_paper.pdf【F:(ECCV2024)SMFA.py†L5-L5】 |
| (ICCV 2021) RA.py | ResidualAttention 【F:(ICCV 2021) RA.py†L11-L24】 | 论文：Residual Attention: A Simple but Effective Method for Multi-Label Recognition；论文地址：https://arxiv.org/pdf/2108.02456【F:(ICCV 2021) RA.py†L5-L6】 |
| (ICCV2023)SAFM.py | SAFM 【F:(ICCV2023)SAFM.py†L6-L39】 | 论文：https://arxiv.org/pdf/2302.13800【F:(ICCV2023)SAFM.py†L4-L5】 |
| (ICLR 2022) Crossformer.py | Mlp, DynamicPosBias, Attention 【F:(ICLR 2022) Crossformer.py†L12-L90】 | 论文：CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention (ICLR 2022 Acceptance).；论文地址：https://arxiv.org/pdf/2108.00154【F:(ICLR 2022) Crossformer.py†L6-L7】 |
| (ICLR 2022) MobileViTAttention.py | PreNorm, FeedForward, Attention 【F:(ICLR 2022) MobileViTAttention.py†L10-L93】 | 论文题目：MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer；论文链接：https://arxiv.org/pdf/2110.02178【F:(ICLR 2022) MobileViTAttention.py†L5-L6】 |
| (ICLR 2024)CA Block.py | ElementScale, ChannelAggregationFFN 【F:(ICLR 2024)CA Block.py†L22-L101】 | 论文：MogaNet: Multi-order Gated Aggregation Network (ICLR 2024)；论文地址：https://arxiv.org/pdf/2211.03295【F:(ICLR 2024)CA Block.py†L4-L6】 |
| (ICPR 2022) MOATransformer.py | Mlp, WindowAttention, GlobalAttention 【F:(ICPR 2022) MOATransformer.py†L12-L160】 | 论文：Aggregating Global Features into Local Vision Transformer；论文地址：https://arxiv.org/pdf/2201.12903【F:(ICPR 2022) MOATransformer.py†L7-L8】 |
| (NeurIPS 2021) CoAtNet.py | ScaledDotProductAttention, SwishImplementation, MemoryEfficientSwish 【F:(NeurIPS 2021) CoAtNet.py†L18-L104】 | 论文：CoAtNet: Marrying Convolution and Attention for All Data Sizes；论文地址：https://arxiv.org/pdf/2106.04803【F:(NeurIPS 2021) CoAtNet.py†L14-L15】 |
| (NeurIPS 2021) GFNet.py | PatchEmbed, GlobalFilter, Mlp 【F:(NeurIPS 2021) GFNet.py†L10-L83】 | 论文：Global Filter Networks for Image Classification；论文地址：https://arxiv.org/pdf/2107.00645【F:(NeurIPS 2021) GFNet.py†L6-L7】 |
| (TPAMI 2022) ViP.py | MLP, WeightedPermuteMLP 【F:(TPAMI 2022) ViP.py†L8-L52】 | 论文：Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition；论文地址：https://arxiv.org/pdf/2106.12368【F:(TPAMI 2022) ViP.py†L4-L5】 |
| (arXiv 2019) ECA.py | ECAAttention 【F:(arXiv 2019) ECA.py†L13-L41】 | 论文：ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks；论文地址：https://arxiv.org/pdf/1910.03151【F:(arXiv 2019) ECA.py†L7-L8】 |
| (arXiv 2020 ) SSAN.py | SimplifiedScaledDotProductAttention 【F:(arXiv 2020 ) SSAN.py†L10-L78】 | — |
| (arXiv 2021) AFT.py | AFT_FULL 【F:(arXiv 2021) AFT.py†L10-L57】 | — |
| (arXiv 2021) EA.py | ExternalAttention 【F:(arXiv 2021) EA.py†L10-L40】 | — |
| (arXiv 2021) MobileViTv2.py | MobileViTv2Attention 【F:(arXiv 2021) MobileViTv2.py†L10-L59】 | — |
| (arXiv 2021) PSA.py | PSA 【F:(arXiv 2021) PSA.py†L11-L70】 | — |
| (arXiv 2021) PSAN.py | ParallelPolarizedSelfAttention, SequentialPolarizedSelfAttention 【F:(arXiv 2021) PSAN.py†L10-L94】 | — |
| (arXiv 2021) S2Attention.py | SplitAttention, S2Attention 【F:(arXiv 2021) S2Attention.py†L28-L70】 | — |
| (arXiv 2023) ScaledDotProductAttention.py | ScaledDotProductAttention 【F:(arXiv 2023) ScaledDotProductAttention.py†L10-L78】 | 论文：Scaled Dot-Product Attention (SDPA)【F:(arXiv 2023) ScaledDotProductAttention.py†L4-L4】 |
| (arXiv 2024) MoHAttention.py | MoHAttention 【F:(arXiv 2024) MoHAttention.py†L12-L132】 | 论文：MoHAttention；论文地址：https://arxiv.org/pdf/2406.19510【F:(arXiv 2024) MoHAttention.py†L4-L5】 |
| CPAM.py | channel_att, local_att, CPAM 【F:CPAM.py†L7-L70】 | 论文：ASF-YOLO: A Novel YOLO Model with Attentional Scale Sequence Fusion for Cell Instance Segmentation(IMAVIS)；论文地址：https://arxiv.org/abs/2312.06458【F:CPAM.py†L4-L5】 |
| CPCA2d.py | ChannelAttention, CPCABlock 【F:CPCA2d.py†L9-L70】 | — |
| CRMSA.py | InnerAttention, CrossRegionAttntion 【F:CRMSA.py†L37-L249】 | — |
| DA.py | DoubleAttentionLayer 【F:DA.py†L6-L68】 | — |
| DASI.py | Bag, conv_block, DASI 【F:DASI.py†L8-L52】 | — |
| DA_Block.py | DepthWiseConv2d, PAM_Module, CAM_Module 【F:DA_Block.py†L7-L123】 | 论文：Dual Attention Network for Scene Segmentation（DANet）；论文地址：https://arxiv.org/abs/1809.02983【F:DA_Block.py†L5-L6】 |
| DICAM.py | Inc, Flatten, CAM 【F:DICAM.py†L5-L82】 | — |
| DSAM.py | DSAMBlock, cubic_attention, spatial_strip_att 【F:DSAM.py†L7-L132】 | — |
| EFF2d.py | SpatialAttention, Efficient_Attention_Gate, EfficientChannelAttention 【F:EFF2d.py†L7-L89】 | — |
| ENLTB.py | ENLA, BasicBlock, Mlp 【F:ENLTB.py†L98-L214】 | — |
| FADConv.py | OmniAttention, FrequencySelection, AdaptiveDilatedConv 【F:FADConv.py†L8-L328】 | 论文：Frequency-Adaptive Dilated Convolution for Semantic Segmentation[CVPR 2024]；论文地址：https://arxiv.org/abs/2403.05369【F:FADConv.py†L1-L2】 |
| FCA.py | Mix, FCAttention 【F:FCA.py†L8-L54】 | — |
| FCHiLo.py | PositionEmbedding, DSC, IDSC 【F:FCHiLo.py†L8-L91】 | — |
| FECAttention.py | dct_channel_block 【F:FECAttention.py†L56-L90】 | — |
| FFA.py | PALayer, CALayer, Block 【F:FFA.py†L9-L55】 | — |
| FMB.py | DMlp, PCFN, SMFA 【F:FMB.py†L7-L50】 | — |
| FMM.py | LayerNorm, CCM, SAFM 【F:FMM.py†L8-L66】 | — |
| GAB.py | LayerNorm, group_aggregation_bridge 【F:GAB.py†L6-L76】 | — |
| GAU.py | TA, SCA 【F:GAU.py†L4-L68】 | — |
| GCTattention.py | GCT 【F:GCTattention.py†L7-L34】 | — |
| GHPA.py | LayerNorm, Grouped_multi_axis_Hadamard_Product_Attention 【F:GHPA.py†L6-L97】 | — |
| GLSA.py | BasicConv2d, ContextBlock, ConvBranch 【F:GLSA.py†L6-L143】 | — |
| HAAM.py | Channelblock, Spatialblock, HAAM 【F:HAAM.py†L10-L147】 | — |
| HWAB.py | DWT, IWT, SALayer 【F:HWAB.py†L43-L170】 | — |
| LAE.py | Conv, LAE 【F:LAE.py†L16-L82】 | — |
| LGAG.py | LGAG 【F:LGAG.py†L26-L86】 | — |
| LPA.py | ChannelAttention, SpatialAttention, LPA 【F:LPA.py†L6-L64】 | — |
| MDTA.py | Attention 【F:MDTA.py†L7-L42】 | Multi-DConv Head Transposed Self-Attention (MDTA)【F:MDTA.py†L1-L1】 |
| MHIASA.py | EHIAAttention 【F:MHIASA.py†L10-L125】 | 论文：MHIAIFormer: Multi-Head Interacted and Adaptive Integrated Transformer with Spatial-Spectral Attention for Hyperspectral Image Classification；https://ieeexplore.ieee.org/abstract/document/10632582/【F:MHIASA.py†L6-L8】 |
| MLA.py | LayerNorm2d, ConvLayer, LiteMLA 【F:MLA.py†L36-L180】 | 论文：EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction；论文地址：https://arxiv.org/abs/2205.14756【F:MLA.py†L1-L2】 |
| MLAttention.py | MLAttention, FastMLAttention 【F:MLAttention.py†L7-L88】 | — |
| MixStructure.py | MixStructureBlock 【F:MixStructure.py†L6-L83】 | — |
| NAF.py | LayerNormFunction, LayerNorm2d, SimpleGate 【F:NAF.py†L9-L169】 | — |
| PCBAM.py | ChannelAttentionModule, SpatialAttentionModule, CBAM 【F:PCBAM.py†L5-L140】 | 论文：DAU-Net: Dual attention-aided U-Net for segmenting tumor in breast ultrasound images；论文地址：https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0303670【F:PCBAM.py†L3-L4】 |
| RAB&HDRAB.py | Basic, ChannelPool, SAB 【F:RAB&HDRAB.py†L6-L167】 | 论文：Dual Residual Attention Network for Image Denoising；论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0031320324000426【F:RAB&HDRAB.py†L3-L4】 |
| RSSG.py | ChannelAttention, CAB, SS2D 【F:RSSG.py†L16-L143】 | 论文：MambaIR: A simple baseline for image restoration with state-space model；论文地址：https://arxiv.org/pdf/2402.15648.pdf【F:RSSG.py†L3-L4】 |
| SWA.py | SWA 【F:SWA.py†L6-L59】 | 论文：DAU-Net: Dual attention-aided U-Net for segmenting tumor in breast ultrasound images；论文地址：https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0303670【F:SWA.py†L3-L4】 |
| TIF.py | PreNorm, FeedForward, Attention 【F:TIF.py†L7-L120】 | 论文：DS-TransUNet: Dual Swin Transformer U-Net for Medical Image Segmentation；论文地址：https://arxiv.org/abs/2106.06716【F:TIF.py†L3-L4】 |
| ULSAM.py | SubSpace, ULSAM 【F:ULSAM.py†L7-L91】 | ULSAM: Ultra-Lightweight Subspace Attention Module for Compact Convolutional Neural Networks(WACV20)【F:ULSAM.py†L4-L4】 |
| axial.py | qkv_transform, AxialAttention 【F:axial.py†L10-L129】 | — |
| scSE.py | cSE, sSE, scSE 【F:scSE.py†L6-L32】 | — |

## 训练策略/损失

| 文件 | 主要类 | 源注释摘要 |
| --- | --- | --- |
| (CVPR 2020)CBDE.py | MoCo, ResBlock, ResEncoder 【F:(CVPR 2020)CBDE.py†L7-L185】 | 论文：Momentum Contrast for Unsupervised Visual Representation Learning；论文地址：https://arxiv.org/pdf/1911.05722【F:(CVPR 2020)CBDE.py†L5-L6】 |
| (Elsevier 2024)CF_loss.py | CF_Loss_3D 【F:(Elsevier 2024)CF_loss.py†L12-L53】 | 论文：CF-Loss: Clinically-relevant feature optimised loss function for retinal multi-class vessel segmentation and vascular feature measurement【F:(Elsevier 2024)CF_loss.py†L5-L6】 |
| LMFLoss.py | FocalLoss, LDAMLoss, LMFLoss 【F:LMFLoss.py†L8-L77】 | — |

## 领域任务组件

| 文件 | 主要类 | 源注释摘要 |
| --- | --- | --- |
| (ACM MM 2023)Deepfake(深度伪造检测).py | CMCE, LFGA 【F:(ACM MM 2023)Deepfake(深度伪造检测).py†L9-L79】 | 论文：Locate and Verify: A Two-Stream Network for Improved Deepfake Detection；论文地址：https://arxiv.org/pdf/2309.11131【F:(ACM MM 2023)Deepfake(深度伪造检测).py†L6-L7】 |
| (ECCV 2024)RCM语义分割.py | ConvMlp, RCA, RCM 【F:(ECCV 2024)RCM语义分割.py†L8-L79】 | 论文：Context-Guided Spatial Feature Reconstruction for Efficient Semantic Segmentation[ECCV 2024]；论文地址：https://arxiv.org/pdf/2405.06228；全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules【F:(ECCV 2024)RCM语义分割.py†L1-L3】 |
| (ICPR 2021)CAN(人群计数,CV2维任务通用).py | ContextualModule 【F:(ICPR 2021)CAN(人群计数,CV2维任务通用).py†L11-L47】 | 论文：Encoder-Decoder Based Convolutional Neural Networks with Multi-Scale-Aware Modules for Crowd Counting；论文地址：https://ieeexplore.ieee.org/document/9413286【F:(ICPR 2021)CAN(人群计数,CV2维任务通用).py†L8-L9】 |
| AFPN.py | BasicBlock, Upsample, Downsample_x2 【F:AFPN.py†L20-L66】 | 论文：AFPN: Asymptotic Feature Pyramid Network for Object Detection；论文地址：https://arxiv.org/pdf/2306.15988【F:AFPN.py†L6-L7】 |
| BFAM.py | simam_module, BFAM 【F:BFAM.py†L7-L79】 | 论文：B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection；论文地址：https://ieeexplore.ieee.org/document/10547405【F:BFAM.py†L1-L2】 |
| DPTAM.py | DPTAM 【F:DPTAM.py†L8-L136】 | 论文：Dual Parallel Transformer Attention Mechanism for Change Detection；论文地址：https://doi.org/10.1016/j.eswa.2024.124939【F:DPTAM.py†L4-L5】 |
| MCM.py | MCM 【F:MCM.py†L10-L99】 | 论文：MAGNet: Multi-scale Awareness and Global fusion Network for RGB-D salient object detection | KBS；论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0950705124007603；github地址：https://github.com/mingyu6346/MAGNet【F:MCM.py†L5-L7】 |
| MDCR.py | conv_block, MDCR 【F:MDCR.py†L7-L125】 | 论文：HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection；论文地址：https://arxiv.org/pdf/2403.10778v1.pdf【F:MDCR.py†L5-L6】 |
| PGM.py | PromptGenBlock 【F:PGM.py†L6-L38】 | — |
| TIAM.py | SpatiotemporalAttentionFull, SpatiotemporalAttentionBase, SpatiotemporalAttentionFullNotWeightShared 【F:TIAM.py†L7-L144】 | 论文：Robust change detection for remote sensing images based on temporospatial interactive attention module；论文地址：https://www.sciencedirect.com/science/article/pii/S1569843224001213【F:TIAM.py†L3-L4】 |
| UCDC.py | UCDC 【F:UCDC.py†L17-L69】 | 论文：ABC: Attention with Bilinear Correlation for Infrared Small Target Detection ICME2023；论文地址：https://arxiv.org/pdf/2303.10321【F:UCDC.py†L4-L5】 |
