# 面向工程代理建模流程的可借鉴研究图谱

## 执行摘要

如果把 NeuralFoil 真正“借到手”，最值得复制的并不是某个具体 MLP 或某个单独 trick，而是一整套**把物理结构提前写进模型**的方法论：先把几何、工况、周期性、极限行为、已知解析关系、可信度边界这些“问题结构”编码进输入、输出、损失和后处理，再让模型只学习剩余的复杂部分。NeuralFoil 本身就是这种路线的代表：它把空气动力分析做成了**物理启发特征编码 + 平滑可微网络 + 解析修正/极限外推 + 自报告可信度**的混合系统，并且用 Kulfan/CST 低维参数化、周期角度编码、平滑 Swish 激活、Mahalanobis 距离修正的 analysis_confidence 形成一套完整工程工作流。对你的 MISES_psd_density 任务，最直接的映射不是“照搬 airfoil 输入”，而是把“频率—模态—局部几何—边界/载荷—对称性—解析基线”都变成结构化输入，再让模型做残差学习。 citeturn6search7turn29view1turn31view2turn29view4turn30view0

过去五年里，最值得借鉴的相关工作大致分成五类。第一类是**几何与坐标结构建模**，代表有 Geo-FNO、PI-GANO、GINOT、Transolver、MARIO，它们共同解决“任意几何/点云/非结构网格如何稳定建模”的问题。第二类是**图与算子学习**，代表有 MeshGraphNets、X-MeshGraphNet、FNO、DeepONet/AE-DeepONet，解决“把输入从标量回归升级为函数—函数映射”这一层级跃迁。第三类是**混合解析+学习/基线+残差**，代表有 NeuralFoil、本征/模态降阶残差修正、局部非线性结构保持框架。第四类是**不确定性/OOD/可信部署**，代表有 NeuralFoil 的 analysis_confidence、基于 DDPM 的 airfoil flow UQ 以及 OPAL-surrogate。第五类是**多保真与主动补样**，代表有 2026 年的主动多保真 airfoil surrogate 优化框架。这个分类本身已经给出了对你任务的主判断：**最先应该优化的不是“大模型”，而是输入编码、基线拆分、频谱损失、OOD 机制和局部几何参数化。** citeturn24view0turn24view7turn24view8turn34view3turn7search5turn25view3turn10search0turn26view8turn28view0turn28view4turn26view0turn27view0turn6search3

对“节点/单元级频域响应或应力 PSD surrogate，且带模态/FRF 信息”的任务，我的结论是：短期内**最有性价比**的路线不是一步到位上 geometry-aware transformer，而是先做四件事：其一，把频率从单标量变成**模态失谐量、相对频率、log 频率、峰值邻域索引**；其二，把输出从“直接预测绝对 PSD”改成**解析/模态基线 + 学习残差**；其三，把损失从单一 MSE 改成**log-PSD、峰值窗口、谱形状/STFT 或 PSD loss、排序/峰位约束**；其四，外挂**confidence / ensemble / conformal** 之类的可部署可信度机制。只有当这些做完后，geometry-aware operator 或 mesh-based model 才真正值得上。这个判断，与 NeuralFoil、结构动力学 AE-DeepONet、模态降阶残差修正和结构保持非线性残差学习几条路线是高度一致的。 citeturn31view2turn29view4turn26view8turn28view3turn27view4turn28view4turn27view6

## 筛选范围与评估框架

这份报告优先筛选了近十年、尤其近五年的**原始论文或官方项目**，并尽量选择了同时具备论文与官方代码/文档/数据的工作。检索重点覆盖了六个方面：物理启发 ML、空气动力/结构代理模型、混合解析+ML、低维几何参数化、UQ/OOD、模态/FRF/频谱感知建模。对每项工作的判断主要看六个维度：是否适用于**任意几何或非结构网格**、是否能容纳**函数型输入**、是否支持**基线+残差**、是否具备**可信度/OOD**机制、是否有**可复现实装**，以及对你当前这类**频域应力/PSD surrogate**是否能直接映射。 citeturn23view0turn23view2turn23view6turn23view5turn24view6turn24view7turn26view8turn32view0

从“像不像 NeuralFoil、但又值得工程借鉴”的角度，我把候选分成四簇。第一簇是**NeuralFoil 周边与 benchmark 基础设施**，它们告诉你怎样搭建可训练、可比较、可部署的数据与评测框架；第二簇是**几何感知算子/图模型**，它们决定未来如果你从 tabular 特征回归走向 mesh/point-cloud surrogate，应选什么技术；第三簇是**不确定性、多保真和可信部署**，它们决定模型何时能在工程里“放心用”；第四簇是**结构动力学与模态/频域专门路线**，这是与你任务最贴近的一簇。后文的优先级推荐，也将按“与你当前任务的直接映射强度”排序，而不是按论文热度排序。 citeturn32view1turn36view0turn24view0turn25view0turn24view7turn26view0turn27view0turn28view0

## 候选论文与项目详注

下面这份 annotated list 按“从更直接可借鉴到更偏体系/升级路线”的顺序组织。每条后面的引文同时充当论文或代码入口链接。

1. **NeuralFoil**。这是最值得逐行研究的标杆工作。它把 airfoil 几何压成 18 维 Kulfan/CST 参数，把攻角做周期编码，把可微平滑性作为架构约束，把 analysis_confidence 作为自报告可信度，再用解析后处理处理压缩性与失速极限，形成“结构化输入—平滑主干—可信度/OOD—解析融合”的完整闭环。对你的任务，最直接的借鉴是：**频率做结构化编码、模态极限/低频高频极限做后处理、输出附带可信度、训练目标服务于优化与部署而不是只服务于均方误差**。论文与代码都很成熟。 citeturn6search7turn29view1turn31view2turn29view4turn30view0turn23view0

2. **AirfRANS**。它本身不是模型，而是工程 surrogate 训练/评测时应有的数据组织范式：1000 个 RANS 仿真、明确的 Re 与攻角范围、四类任务、标准化访问方式 `pip install airfrans`。对你最有价值的不是 airfoil 本身，而是它示范了**统一数据/任务拆分、公开 benchmark、分任务评价**，这比单纯堆模型更接近真正可迭代的工程工作流。 citeturn3search0turn23view2

3. **NeurIPS 2024 ML4CFD Competition 与 Starting Kit**。这套体系把 surrogate 的评估从“只看 MAE/MSE”提升到**精度、物理一致性、效率、OOD**四维联合评分，并且通过 LIPS/Starting Kit 交付 notebook、数据与提交规范。对你的任务，这意味着你也应放弃“单指标评估”，转向**点值误差 + 峰位/峰高误差 + 全谱形状误差 + OOD 识别 + 推理成本**的联合选择逻辑。 citeturn32view1turn32view0turn23view9

4. **AFBench**。虽然其任务是 airfoil inverse design，但它最值得借鉴的是**低维几何参数化 + 多模态条件控制 + benchmark 化编辑任务**。AFBench 提供 20 万 airfoil、11 个几何参数、66 个工况，且把 generation/editing 任务做成了可训练可评估问题。对你的任务，最可借鉴的是：把“局部几何—模态—频率—边界条件”组织成**多条件 controllable surrogate**，而不是把所有信息扔给单一黑箱回归器。 citeturn36view0turn36view1turn36view2

5. **Learning Mesh-Based Simulation with Graph Networks**。MeshGraphNets 的意义在于把物理场从规则栅格解放到网格图上，并在 aerodynamics、structural mechanics 等场景证明了基于图的信息传递可以学习分辨率无关的动力学。对你而言，它最有价值的不是“马上上 GNN”，而是提醒你：**如果未来要直接吃 FE 网格或节点邻接关系，图结构是自然升级方向**，尤其适合热点附近强局部耦合的应力/PSD 场。 citeturn7search5turn17search8turn24view5

6. **X-MeshGraphNet**。这是更工程化的 MeshGraphNet 续作，核心贡献是大图分块训练、halo 区域通信、直接从 STL/几何点云构图、多尺度点云层级，解决了大规模图模拟的可扩展性与推理时对仿真网格的依赖。对你的任务，这一思路非常有借鉴价值：如果你以后要把局部热点的邻域关系显式建模，**“从几何/表面点云构图”比“完全绑定某个 FE 网格拓扑”更有迁移性**。 citeturn17search1turn25view3turn25view4

7. **Fourier Neural Operator 与 NeuralOperator 官方库**。FNO 的核心不是某种固定网络，而是把映射提升为**函数空间到函数空间**的学习；官方库进一步把 FNO、TFNO 等做成了可复用工具链，并强调分辨率不变性。对你的任务，如果你把“频率轴上的 FRF/PSD 曲线”看作函数，把“几何/模态/边界条件”看作条件输入，那么 operator learning 比普通 tabular MLP 更自然。尤其是当输出不是某一个频点而是整条谱线时，这类思想会明显优于逐频点独立回归。 citeturn10search0turn23view6turn33view1

8. **Geo-FNO**。Geo-FNO 把 FNO 扩展到任意几何：先学习把物理域变形到潜在规则网格，再在潜在网格上做 FFT。它专门解决“FNO 只能处理规则矩形域”的痛点，并公开了 airfoil flow 数据与代码。对你的任务，如果你最终需要**在不同结构外形、不同局部曲率、不同边界上预测频域响应场**，Geo-FNO 是最值得模仿的一条几何感知算子路线。 citeturn24view8turn12search10turn33view4

9. **PI-GANO**。这项工作把 geometry-aware neural operator 与 physics-informed training 结合起来，强调**不必依赖大量 FEM 监督数据**也能同时泛化到 PDE 参数和几何变化；官方代码还给出了 plate stress 这类固体力学任务。对你的任务，PI-GANO 的启示有两层：一是**几何编码器**本身值得借；二是当你高保真标注稀缺时，可以考虑在残差头或辅助任务里引入**物理一致性约束**，而不是完全纯监督。 citeturn25view0turn24view6

10. **GINOT**。GINOT 用点云几何编码器 + attention + neural operator decoder 处理任意 2D/3D 几何，而且明确不依赖 SDF 作为唯一几何表达。对你的任务尤其重要，因为结构仿真常常已经有表面/体点云或节点坐标，不一定愿意再生成 SDF。GINOT 最值得借鉴的是：**把几何编码和查询点求解解耦**，即先编码整体几何，再对每个节点/单元位置做 query 式预测，这与你的“每个网格点一个 MISES_psd_density”任务形式高度一致。 citeturn24view7turn24view3

11. **Transolver**。Transolver 的 Physics-Attention 会把网格点按“内在物理状态”自适应分片，再在切片级别做注意力；而且官方 PhysicsNeMo 文档明确支持 structured 与 unstructured mesh。对你的任务，它的现实价值在于：如果将来发现局部热点与远场边界之间存在长程依赖，普通局部 MLP/GNN 可能不够，**Transolver 这类切片注意力**会比全局 dense attention 更现实。 citeturn24view0turn35view0turn23view4

12. **MARIO**。MARIO 是条件 Neural Field 架构，强调大尺度 PDE surrogate、参数化与非参数化几何变化，以及 SDF 编码 + multiscale Fourier features + hypernetwork modulation；它已在 AirfRANS 与 PLAID 场景做了公开实现。对于你的任务，MARIO 特别值得借的是**“坐标查询 + 条件调制”**这条路线：不要把节点 ID 当特征，而是让模型学会在给定局部坐标、全局条件、几何潜变量时输出场值，这非常适合非统一网格、不同模型尺寸和未来形状迁移。 citeturn18search6turn34view3turn34view1

13. **Uncertainty-aware Surrogate Models for Airfoil Flow Simulations with DDPMs**。这项工作不是追求单一均值预测，而是直接学习**解分布**，并与 BNN、heteroscedastic baseline 做比较，解决 surrogate 的不确定性估计问题。对工程部署而言，这非常关键：如果你的 MISES_psd_density 在模态峰附近天生不稳定、对局部几何/阻尼极敏感，那么只给单值预测并不够，最好给**可信区间、方差或样本分布**。哪怕你不直接上 diffusion，也应该借它的评测与校准思想。 citeturn20search0turn26view0

14. **Optimization-Embedded Active Multi-Fidelity Surrogate Learning for Multi-Condition Airfoil Shape Optimization**。这篇 2026 工作把低保真 panel method 与高保真 RANS 结合，通过不确定性触发高保真采样做主动多保真优化。对你的任务，这对应的是：先用便宜的解析/模态近似、简化壳/梁模型或粗网格结果提供 baseline，再只在**峰值附近、热点区域、几何边界样本**上调用高保真数据做增量学习。多保真不是“可有可无”，而是频域任务里最经济的补样方式之一。 citeturn6search3

15. **OPAL-surrogate**。这篇工作关注的不是再发明一个网络，而是**如何在大量候选网络和超参数中找到“可信”的 surrogate**，其框架基于 hierarchical Bayesian inference 与 model validation tests。对你的任务，OPAL 最重要的借鉴是：不要把模型选择仅仅交给验证集 MSE，而应把**复杂度、误差、置信度、校准质量**一起纳入选择，这对工程 surrogate 的上线决策非常关键。 citeturn13search0turn27view0

16. **Deep learning-based predictive modelling of transonic flow over an aerofoil**。这篇工作最有价值的点，在于作者不满足于“预测得准”，而是进一步利用模型可微性去做**global instability analysis**，把神经网络与传统模态分析衔接起来。对你的任务，这是非常强的启发：未来你的 surrogate 不应只输出 MISES_psd_density，还应支持**对几何、阻尼、模态频率、载荷幅值的灵敏度分析**，这样模型才真正成为工程分析工具，而不是只会回归的近似器。 citeturn25view5turn25view6

17. **Parameter Estimation of Structural Dynamics with Neural Operators Enabled Surrogate Modeling**。这是与你任务最贴近的工作之一。它把 excitation force 视为函数输入，把系统参数与时间点一起输入 AE-DeepONet，完成 forward surrogate 与 inverse parameter estimation 两个任务。对你的任务，最直接的可借鉴点是：**把频率响应/载荷谱/基底激励看作函数，不要只把频率点当孤立标量；把模态参数与频率轴作为 operator 的条件输入。** 这比传统“拼一串标量特征给 MLP”更符合结构动力学本质。 citeturn26view8turn28view2turn28view3

18. **Machine Learning Assisted State Prediction of Misspecified Linear Dynamical System via Modal Reduction**。这篇 2026 工作几乎就是为“模态信息该怎么用”给出的答案：先把 FE 系统投影到 reduced modal basis，再用 GPLFM 在模态域表示 discrepancy，用 mesh-invariant neural network 把模态状态映射到残差修正，并联合处理 epistemic 与 aleatoric uncertainty。对你的任务，它直接给出一个强策略：**不要在物理空间直接“硬吃全谱应力”，先在模态域/降阶域做主建模，再把残差抬回物理空间。** citeturn28view0turn27view4turn27view5

19. **A Structure-Preserving Machine Learning Framework for Accurate Prediction of Structural Dynamics for Systems with Isolated Nonlinearities**。这篇工作提出把复杂局部非线性从整体线性系统里分离出来，只用 MLP 预测局部边界上的 deviatoric force，再由理想线性系统恢复全局响应。对你的任务，这几乎就是“基线+残差”的结构力学版本：**能用模态叠加、FRF、局部解析近似解释的部分，先用物理基线算；把基线解释不了的局部非线性/热点放大效应交给网络。** 这是比端到端回归更稳、更省样本的一条路。 citeturn26view9turn28view4

20. **Fourier Neural Operators for Structural Dynamics Models with Spectrogram Loss**。这篇工作非常值得你专门读，因为它直接讨论了 FNO 在结构动力学中的局限，并展示了 spectrogram loss 对频谱保真与能量保持的帮助，尤其在非线性、小样本和高频成分上更有价值。对于你的任务，这对应的是：如果你输出的是 PSD/FRF 或与频率结构强相关的量，**损失就应该显式惩罚谱形状错误，而不只是点值误差**。 citeturn26view6turn27view6

## 横向对比与快选表

| 名称 | 年份 | 类型 | 核心想法 | 模型类型 | 可借鉴点 | 成熟度/可用性 |
|---|---:|---|---|---|---|---|
| NeuralFoil citeturn6search7turn23view0turn29view4 | 2025 | 论文+代码 | 物理编码、平滑 MLP、解析修正、可信度输出 | ML+解析 | 周期/极限编码、C∞ 平滑、OOD confidence、基线融合 | 高，论文+GitHub+PyPI |
| AirfRANS citeturn3search0turn23view2 | 2022 | 数据集+代码 | 标准化 airfoil RANS benchmark | 基础设施 | 任务拆分、统一数据接口、标准评测 | 高，论文+库+文档 |
| ML4CFD Competition + Starting Kit citeturn32view1turn32view0turn23view9 | 2024–2025 | 论文+基线 | 多指标评测 surrogate | 基础设施 | 精度/物理/效率/OOD 联评 | 高，论文+starting kit |
| AFBench citeturn36view0turn36view2 | 2024 | 论文+代码+数据 | 多条件 airfoil generation/editing benchmark | ML | 多模态条件控制、低维几何标签 | 中高，代码+数据可用 |
| MeshGraphNets citeturn7search5turn17search8 | 2020 | 论文 | 网格图上的物理场学习 | ML | FE/CFD 网格自然表达、邻域耦合 | 高，论文成熟 |
| X-MeshGraphNet citeturn17search1turn25view3 | 2024 | 论文 | 大图分块、多尺度、STL/点云构图 | ML | 从几何直接构图、可扩展 mesh surrogate | 中高，论文+PhysicsNeMo 生态 |
| FNO / NeuralOperator citeturn10search0turn23view6turn33view1 | 2020–2025 | 论文+官方库 | 学习函数到函数映射 | ML | 整条谱线/场输出、分辨率不变、现成库 | 很高，官方库成熟 |
| Geo-FNO citeturn24view8turn12search10 | 2023 | 论文+代码 | 几何变形到潜在规则网格再做 FNO | ML | 任意几何上的 operator learning | 中高，官方代码有但旧 repo 已弃用 |
| PI-GANO citeturn25view0turn24view6 | 2024/2025 | 论文+代码 | geometry-aware + physics-informed neural operator | ML+物理约束 | 少监督/弱监督、plate stress 任务 | 中高，官方代码可跑 |
| GINOT citeturn24view7turn24view3 | 2025/2026 | 论文+代码 | 点云几何编码 + attention + operator decoder | ML | 节点查询式预测、无需 SDF | 中，代码有，较新 |
| Transolver citeturn24view0turn35view0 | 2024 | 论文+代码 | Physics-Attention 切片求解一般几何 PDE | ML | 长程依赖、structured/unstructured 通吃 | 高，论文+THU 代码+PhysicsNeMo 接入 |
| MARIO citeturn18search6turn34view3turn34view1 | 2025 | 论文+代码 | 条件 neural field + multiscale Fourier features + modulation | ML | 坐标查询、SDF 潜变量、跨几何场预测 | 中高，官方代码有 |
| DDPM UQ for Airfoil Flows citeturn20search0turn26view0 | 2023/2024 | 论文+代码 | 直接学习解分布而非均值 | ML | 峰区不确定性、全分布 UQ | 中高，官方实现公开 |
| Active Multi-Fidelity Airfoil Surrogate citeturn6search3 | 2026 | 论文 | LF/HF 融合 + 不确定性触发采样 | ML+解析/多保真 | 主动补样、局部高保真 refinement | 中，论文新，未见官方代码 |
| OPAL-surrogate citeturn13search0turn27view0 | 2024 | 论文 | 分层 Bayes + validation tests 选“可信代理” | 方法框架 | 模型选择、校准/可信度治理 | 中，偏框架论文 |
| Transonic Flow + Global Instability Analysis citeturn25view5turn25view6 | 2024 | 论文 | 用可微网络连接流场预测与模态/稳定性分析 | ML | surrogate 之外做灵敏度/模态分析 | 中，论文为主 |
| AE-DeepONet for Structural Dynamics citeturn26view8turn28view3 | 2024 | 论文 | excitation 函数 + 系统参数 → 动态响应 | ML | 将载荷/频谱视为函数输入 | 中高，论文细节充分 |
| Modal Reduction + GPLFM Discrepancy citeturn28view0turn27view4 | 2026 | 论文 | 模态降阶域上建 discrepancy + UQ | ML+解析/贝叶斯 | 模态域主建模、mesh-invariant residual | 中，较新、偏研究型 |
| Structure-Preserving Nonlinearity Residual Model citeturn26view9turn28view4 | 2024 | 论文 | 线性系统 + 局部非线性边界 traction 残差 | ML+解析 | 基线+残差、局部热点建模 | 中，论文清晰，未见官方代码 |
| FNO for Structural Dynamics with Spectrogram Loss citeturn26view6turn27view6 | 2025 | 论文 | 频谱损失改善谱保真与能量保持 | ML | PSD/STFT loss、频谱保真训练 | 中，论文新，偏方法验证 |

这张表的“快选”结论很明确：如果你要**马上落地**，优先读 NeuralFoil、AE-DeepONet、Modal Reduction、Structure-Preserving Residual 这四类；如果你要**准备下一代几何迁移模型**，优先读 GINOT、Geo-FNO、Transolver、MARIO；如果你要**解决上线可信度**，优先补 DDPM UQ 与 OPAL；如果你要**做数据效率提升**，主动多保真路线最值得试。 citeturn23view0turn26view8turn28view0turn28view4turn24view7turn24view8turn24view0turn34view3turn26view0turn27view0turn6search3

## 最值得深挖的方向与可直接借鉴的做法

我建议你优先深挖下面五个方向，因为它们对“频域应力/PSD surrogate + 模态特征”最能直接转化为实验。

**NeuralFoil。** 需要学的不是 airfoil 细节，而是它的工程哲学：把输入空间改写成“更像物理”的潜空间，把不连续/非唯一/极端外推留给解析修正与 confidence 机制去处理。对你的直接做法是：把频率换成 `log f`、相对模态频率 `f/f_n`、失谐量 `Δ_n=(f-f_n)/(ζ_n f_n)`、最近若干模态的 pole-distance、参与系数强度、累计模态质量分数等结构化特征；把局部坐标换成**全局归一化坐标 + 局部曲率/厚度/到边界距离/到约束距离/到载荷中心距离**；把几何和频率的周期/对称/极限行为写进编码。若你已经知道某些极限下 PSD 必须衰减、某些对称面上响应应满足特定关系，这些都该进模型，而不该让模型自己“悟”。这正是 NeuralFoil 最值钱的地方。 citeturn31view2turn29view1turn30view0turn29view4

**GINOT。** 你的任务如果最终想跨不同形状、不同 FE 网格、不同节点数泛化，GINOT 的 query-style 思路非常关键。建议直接借它的“几何编码器—查询解码器”范式：先用点云/表面网格编码整体结构，再对每个需要预测的节点输入局部坐标、局部法向/主曲率、材料/厚度、边界标签和模态条件做 query。这样你就不用绑死在某个固定节点拓扑上，未来换模型或换网格，只要 query 点与几何编码仍可定义，模型就能迁移。 citeturn24view7turn24view3

**AE-DeepONet 结构动力学路线。** 这项工作最重要的启发，是把“系统参数 + 激励函数 → 响应函数”当成主问题。对你的任务，我建议做两个头：一个头学**`(geometry, modal, BC, load PSD/FRF summary) -> 整段频谱`**；另一个头学**给定单频率 query 的点值预测**。前者保证整体谱形状，后者保证局部精度。换句话说，不要只把频率当作普通 feature，而应该把**频域输入看成 operator condition**。如果短期不想直接上 DeepONet，也至少要模仿它的 branch/trunk 分工：branch 负责载荷/激励函数，trunk 负责频率与空间 query。 citeturn26view8turn28view2turn28view3

**模态降阶残差修正。** 这是一条非常适合你任务的主线。我更建议你把网络的第一职责定义成：在 reduced modal domain 中预测主响应或 discrepancy，然后再映射回物理空间，而不是让网络直接从杂乱特征回归最终 MISES_psd_density。一个很可行的实验是：先用模态叠加/近似 FRF 做 baseline，得到节点级初始 PSD 估计；再让网络只预测 `log(true PSD) - log(baseline PSD)` 的残差。这样网络会自然把容量集中到热点放大、局部边界效应、阻尼失配、模态耦合和高频尾部这些“基线漏掉”的部分。对 sample efficiency、可解释性和 OOD 稳定性，这通常比端到端绝对值回归更好。 citeturn28view0turn27view4turn27view5turn28view4

**结构保持与可信部署。** 这里我建议把两条线合并吸收：一条来自局部非线性 structure-preserving residual model，另一条来自 NeuralFoil/DDPM/OPAL 的可信度思想。你的实际落地做法可以是：保留一个物理 baseline；残差用小网络学；训练时用 deep ensemble 或至少 3–5 个种子模型；推理时输出均值、方差，以及一个基于 latent Mahalanobis 或 conformal calibration 的 confidence 分数；模型选择时把误差、峰区误差、校准误差和复杂度联合考虑，而不是只看 RMSE。这比简单地在最后加一个 dropout 更接近工程可上线方案。 citeturn28view4turn29view4turn26view0turn27view0

除了前五，我还建议并行阅读两篇“配套文献”。一篇是**频谱损失的 FNO 结构动力学论文**，它会直接帮助你重新设计 loss；另一篇是**主动多保真 airfoil surrogate**，它会帮助你设计“哪里最值得补高保真样本”的采样策略。前者解决“怎么学频谱”，后者解决“怎么省数据”。 citeturn27view6turn6search3

## 实验路线图与实施建议

```mermaid
flowchart LR
A[现有数据与解析基线] --> B[物理结构化编码]
B --> C[基线 + 残差双路模型]
C --> D[频谱感知损失]
D --> E[可信度与OOD校准]
E --> F[主动多保真补样]
F --> G[几何感知升级版模型]
```

我建议按下面这条优先顺序推进，而不是一开始就切换到最复杂的 geometry-aware operator。因为你的问题最可能的瓶颈，通常先在**表征与目标函数**，而不是先在**主干网络深度**。这个顺序也与上面几类工作给出的证据一致。 citeturn31view2turn28view0turn27view6turn26view0turn6search3

| 建议步骤 | 具体动作 | 预期影响 | 实施代价 | 主要灵感来源 |
|---|---|---|---|---|
| 物理编码优先 | 把 `f` 扩成 `log f`、最近 `k` 个模态 `f/f_n`、失谐量、阻尼归一化距离、峰区标记；把坐标扩成归一化位置、到边界/约束/载荷/热点候选位置的距离；加入对称标签与局部几何参数 | 很高 | 低 | NeuralFoil、模态降阶残差 citeturn31view2turn29view1turn28view0 |
| 基线+残差重构 | 用模态叠加/FRF/经验谱估计做 baseline，网络只学 `log-residual`；输出拆成“全谱背景 + 峰值修正 + 局部热点修正” | 很高 | 中 | NeuralFoil、Structure-Preserving、Modal Reduction citeturn23view0turn28view4turn27view4 |
| 频谱感知损失 | 组合 `MSE(log PSD)`、峰区加权、峰位/峰高误差、STFT/PSD shape loss；必要时做多头 loss | 很高 | 中 | Structural FNO spectrogram loss、AE-DeepONet citeturn27view6turn28view3 |
| 可信度与 OOD | 加一个 confidence head；做 latent Mahalanobis、ensemble 方差或 conformal calibration；模型选择看误差+校准 | 高 | 中 | NeuralFoil、DDPM UQ、OPAL citeturn29view4turn26view0turn27view0 |
| 多保真与主动补样 | 先用便宜基线筛出峰区、热点区、几何极端样本，再追加高保真 FE/试验标注 | 高 | 中高 | Active Multi-Fidelity、ML4CFD 评测思想 citeturn6search3turn32view1 |
| 几何感知升级 | 若跨网格/跨几何迁移确实成为瓶颈，再尝试 GINOT/Geo-FNO/Transolver/MARIO | 中到很高 | 高 | GINOT、Geo-FNO、Transolver、MARIO citeturn24view7turn24view8turn24view0turn34view3 |

把这个路线翻成更具体的实验，我建议先做三组 A/B test。第一组，只改输入编码，不改主干，验证物理编码本身能带来多少收益。第二组，在相同主干下比较“绝对值回归”与“baseline+residual”。第三组，在同一网络下比较普通 MSE 与频谱感知 loss。通常如果这三组实验都成立，你再决定是否值得上更复杂的 geometry-aware model；如果这三组都不成立，那么换更大网络往往也帮不了太多。这个策略的优点是：每一步都可归因，而不是一次改十个东西后不知道什么真正起作用。 citeturn31view2turn28view4turn27view6

## 开源实现说明与开放问题

从落地角度看，下列开源项目最值得直接 fork 或至少抄其数据组织方式。**NeuralFoil** 提供了从 `xxsmall` 到 `xxxlarge` 的 8 档 MLP 尺寸，隐藏层/宽度从 `1×48` 到 `5×512`，训练使用 7.5M XFoil 样本，运行端是纯 Python/NumPy，且可经由 AeroSandbox 直接进入优化工作流；这说明**并不一定需要庞大 backbone 才能做出非常强的工程 surrogate，关键在输入输出与后处理结构。** citeturn31view2turn29view5turn23view1

**NeuralOperator** 适合你未来尝试 FNO/TFNO/GINO 一类 operator 模型。官方 quickstart 已给出 `FNO(n_modes=(64,64), hidden_channels=64, in_channels=2, out_channels=1)` 这样的最小骨架，也提供了 Tensorized FNO 降参数路线。若你要从“逐频点回归”转向“整段谱线或整片场输出”，这是现成的基础设施。 citeturn33view1turn23view6

**PhysicsNeMo** 则更适合做大规模实验与工程部署。它含 benchmarks、examples 与容器化环境；官方仓库明确给出 `examples` 目录和容器镜像，文档里还把 Transolver、MeshGraphNet 等模型参数化到了可直接调用的 API。若你预计后期要做 mesh surrogate、点云 surrogate 或更重的 GPU 训练，这个生态比零搭环境省事得多。 citeturn23view5turn35view2turn35view3turn35view0

**PI-GANO** 和 **MARIO** 都很适合研究“几何潜变量如何进模型”。PI-GANO 需要单独下载数据与预训练模型，Python 3.8+ 即可复现；MARIO 则是典型两阶段流程：先训练 `train_sdf.py` 得到几何 latent，再用 `train.py` 训练 flow surrogate，几何调制以 `.npz` latent 的形式喂入主模型。这两种实现都很值得借来改造你的数据预处理：你完全可以把“几何 latent”换成**模态 latent**，或者把“几何 latent + 模态 latent”一起作为条件调制项。 citeturn24view6turn34view2turn34view3

**AFBench** 与 **AirfRANS** 给出了两个很好的数据工程模板。AFBench 的数据最终组织为 `.dat` 形状文件、split 文件与标签文本，支持 `full/scarce/reynolds/aoa` 等任务切换；AirfRANS 则直接 `pip install airfrans` 获取 1000 个仿真与四类任务。这说明工程 surrogate 不应只重“模型类”，还必须重“任务类”与“split 类”。对你来说，一个很有价值的落地动作是把数据集拆成：**插值测试、几何外推测试、模态外推测试、频段外推测试、峰区专项测试**。 citeturn36view2turn36view0turn23view2

**AE-DeepONet 结构动力学论文** 还给出了很实用的训练尺度：作者在 PyTorch 上、RTX 3080 上，forward surrogate 训练使用 10000 epochs、batch size 64，并通过 gradient-based initialization + neural refinement 做 inverse estimation。这至少说明，对于结构动力学 surrogate，**“先把 forward surrogate 训练好，再把它作为 inverse/校正器的内核”**是成熟路线，而不是把所有任务混成一个 end-to-end 大模型。 citeturn28view2turn28view3

开放问题也需要明确说清。其一，这次检索中有些较新工作论文完整，但官方代码未充分公开或尚不成熟，例如主动多保真 airfoil surrogate、部分结构动力学残差修正类工作；对这些工作，我更建议借思想而不是直接照搬实现。其二，大多数 geometry-aware/operator 论文验证的目标是**场变量或时域响应**，而不是节点级频域应力 PSD，所以落地时一定要自己重做输出定义与损失设计。其三，DDPM/Transformer 类方法在工程数据中很容易被“看起来更强”所诱惑，但如果你的数据量不大、标签噪声高、峰区样本极不平衡，先做物理编码与 baseline+residual，往往比直接上生成式模型更有效。最后，若你的任务最终还要服务优化或设计筛选，那么**可解释与可信度**应从第一版模型就进入接口，而不是后补。以上判断主要是基于本次检索到的公开论文/官方仓库；对于极新的工作，结论仍应以你自己的小规模 A/B test 为准。 citeturn6search3turn26view0turn24view7turn24view8turn24view0turn29view4turn27view0