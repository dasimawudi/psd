# toBdoc 6-9章过程数据

本目录保存《（终）应力仿真 AI 代理技术报告.docx》6-9章补充内容使用的可复用过程数据。

- `psd_top_percent_metrics_threshold_strategy.csv`：逐频 PSD 独立测试集高响应分层指标。
- `rmises_region_top_percent_metrics_threshold_strategy.csv`：最终 RMises 分区域高响应分层指标。
- `baseline_comparison_topk_relative_metrics.csv`：前序基线模型与当前模型的 within25 / 相对 MAE 分层对比数据。
- `final_result_topk_metrics_compact.csv`：报告表 7-1 使用的终版模型 topk 分层指标。
- `relative_error_spatial_selected_cases.csv`：三个展示样本的最终 RMises 整体 within25 / 相对 MAE 等摘要指标。
- `relative_error_spatial_representative_case.csv`：三个展示样本的最终 RMises 逐节点相对误差空间数据。
- `relative_error_psd_node_aggregate_selected_cases.csv`：三个展示样本的逐频 PSD 节点聚合误差摘要指标。
- `relative_error_psd_node_aggregate.csv`：三个展示样本的逐频 PSD 节点聚合误差空间数据。
- `low_response_threshold_stats.csv`：target<200 低响应点统计。
- `dataset_scale.csv`：训练/验证/测试集规模。
- `model_config_summary.csv`：终版模型配置摘要。
- `followup_work_plan.csv`：第 10 章后续工作计划结构化数据。
- `x_axis_bin_stress_ranges_100bins.csv`：log1p 直方图 bin 到原始应力范围映射。

注意：低响应阈值处理策略相关指标均采用 target<200 统一按 200 处理的训练/评估口径，不等同于原始未阈值处理口径。
