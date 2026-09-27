"""analysis.attribution — 模型专属 attribution 统一抽取 (线性/树/深度)。

铁律: 输出为 importance/attribution (非统计检验)。SNR 与 effect 分列存放;
p/FDR 只存在于统计层 (线性检验另由 statistics 层处理)。
"""
