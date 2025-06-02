# ChatGPT 数据分析

[English Version](README.md)

一个全面的 ChatGPT 对话数据分析工具，将你导出的对话历史转化为富有洞察力的可视化图表和成本分析。

## 功能特点

有没有想过你的 ChatGPT 订阅到底值不值？这个项目通过分析你导出的 ChatGPT 对话数据来回答这个问题，还有更多发现等着你。最初受 [Chip Huyen 的 AI 热力图](https://github.com/chiphuyen/aie-book/blob/main/scripts/ai-heatmap.ipynb)启发，这个分析工具走得更远：

- **GitHub 风格的热力图** 展示每日 ChatGPT 使用模式
- **Token 使用分析** 基于 API 定价的准确成本计算
- **模型使用趋势** 展示你的模型偏好演变过程
- **深度研究追踪** 监控配额使用情况
- **订阅价值分析** 比较实际使用成本与订阅费用

## 快速开始

### 1. 导出 ChatGPT 数据
1. 前往 [ChatGPT 设置](https://chatgpt.com/settings) → 数据控制 → 导出数据
2. 收到邮件链接后下载并解压 ZIP 文件

### 2. 安装依赖
```bash
pip install numpy pandas pytz plotly tiktoken jupyter
```

### 3. 克隆并运行
```bash
git clone https://github.com/YuanpingSong/chatgpt-analytics.git
cd chatgpt-analytics
jupyter notebook scripts/analysis.ipynb
```

### 4. 配置分析参数
运行前在 notebook 中更新以下参数：

```python
# 必填：你解压后的 ChatGPT 数据文件夹路径
convo_folder = "/path/to/your/chatgpt_export"

# 你的时区，用于准确的日期聚合
user_timezone = "America/New_York"

# 你的订阅级别（影响深度研究配额分析）
subscription_level = "pro"  # "free", "plus", 或 "pro"
```

## 分析结果

notebook 会生成以下交互式可视化图表：

- **活动热力图**：GitHub 风格的每日使用模式
- **模型使用趋势**：展示你的模型偏好如何演变
- **深度研究分析**：使用量对比配额限制（免费版 5 次/月，Plus 10 次/月，Pro 125 次/月）
- **成本效益分析**：订阅节省费用对比按使用付费定价

## 问题排查

**路径问题**：确保 `convo_folder` 指向解压后的文件夹，而不是 ZIP 文件  
**导入错误**：尝试 `pip install --upgrade tiktoken plotly`  
**无数据**：确保你的导出文件中包含实际的 ChatGPT 对话

## 博客文章

在配套的博客文章中阅读完整的分析和见解，文章涵盖了个人使用模式、不同订阅层级的成本效益，以及运行此分析的详细发现。

📖 **[阅读博客文章：分析我的 ChatGPT 使用情况 - Pro 订阅值得吗？](https://songyp.com/zh/blog/analyzing-chatgpt-usage)**

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- 原始热力图概念来自 [Chip Huyen](https://github.com/chiphuyen/aie-book)
- 基于 pandas、plotly 和 tiktoken 构建