# ChatGPT Analytics

[中文版本](README_CN.md)

A comprehensive analysis tool for your ChatGPT conversation data that transforms your exported conversation history into insightful visualizations and cost analysis.

## Features

Ever wondered if your ChatGPT subscription is worth it? This project
analyzes your exported ChatGPT conversation data to answer that question and more. Originally inspired by [Chip Huyen's AI heatmap](https://github.com/chiphuyen/aie-book/blob/main/scripts/ai-heatmap.ipynb), this analysis goes much deeper with:

- **GitHub-style heatmaps** showing daily ChatGPT usage patterns
- **Token usage analysis** with accurate cost calculations based on API pricing
- **Model usage trends** showing how your preferences evolved over time
- **Deep Research tracking** to monitor quota usage against limits
- **Subscription value analysis** comparing actual usage costs vs subscription fees

## Quick Start

### 1. Export Your ChatGPT Data

1. Go to [ChatGPT Settings](https://chatgpt.com/settings) → Data controls → Export data
2. Download and extract the ZIP file when you receive the email link

### 2. Install Dependencies

```bash
pip install numpy pandas pytz plotly tiktoken jupyter
```

### 3. Clone and Run

```bash
git clone https://github.com/YuanpingSong/chatgpt-analytics.git
cd chatgpt-analytics
jupyter notebook scripts/analysis.ipynb
```

### 4. Configure Your Analysis

Update these parameters in the notebook before running:

```python
# REQUIRED: Path to your extracted ChatGPT data folder
convo_folder = "/path/to/your/chatgpt_export"

# Your timezone for accurate daily aggregation
user_timezone = "America/New_York"

# Your subscription level (affects Deep Research quota analysis)
subscription_level = "pro"  # "free", "plus", or "pro"
```

## What You'll Get

The notebook generates interactive visualizations including:

- **Activity Heatmaps**: GitHub-style daily usage patterns
- **Model Usage Trends**: How your model preferences evolved over time
- **Deep Research Analysis**: Usage vs quota limits (5 free, 10 Plus, 125 Pro per month)
- **Cost/Benefit Analysis**: Subscription savings vs pay-per-use pricing

## Troubleshooting

**Path issues**: Make sure `convo_folder` points to the extracted folder, not the ZIP file  
**Import errors**: Try `pip install --upgrade tiktoken plotly`  
**No data**: Ensure you have actual ChatGPT conversations in your export

## Blog Post

Read the full analysis and insights in the accompanying blog post that covers personal usage patterns, cost-effectiveness of different subscription tiers, and detailed findings from running this analysis.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Original heatmap concept by [Chip Huyen](https://github.com/chiphuyen/aie-book)
- Built with pandas, plotly, and tiktoken
