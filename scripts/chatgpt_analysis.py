import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pytz

try:
    import tiktoken  # type: ignore
    try:  # verify encoding files are available; disable if not
        tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - offline or missing data
        tiktoken = None
except Exception:  # pragma: no cover - fallback when tiktoken missing
    tiktoken = None

# price in USD per million tokens
# https://openai.com/api/pricing/
MODEL_PRICING = {
    "gpt-4-1": {"input": 2, "output": 8},
    "gpt-4-1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4-1-nano": {"input": 0.1, "output": 0.4},
    "gpt-4-5": {"input": 75, "output": 150 },
    "gpt-4o": {"input": 2.5, "output": 10 },
    "gpt-4o-mini": {"input": 0.15, "output": 0.6 },
    "o1": { "input": 15, "output": 60 },
    "o1-pro": { "input": 150, "output": 600},
    "o3": { "input": 10, "output": 40 },
    "o4-mini": { "input": 1.1, "output": 4.4},
    "o3-mini": { "input": 1.1, "output": 4.4},
    "o1-mini": { "input": 1.1, "output": 4.4},
    "codex-mini": {"input": 1.5, "output": 6},

    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


@dataclass
class Message:
    conv_id: str
    msg_id: str
    parent_id: Optional[str]
    children: List[str]
    role: str
    create_time: float
    content: str
    content_type: str
    model: Optional[str]
    metadata: Dict
    tokens: int = 0
    context_tokens: int = 0


def load_conversations(path: str) -> List[dict]:
    with open(f"{path}/conversations.json", "r", encoding="utf-8") as f:
        return json.load(f)


def extract_messages(convs: Iterable[dict]) -> List[Message]:
    messages: List[Message] = []
    for idx, conv in enumerate(convs):
        conv_id = str(conv.get("conversation_id", idx))
        mapping = conv.get("mapping", {})
        for node_id, node in mapping.items():
            msg = node.get("message")
            if not msg:
                continue
            author = msg.get("author", {}).get("role", "")
            ts = msg.get("create_time")
            if ts is None:
                continue
            content = msg.get("content", {})
            ctype = content.get("content_type", "text")
            text = ""
            if isinstance(content, dict):
                if ctype == "text" and isinstance(content.get("parts"), list):
                    # a message can have images, we will ignore them for now
                    parts = list(filter(lambda x: isinstance(x, str), content.get("parts", [])))
                    text = "\n".join(parts)
                else:
                    text = content.get("text") or ""
                    parts = list(filter(lambda x: isinstance(x, str), content.get("parts", [])))
                    text = text or "\n".join(parts)
            model = msg.get("metadata", {}).get("model_slug")
            metadata = msg.get("metadata", {})
            messages.append(
                Message(
                    conv_id=conv_id,
                    msg_id=node_id,
                    parent_id=node.get("parent"),
                    children=node.get("children", []),
                    role=author,
                    create_time=ts,
                    content=text,
                    content_type=ctype,
                    model=model,
                    metadata=metadata,
                )
            )
    return messages


def assign_user_models(messages: List[Message]) -> None:
    model_map = {m.parent_id: m.model for m in messages if m.role == "assistant"}
    for m in messages:
        if m.role == "user" and not m.model:
            m.model = model_map.get(m.msg_id)
    for m in messages:
        if m.tokens == 0:
            m.tokens = token_counter(m.content, m.model)


def token_counter(text: str, model: Optional[str]) -> int:
    if not text:
        return 0
    encoding = None
    if tiktoken is not None and model:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            pass
    if encoding:
        try:
            return len(encoding.encode(text))
        except Exception:
            pass
    # simple heuristic fallback ~4 chars per token
    return max(1, int(len(text) / 4))


def calculate_conversation_tokens(messages: List[Message]) -> Dict[str, Dict[str, int]]:
    """
    Calculate actual input/output tokens for each message considering conversation context.
    
    In conversational AI:
    - Each assistant response uses the entire conversation history as input context
    - User messages don't incur API costs (they're just added to context)
    - Input tokens = all previous messages in the conversation
    - Output tokens = only the current assistant response
    """
    conv_map: Dict[str, Dict[str, Message]] = defaultdict(dict)
    for msg in messages:
        conv_map[msg.conv_id][msg.msg_id] = msg

    token_data: Dict[str, Dict[str, int]] = {}

    def traverse(cdict: Dict[str, Message], mid: str, context_tokens: int) -> None:
        msg = cdict[mid]
        msg.context_tokens = context_tokens
        if msg.role == "assistant":
            input_tokens = context_tokens
            output_tokens = msg.tokens
        else:
            input_tokens = 0
            output_tokens = 0
        token_data[mid] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        next_ctx = context_tokens + msg.tokens
        for child_id in msg.children:
            child = cdict.get(child_id)
            if child:
                traverse(cdict, child_id, next_ctx)

    for conv_id, cdict in conv_map.items():
        roots = [m for m in cdict.values() if not m.parent_id or m.parent_id not in cdict]
        for r in roots:
            traverse(cdict, r.msg_id, 0)

    return token_data


def build_dataframe(messages: Iterable[Message], tz: str) -> pd.DataFrame:
    messages_list = list(messages)
    token_data = calculate_conversation_tokens(messages_list)
    
    rows = []
    for msg in messages_list:
        local_dt = (
            datetime.fromtimestamp(msg.create_time, tz=timezone.utc)
            .astimezone(pytz.timezone(tz))
        )
        
        msg_tokens = token_data.get(msg.msg_id, {"input_tokens": 0, "output_tokens": 0})
        
        rows.append(
            {
                "conv_id": msg.conv_id,
                "msg_id": msg.msg_id,
                "role": msg.role,
                "model": msg.model,
                "content": msg.content,
                "content_type": msg.content_type,
                "dt": local_dt,
                "metadata": msg.metadata,
                "tokens": msg.tokens,
                "input_tokens": msg_tokens["input_tokens"],
                "output_tokens": msg_tokens["output_tokens"],
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = df["dt"].dt.date
    return df


def detect_advanced_features(df: pd.DataFrame) -> pd.Series:
    def is_advanced(row: pd.Series) -> bool:
        meta = row.get("metadata", {}) or {}
        text = row.get("content", "") or ""
        hints = meta.get("system_hints", [])
        if isinstance(hints, str):
            hints = [hints]
        if "search" in hints or meta.get("command") == "search":
            return True
        if "search(" in text or row.get("role") == "tool":
            return True
        return False

    return df.apply(is_advanced, axis=1)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("date")
        .agg(
            conversations=("conv_id", lambda x: x.nunique()),
            user_messages=("role", lambda x: (x == "user").sum()),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
        )
        .reset_index()
    )
    return agg


def aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("model")
        .agg(
            conversations=("conv_id", lambda x: x.nunique()),
            messages=("msg_id", "count"),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
        )
        .reset_index()
    )
    return agg


def format_model_stats_display(model_stats: pd.DataFrame) -> pd.DataFrame:
    """Format model stats with token counts in millions for better readability."""
    display_stats = model_stats.copy()
    display_stats["input_tokens_M"] = (display_stats["input_tokens"] / 1_000_000).round(2)
    display_stats["output_tokens_M"] = (display_stats["output_tokens"] / 1_000_000).round(2)
    
    # Reorder columns to show millions first, then raw counts
    columns = ["model", "conversations", "messages", "input_tokens_M", "output_tokens_M", "input_tokens", "output_tokens"]
    return display_stats[columns]


def estimate_cost(df: pd.DataFrame) -> pd.DataFrame:
    costs = []
    for model, data in df.groupby("model"):
        # clean up model name
        cleaned_model_name = model.replace("-high", "").replace("-preview", "").replace("-browsing", "").replace("-plugins", "")
        price = MODEL_PRICING.get(cleaned_model_name, {"input": 0.0, "output": 0.0})
        in_tok = data["input_tokens"].sum() / 1e6
        out_tok = data["output_tokens"].sum() / 1e6
        thinking_tok = 0

        is_reasoning_model = model.startswith("o")
        if is_reasoning_model:
            # assume thinking tokens are 3 times the output tokens
            thinking_tok = out_tok * 3
            cost = in_tok * price["input"] + out_tok * price["output"] + thinking_tok * price["output"]
        else:
            cost = in_tok * price["input"] + out_tok * price["output"]

        costs.append({
            "model": model, 
            "input_tokens (M)": round(in_tok, 3), 
            "output_tokens (M)": round(out_tok, 3), 
            "thinking_tokens (M)": round(thinking_tok, 3), 
            "cost_usd": round(cost, 2)
        })
    return pd.DataFrame(costs)


def subscription_value(df_cost: pd.DataFrame, months: int) -> pd.DataFrame:
    payg = df_cost["cost_usd"].sum()
    sub_cost = 20 * months
    return pd.DataFrame(
        {
            "period_months": [months],
            "pay_per_use": [payg],
            "subscription_cost": [sub_cost],
            "difference": [sub_cost - payg],
        }
    )


def plot_heatmap(df: pd.DataFrame, year: int, column: str, title: str) -> None:
    df_year = df[df["date"].apply(lambda d: d.year == year)]
    counts = dict(zip(df_year["date"], df_year[column]))
    start = datetime(year, 1, 1).date()
    end = datetime(year, 12, 31).date()
    total_days = (end - start).days + 1
    date_range = [start + pd.Timedelta(days=i) for i in range(total_days)]
    data = []
    for dt in date_range:
        week = ((dt - start).days + start.weekday()) // 7
        day = dt.weekday()
        count = counts.get(dt, 0)
        data.append((week, day, count))
    weeks_in_year = (end - start).days // 7 + 1
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    ax.set_aspect("equal")
    vals = list(counts.values()) or [0]
    p90 = np.percentile(vals, 90) or 1
    for week, day, count in data:
        color = plt.cm.Greens((count + 1) / p90) if count > 0 else "lightgray"
        rect = patches.Rectangle((week, day), 1, 1, linewidth=0.5, edgecolor="black", facecolor=color)
        ax.add_patch(rect)
    months = [start + pd.Timedelta(days=i) for i in range(total_days) if (start + pd.Timedelta(days=i)).day == 1]
    for m in months:
        week = (m - start).days // 7
        plt.text(week + 0.5, 7.75, m.strftime("%b"), ha="center", va="center", fontsize=10)
    ax.set_xlim(-0.5, weeks_in_year + 0.5)
    ax.set_ylim(-0.5, 8.5)
    plt.title(title)
    plt.xticks([])
    plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.gca().invert_yaxis()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatGPT usage analytics")
    parser.add_argument("convo_folder", help="Path to exported ChatGPT folder")
    parser.add_argument("--tz", default="UTC", help="Local timezone name")
    parser.add_argument("--year", type=int, help="Year to plot")
    args = parser.parse_args()

    convs = load_conversations(args.convo_folder)
    msgs = extract_messages(convs)
    assign_user_models(msgs)
    df = build_dataframe(msgs, args.tz)

    daily = aggregate_daily(df)
    model_stats = aggregate_by_model(df)
    costs = estimate_cost(df)
    months = df["dt"].dt.to_period("M").nunique()
    value = subscription_value(costs, months)

    year = args.year or datetime.now().year
    plot_heatmap(daily, year, "conversations", f"{year} ChatGPT Conversation Heatmap")
    plot_heatmap(daily, year, "user_messages", f"{year} ChatGPT User Messages Heatmap")

    advanced = detect_advanced_features(df)
    adv_count = advanced.sum()
    print(f"Advanced feature messages: {adv_count}")
    print("\nModel Statistics (token counts in millions):")
    print(format_model_stats_display(model_stats))
    print("\nCost Estimates:")
    print(costs)
    print("\nSubscription Value Analysis:")
    print(value)


if __name__ == "__main__":
    main()
