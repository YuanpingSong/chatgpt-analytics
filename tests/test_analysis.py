import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import chatgpt_analysis as heat

SAMPLE = ROOT / 'samples/node-js-network-libraries.json'

def load_sample():
    with SAMPLE.open() as f:
        return [json.load(f)]

def test_extract_and_dataframe():
    convs = load_sample()
    msgs = heat.extract_messages(convs)
    heat.assign_user_models(msgs)
    df = heat.build_dataframe(msgs, 'UTC')
    assert len(df) >= 5
    assert (df['role'] == 'user').sum() >= 2

def test_token_counter():
    count = heat.token_counter('Hello world', 'gpt-3.5-turbo')
    assert isinstance(count, int) and count > 0

def test_cost_estimate():
    convs = load_sample()
    msgs = heat.extract_messages(convs)
    heat.assign_user_models(msgs)
    df = heat.build_dataframe(msgs, 'UTC')
    costs = heat.estimate_cost(df)
    assert not costs.empty


def test_token_totals_sample():
    convs = load_sample()
    msgs = heat.extract_messages(convs)
    heat.assign_user_models(msgs)
    df = heat.build_dataframe(msgs, 'UTC')
    assert df['input_tokens'].sum() == 432
    assert df['output_tokens'].sum() == 909


def test_small_export_integration():
    export = ROOT / 'chatgpt_export/conversations.json'
    with export.open() as f:
        convs = json.load(f)[:3]
    msgs = heat.extract_messages(convs)
    heat.assign_user_models(msgs)
    df = heat.build_dataframe(msgs, 'UTC')
    daily = heat.aggregate_daily(df)
    model_stats = heat.aggregate_by_model(df)
    assert not daily.empty
    assert not model_stats.empty
