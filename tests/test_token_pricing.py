#!/usr/bin/env python3
"""
Test for token pricing calculation using pricing data files.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from native_transparent_agent import calculate_token_cost

def test_gpt5_mini_pricing():
    """Test that GPT-5 Mini pricing is correctly retrieved from openai_pricing.tsv"""

    # Test case: 1 million input tokens, 1 million output tokens
    # Expected: $0.25 + $2.00 = $2.25
    cost = calculate_token_cost("openai:gpt-5-mini", 1_000_000, 1_000_000)
    expected_cost = 0.25 + 2.00
    assert abs(cost - expected_cost) < 0.001, f"Expected {expected_cost}, got {cost}"
    print(f"✅ GPT-5 Mini 1M/1M tokens: ${cost:.2f} (expected ${expected_cost:.2f})")

    # Test case: 100K input tokens, 50K output tokens
    # Expected: (100K/1M) * $0.25 + (50K/1M) * $2.00 = $0.025 + $0.10 = $0.125
    cost = calculate_token_cost("openai:gpt-5-mini", 100_000, 50_000)
    expected_cost = (100_000 / 1_000_000) * 0.25 + (50_000 / 1_000_000) * 2.00
    assert abs(cost - expected_cost) < 0.001, f"Expected {expected_cost}, got {cost}"
    print(f"✅ GPT-5 Mini 100K/50K tokens: ${cost:.3f} (expected ${expected_cost:.3f})")

    # Test case: 1K input tokens, 2K output tokens
    # Expected: (1K/1M) * $0.25 + (2K/1M) * $2.00 = $0.00025 + $0.004 = $0.00425
    cost = calculate_token_cost("openai:gpt-5-mini", 1_000, 2_000)
    expected_cost = (1_000 / 1_000_000) * 0.25 + (2_000 / 1_000_000) * 2.00
    assert abs(cost - expected_cost) < 0.00001, f"Expected {expected_cost}, got {cost}"
    print(f"✅ GPT-5 Mini 1K/2K tokens: ${cost:.5f} (expected ${expected_cost:.5f})")

    print("🎉 All GPT-5 Mini pricing tests passed!")

def test_claude_4_sonnet_pricing():
    """Test that Claude 4 Sonnet pricing is correctly retrieved from anthropic_pricing.tsv"""

    # Test case: 1 million input tokens, 1 million output tokens
    # Expected: $3.00 + $15.00 = $18.00
    cost = calculate_token_cost("anthropic:claude-sonnet-4-20250514", 1_000_000, 1_000_000)
    expected_cost = 3.00 + 15.00
    assert abs(cost - expected_cost) < 0.001, f"Expected {expected_cost}, got {cost}"
    print(f"✅ Claude 4 Sonnet 1M/1M tokens: ${cost:.2f} (expected ${expected_cost:.2f})")

    # Test case: 10K input tokens, 5K output tokens
    # Expected: (10K/1M) * $3.00 + (5K/1M) * $15.00 = $0.03 + $0.075 = $0.105
    cost = calculate_token_cost("anthropic:claude-sonnet-4-20250514", 10_000, 5_000)
    expected_cost = (10_000 / 1_000_000) * 3.00 + (5_000 / 1_000_000) * 15.00
    assert abs(cost - expected_cost) < 0.001, f"Expected {expected_cost}, got {cost}"
    print(f"✅ Claude 4 Sonnet 10K/5K tokens: ${cost:.3f} (expected ${expected_cost:.3f})")

    print("🎉 All Claude 4 Sonnet pricing tests passed!")

def test_model_prefix_handling():
    """Test that model names work with and without prefixes"""

    # Test both with and without "openai:" prefix
    cost_with_prefix = calculate_token_cost("openai:gpt-5-mini", 1000, 1000)
    cost_without_prefix = calculate_token_cost("gpt-5-mini", 1000, 1000)
    assert abs(cost_with_prefix - cost_without_prefix) < 0.00001, "Prefix handling should be consistent"
    print(f"✅ Prefix handling works: ${cost_with_prefix:.5f}")

    # Test both with and without "anthropic:" prefix
    cost_with_prefix = calculate_token_cost("anthropic:claude-sonnet-4-20250514", 1000, 1000)
    cost_without_prefix = calculate_token_cost("claude-sonnet-4-20250514", 1000, 1000)
    assert abs(cost_with_prefix - cost_without_prefix) < 0.00001, "Prefix handling should be consistent"
    print(f"✅ Prefix handling works: ${cost_with_prefix:.5f}")

    print("🎉 All prefix handling tests passed!")

if __name__ == "__main__":
    print("Testing token cost calculation function...")
    print("=" * 60)

    test_gpt5_mini_pricing()
    print()

    test_claude_4_sonnet_pricing()
    print()

    test_model_prefix_handling()
    print()

    print("=" * 60)
    print("🏆 All tests passed! The pricing function works correctly.")
    print(f"📊 GPT-5 Mini pricing confirmed: $0.25 input / $2.00 output per million tokens")
    print(f"📊 Claude 4 Sonnet pricing confirmed: $3.00 input / $15.00 output per million tokens")