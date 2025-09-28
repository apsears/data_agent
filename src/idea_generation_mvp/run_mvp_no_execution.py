#!/usr/bin/env python3
"""
Simplified MVP without execution - demonstrates idea generation and scoring only
"""

import json
import yaml
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append("src")

from idea_generation_mvp.inspiration.events import EventLibrary
from idea_generation_mvp.inspiration.techniques import TechniqueLibrary
from idea_generation_mvp.core.seeds import SeedGenerator
from idea_generation_mvp.evolution.operators import EvolutionOperators
from idea_generation_mvp.evaluation.scorer import IdeaScorer, PortfolioSelector


def main():
    """Simplified MVP demonstration without execution"""
    print("🚀 Starting Simplified Idea Generation MVP Demo")
    print("=" * 50)

    # Load inspiration libraries
    print("\n📚 Loading inspiration libraries...")
    events = EventLibrary()
    techniques = TechniqueLibrary()
    print(f"✅ Loaded {len(events.events)} events and {len(techniques.techniques)} techniques")

    # Generate seeds
    print("\n🌱 Generating seed ideas...")
    generator = SeedGenerator(events, techniques)

    # Use smaller numbers for demo
    config = {
        "n_event_driven": 3,
        "n_technique_driven": 3,
        "n_hybrid": 2,
        "n_trader_questions": 2
    }

    seeds = generator.generate_all_seeds(config)
    print(f"✅ Generated {len(seeds)} seed ideas")

    # Show sample seeds
    print("\n📋 Sample seed ideas:")
    for i, seed in enumerate(seeds[:5]):
        print(f"   {i+1}. {seed.query[:80]}...")
        print(f"      Method: {seed.method}")
        print(f"      Assets: {', '.join(seed.assets[:2])}")
        print(f"      Trader hook: {seed.trader_hook[:60]}...")
        print()

    # Test evolution on first seed
    print("\n🧬 Testing evolution on first seed...")
    operators = EvolutionOperators(events, techniques)

    try:
        evolved = operators.evolve_idea(seeds[0], ['make_more_specific', 'add_robustness'])
        print(f"✅ Evolved into {len(evolved)} variants")

        if evolved:
            print(f"\nExample evolution: {evolved[0].query[:80]}...")

    except Exception as e:
        print(f"❌ Evolution failed: {e}")

    # Score all ideas
    print(f"\n🏆 Scoring {len(seeds)} ideas...")
    scorer = IdeaScorer()
    all_scores = {}

    for idea in seeds:
        scores = scorer.score_idea(idea)
        all_scores[idea.id] = scores
        idea.scores = scores

    print(f"✅ Scored all ideas")

    # Select portfolio
    print(f"\n🎯 Selecting diverse portfolio...")
    selector = PortfolioSelector()
    portfolio = selector.select_portfolio(seeds, all_scores, n=3)

    print(f"✅ Selected portfolio of {len(portfolio)} diverse ideas")

    # Show final portfolio
    print(f"\n🎯 FINAL PORTFOLIO:")
    for i, idea in enumerate(portfolio):
        scores = all_scores.get(idea.id, {})
        print(f"\n{i+1}. {idea.query}")
        print(f"   Overall Score: {scores.get('overall', 0):.2f}")
        print(f"   Domain: {scores.get('domain_relevance', 0):.2f} | "
              f"Trader: {scores.get('trader_value', 0):.2f} | "
              f"Technical: {scores.get('technical_rigor', 0):.2f} | "
              f"Novelty: {scores.get('novelty', 0):.2f}")
        print(f"   Method: {idea.method}")
        print(f"   Assets: {', '.join(idea.assets)}")
        print(f"   Business hook: {idea.trader_hook[:100]}...")

    # Summary
    print("\n" + "=" * 50)
    print("🎉 SIMPLIFIED MVP DEMO COMPLETE!")
    print("=" * 50)
    print(f"📈 Generated: {len(seeds)} total ideas")
    print(f"🏆 Portfolio: {len(portfolio)} diverse, high-value ideas")
    print(f"🎊 Idea Generation MVP core functionality demonstrated!")


if __name__ == "__main__":
    main()