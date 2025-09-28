#!/usr/bin/env python3
"""
Export formalized portfolio seeds for execution refinement loop
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append("src")

from idea_generation_mvp.inspiration.events import EventLibrary
from idea_generation_mvp.inspiration.techniques import TechniqueLibrary
from idea_generation_mvp.core.seeds import SeedGenerator
from idea_generation_mvp.evaluation.scorer import IdeaScorer, PortfolioSelector


def export_portfolio_seeds():
    """Generate and export the portfolio seeds as formalized specifications"""

    print("🌱 Generating portfolio seeds...")

    # Initialize components
    events = EventLibrary()
    techniques = TechniqueLibrary()
    generator = SeedGenerator(events, techniques)
    scorer = IdeaScorer()
    selector = PortfolioSelector()

    # Generate seeds using small config for consistency
    config = {
        "n_event_driven": 3,
        "n_technique_driven": 3,
        "n_hybrid": 2,
        "n_trader_questions": 2
    }

    seeds = generator.generate_all_seeds(config)

    # Score all seeds
    all_scores = {}
    for idea in seeds:
        scores = scorer.score_idea(idea)
        all_scores[idea.id] = scores
        idea.scores = scores

    # Select portfolio
    portfolio = selector.select_portfolio(seeds, all_scores, n=3)

    print(f"✅ Generated {len(seeds)} seeds, selected portfolio of {len(portfolio)}")

    # Export portfolio as formalized JSON
    portfolio_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_seeds": len(seeds),
            "portfolio_size": len(portfolio),
            "selection_method": "diversity_optimized"
        },
        "portfolio": []
    }

    for i, idea in enumerate(portfolio):
        scores = all_scores.get(idea.id, {})

        # Convert to serializable format
        idea_data = {
            "id": idea.id,
            "rank": i + 1,
            "query": idea.query,
            "method": idea.method,
            "assets": idea.assets,
            "event_context": idea.event_context,
            "technique_context": idea.technique_context,
            "trader_hook": idea.trader_hook,
            "time_window": idea.time_window,
            "expected_artifacts": idea.expected_artifacts,
            "design_type": idea.design_type,
            "complexity": idea.complexity,
            "confidence": idea.confidence,
            "scores": scores
        }

        portfolio_data["portfolio"].append(idea_data)

    # Save to JSON file
    output_file = Path("data/portfolio_seeds.json")
    with open(output_file, 'w') as f:
        json.dump(portfolio_data, f, indent=2, default=str)

    print(f"💾 Portfolio exported to: {output_file}")

    # Print summary
    print(f"\n🎯 FORMALIZED PORTFOLIO:")
    for i, idea in enumerate(portfolio):
        scores = all_scores.get(idea.id, {})
        print(f"\n{i+1}. {idea.query}")
        print(f"   ID: {idea.id}")
        print(f"   Method: {idea.method}")
        print(f"   Assets: {', '.join(idea.assets)}")
        print(f"   Overall Score: {scores.get('overall', 0):.2f}")
        print(f"   Event Context: {idea.event_context or 'None'}")

    return portfolio_data


if __name__ == "__main__":
    export_portfolio_seeds()