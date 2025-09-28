"""Parse and structure technique inspiration for analysis methods"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class AnalysisTechnique:
    """Structured representation of an analysis technique"""
    name: str
    method_type: str  # "causal", "pattern", "anomaly", etc.
    description: str
    trader_value: str
    implementation: str
    required_data: List[str]
    output_artifacts: List[str]
    robustness_checks: List[str]


class TechniqueLibrary:
    """Parse and manage technique inspiration"""

    def __init__(self, filepath: str = "data/technique_inspiration.md"):
        self.filepath = Path(filepath)
        self.techniques = self._parse_techniques()
        self.trader_questions = self._extract_trader_questions()

    def _parse_techniques(self) -> List[AnalysisTechnique]:
        """Extract structured techniques from inspiration"""
        techniques = [
            # Shock & Impact Analysis (Section A)
            AnalysisTechnique(
                name="Event Study DiD",
                method_type="causal",
                description="Measure causal impact of infrastructure events using difference-in-differences",
                trader_value="Quantify flow redistributions and basis impacts from outages/expansions",
                implementation="Panel DiD with pipeline×location and day fixed effects",
                required_data=["event dates", "treated/control delivery points", "daily scheduled_quantity"],
                output_artifacts=["ATT_estimate", "pretrend_plot", "effect_window", "robustness_table"],
                robustness_checks=["Pre-trend testing", "Placebo dates", "Synthetic DiD for small pools"]
            ),
            AnalysisTechnique(
                name="Regression Discontinuity in Time",
                method_type="causal",
                description="Analyze threshold effects around capacity constraints or policy changes",
                trader_value="Identify constraint activation points and scarcity pricing triggers",
                implementation="RDiT around capacity thresholds with local linear regression",
                required_data=["capacity utilization", "threshold crossings", "power plant deliveries"],
                output_artifacts=["local_ATE", "bandwidth_sensitivity", "lead_lag_plot"],
                robustness_checks=["Multiple bandwidths", "Placebo thresholds", "Density tests"]
            ),
            AnalysisTechnique(
                name="Synthetic Control",
                method_type="causal",
                description="Construct counterfactual for single treated unit using donor pool",
                trader_value="Precise impact measurement for major single events",
                implementation="Weighted donor matching with ridge-stabilized weights",
                required_data=["single treatment unit", "donor pool", "pre-period outcomes"],
                output_artifacts=["synthetic_path", "gap_plot", "donor_weights", "placebo_inference"],
                robustness_checks=["Leave-one-donor-out", "Randomization inference", "RMSPE ratios"]
            ),
            AnalysisTechnique(
                name="Instrumental Variables",
                method_type="causal",
                description="Use upstream shocks to instrument downstream flows",
                trader_value="Causal elasticities for forecasting and risk management",
                implementation="2SLS with upstream production shocks as instruments",
                required_data=["upstream production", "downstream deliveries", "flow shares"],
                output_artifacts=["first_stage_F", "2SLS_elasticity", "IV_diagnostics"],
                robustness_checks=["Weak IV tests", "Alternative instruments", "Jackknife by pipeline"]
            ),
            AnalysisTechnique(
                name="Bayesian Structural Time Series",
                method_type="causal",
                description="CausalImpact-style counterfactual analysis with uncertainty",
                trader_value="Probabilistic impact assessment with confidence bounds",
                implementation="BSTS with controls from non-overlapping pipelines",
                required_data=["treated series", "control series", "pre/post periods"],
                output_artifacts=["pointwise_impact", "cumulative_impact", "posterior_intervals"],
                robustness_checks=["Weekly aggregation", "Alternative control sets", "Prior sensitivity"]
            ),

            # Propagation & Network Analysis (Section B)
            AnalysisTechnique(
                name="Panel VAR Lead-Lag",
                method_type="causal",
                description="Identify which pipelines lead others using Vector Autoregression",
                trader_value="Forecasting advantage and substitution pattern discovery",
                implementation="Panel VAR with Granger causality on state-level aggregates",
                required_data=["daily pipeline aggregates", "state-level flows", "interconnect data"],
                output_artifacts=["lead_lag_heatmap", "impulse_responses", "significant_edges"],
                robustness_checks=["Seasonal controls", "Block bootstrap", "Lag selection criteria"]
            ),
            AnalysisTechnique(
                name="Transfer Entropy",
                method_type="pattern",
                description="Detect nonlinear lead-lag relationships between pipeline flows",
                trader_value="Capture complex propagation patterns missed by linear methods",
                implementation="Cross-mutual information with permutation testing",
                required_data=["interconnect receipts", "downstream deliveries", "time series"],
                output_artifacts=["predictive_pairs", "lag_structure", "permutation_pvalues"],
                robustness_checks=["Outlier winsorization", "Duplicate removal", "Bootstrap CIs"]
            ),
            AnalysisTechnique(
                name="Network Flow Centrality",
                method_type="pattern",
                description="Identify structural bottlenecks using graph analysis",
                trader_value="Locate critical infrastructure for capacity planning",
                implementation="Edge-betweenness centrality on flow correlation network",
                required_data=["location pairs", "flow correlations", "pipeline connections"],
                output_artifacts=["bottleneck_locations", "centrality_scores", "community_structure"],
                robustness_checks=["Correlation thresholds", "Louvain vs Infomap clustering", "Temporal stability"]
            ),

            # Constraint & Scarcity Analysis (Section C)
            AnalysisTechnique(
                name="Utilization Constraint Detection",
                method_type="pattern",
                description="Identify capacity constraints using rolling quantile caps",
                trader_value="Early warning for scarcity pricing and constraint activation",
                implementation="Rolling 95th percentile capacity proxy with regime detection",
                required_data=["pipeline flows", "location deliveries", "time series"],
                output_artifacts=["constraint_periods", "utilization_series", "regime_changes"],
                robustness_checks=["90th vs 97.5th percentiles", "Weekly aggregation", "Storage adjustments"]
            ),
            AnalysisTechnique(
                name="Mass Balance Tightness",
                method_type="pattern",
                description="Quantify pipeline mass balance constraints on deliveries",
                trader_value="Operational stress indicators for flow reallocation",
                implementation="Constrained regression: ΔDeliveries ~ HighUtil(Receipts)",
                required_data=["receipts by pipeline", "deliveries by location", "utilization ratios"],
                output_artifacts=["marginal_effects", "tightness_by_state", "composition_controls"],
                robustness_checks=["Net flow normalization", "Category mix controls", "Time-varying effects"]
            ),

            # Heterogeneity & Treatment Effects (Section D)
            AnalysisTechnique(
                name="Causal Forests",
                method_type="causal",
                description="Heterogeneous treatment effects using machine learning",
                trader_value="Identify which customer types benefit most from capacity changes",
                implementation="X-learner on shock exposure with honest splitting",
                required_data=["treatment exposure", "delivery changes", "customer categories"],
                output_artifacts=["heterogeneous_effects", "SHAP_drivers", "segment_analysis"],
                robustness_checks=["Honest splitting", "Calibration tests", "Monotonicity checks"]
            ),
            AnalysisTechnique(
                name="Quantile DiD",
                method_type="causal",
                description="Analyze treatment effects across the outcome distribution",
                trader_value="Understand impacts on delivery tail risk and extreme flows",
                implementation="Quantile fixed-effects DiD at multiple quantiles",
                required_data=["treatment/control groups", "outcome quantiles", "panel structure"],
                output_artifacts=["quantile_ATT", "tail_effects", "distribution_changes"],
                robustness_checks=["Wild bootstrap", "Pre-trend at each quantile", "Uniform inference"]
            ),

            # Data Quality & Anomaly Detection (Section E)
            AnalysisTechnique(
                name="Balance Violation Detection",
                method_type="anomaly",
                description="Detect mass balance anomalies using extreme value theory",
                trader_value="Operational disruption signals and data quality flags",
                implementation="EVT tail fitting on |receipts - deliveries| outliers",
                required_data=["receipts by pipeline", "deliveries by pipeline", "daily imbalances"],
                output_artifacts=["anomaly_rankings", "segment_contributions", "tail_parameters"],
                robustness_checks=["Storage category adjustments", "Deduplication sensitivity", "Seasonal patterns"]
            ),
            AnalysisTechnique(
                name="Zero-Inflation Regime Analysis",
                method_type="pattern",
                description="Model shifts in zero-flow probability and intensity",
                trader_value="Operational state changes and duty cycle analysis",
                implementation="Two-part hurdle model with changepoint detection",
                required_data=["flow indicators", "positive flows", "seasonal controls"],
                output_artifacts=["zero_probability_shifts", "intensity_changes", "regime_dates"],
                robustness_checks=["Complementary log-log links", "Alternative hurdle models", "Stationarity tests"]
            ),
            AnalysisTechnique(
                name="Duplicate Impact Analysis",
                method_type="pattern",
                description="Assess how data duplicates affect analysis conclusions",
                trader_value="Data quality assurance for trading model reliability",
                implementation="Recompute key effects with/without duplicate records",
                required_data=["full dataset", "deduplicated dataset", "key analysis results"],
                output_artifacts=["sensitivity_table", "rank_changes", "effect_deltas"],
                robustness_checks=["Multiple duplicate definitions", "Exact vs fuzzy matching", "Volume thresholds"]
            ),

            # Segment Discovery & Clustering (Section F)
            AnalysisTechnique(
                name="Hidden Segment Clustering",
                method_type="pattern",
                description="Discover operational archetypes using unsupervised learning",
                trader_value="Market segmentation for targeted trading strategies",
                implementation="UMAP + HDBSCAN on seasonal/utilization/volatility features",
                required_data=["pipeline characteristics", "seasonal patterns", "operational metrics"],
                output_artifacts=["segment_clusters", "feature_importance", "exemplar_locations"],
                robustness_checks=["Feature standardization", "Stability across seeds", "Silhouette analysis"]
            ),
            AnalysisTechnique(
                name="Conditional Dependency Mapping",
                method_type="pattern",
                description="Uncover sparse conditional associations using graphical models",
                trader_value="Identify basis trading opportunities through dependency structure",
                implementation="Graphical Lasso on state×category daily panels",
                required_data=["state aggregates", "category breakdowns", "time series"],
                output_artifacts=["dependency_graph", "basis_state_candidates", "conditional_correlations"],
                robustness_checks=["Regularization path", "Cross-validation", "Network stability"]
            ),

            # Counterfactual Forecasting (Section G)
            AnalysisTechnique(
                name="Counterfactual Flow Forecasting",
                method_type="forecasting",
                description="Predict flows under alternative shock scenarios",
                trader_value="Risk scenario analysis and hedging strategy development",
                implementation="SARIMAX with/without shock features for counterfactual comparison",
                required_data=["historical flows", "shock indicators", "external features"],
                output_artifacts=["counterfactual_paths", "confidence_intervals", "feature_importance"],
                robustness_checks=["Cross-validation", "Alternative models", "Shock magnitude sensitivity"]
            ),
            AnalysisTechnique(
                name="Optimal Transport Reallocation",
                method_type="optimization",
                description="Model flow reallocation using constrained optimal transport",
                trader_value="Predict alternative routing during capacity constraints",
                implementation="Network flow optimization with historical reallocation patterns",
                required_data=["capacity constraints", "historical rerouting", "network topology"],
                output_artifacts=["reallocation_probabilities", "destination_rankings", "expected_lags"],
                robustness_checks=["Capacity bound sensitivity", "Historical pattern validation", "Network structure tests"]
            )
        ]
        return techniques

    def _extract_trader_questions(self) -> List[Dict[str, str]]:
        """Extract specific trader-relevant questions"""
        return [
            {
                "question": "Which locations act as swing suppliers during cold snaps?",
                "method": "RDiD",
                "value": "Reliability premiums and backup capacity value"
            },
            {
                "question": "How quickly do new pipeline projects reach capacity?",
                "method": "structural_break",
                "value": "Market share shifts and basis convergence timing"
            },
            {
                "question": "What are the elasticities of flows to weather shocks?",
                "method": "IV",
                "value": "Demand response and price sensitivity"
            },
            {
                "question": "Where do constrained flows get rerouted?",
                "method": "spillover_analysis",
                "value": "Alternative route values and congestion rents"
            },
            {
                "question": "Which interconnects show persistent imbalances?",
                "method": "time_series",
                "value": "Structural bottlenecks and investment opportunities"
            }
        ]

    def get_technique_by_type(self, method_type: str) -> List[AnalysisTechnique]:
        """Get techniques of a specific type"""
        return [t for t in self.techniques if t.method_type == method_type]

    def suggest_technique_for_question(self, trader_question: str) -> Optional[AnalysisTechnique]:
        """Match a trader question to appropriate technique"""
        # Simple keyword matching for MVP
        question_lower = trader_question.lower()

        if "constraint" in question_lower or "capacity" in question_lower:
            return next((t for t in self.techniques if "constraint" in t.name.lower()), self.techniques[0])
        elif "lead" in question_lower or "predict" in question_lower:
            return next((t for t in self.techniques if "lead" in t.name.lower() or "var" in t.name.lower()), self.techniques[0])
        elif "impact" in question_lower or "effect" in question_lower:
            return next((t for t in self.techniques if "event study" in t.name.lower()), self.techniques[0])
        elif "anomaly" in question_lower or "unusual" in question_lower:
            return next((t for t in self.techniques if "balance" in t.name.lower() or "anomaly" in t.description.lower()), self.techniques[0])

        return self.techniques[0]  # Default to first technique