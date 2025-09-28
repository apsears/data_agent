"""Parse and structure event inspiration for idea generation"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path


@dataclass
class Event:
    """Structured representation of a major pipeline event"""
    name: str
    description: str
    start_date: Optional[str]
    end_date: Optional[str]
    affected_assets: List[str]
    event_type: str  # "outage", "expansion", "weather", etc.
    causal_design: str  # Suggested analysis approach
    treatment_suggestion: str
    control_suggestion: str
    expected_impact: str


class EventLibrary:
    """Parse and manage event inspiration"""

    def __init__(self, filepath: str = "data/event_inspiration.md"):
        self.filepath = Path(filepath)
        self.events = self._parse_events()

    def _parse_events(self) -> List[Event]:
        """Extract structured events from markdown"""
        events = []

        # Comprehensive events extracted from inspiration
        major_events = [
            # LNG Terminal Events
            Event(
                name="Freeport LNG Explosion",
                description="Major LNG terminal explosion causing long outage",
                start_date="2022-06-08",
                end_date="2023-02-01",
                affected_assets=["Gulf South Pipeline", "Freeport LNG meters"],
                event_type="outage",
                causal_design="event_study",
                treatment_suggestion="Freeport-named delivery points",
                control_suggestion="Sabine Pass/Cameron LNG or inland pipes",
                expected_impact="Sharp drop in scheduled deliveries, step-up on restart"
            ),
            Event(
                name="Hurricane Beryl Freeport Shutdown",
                description="Hurricane causing temporary ~2 Bcf/d drop in feedgas",
                start_date="2024-07-07",
                end_date="2024-07-24",
                affected_assets=["Gulf South Pipeline", "Texas feeders", "Freeport delivery points"],
                event_type="weather",
                causal_design="interrupted_time_series",
                treatment_suggestion="Gulf Coast LNG-adjacent meters",
                control_suggestion="Inland power plant meters",
                expected_impact="Temporary 2 Bcf/d drop, phased restart"
            ),
            Event(
                name="Freeport Lightning Interruption",
                description="Brief lightning-related feed interruption",
                start_date="2025-03-25",
                end_date="2025-03-25",
                affected_assets=["Gulf South Pipeline", "Freeport feedgas"],
                event_type="outage",
                causal_design="event_study",
                treatment_suggestion="Freeport feedgas meters",
                control_suggestion="Other LNG terminals",
                expected_impact="Short, sharp dip in feedgas"
            ),
            Event(
                name="Corpus Christi Stage 3 Ramp",
                description="New LNG train ramping from Dec 2024 to Feb 2025",
                start_date="2024-12-01",
                end_date="2025-02-28",
                affected_assets=["NGPL", "Tennessee Gas Pipeline", "Gulf Coast intrastates", "Corpus Christi meters"],
                event_type="expansion",
                causal_design="event_study",
                treatment_suggestion="Corpus-named meters",
                control_suggestion="Non-LNG meters on same pipelines",
                expected_impact="Rising feedgas into NGPL/Tennessee/Gulf Coast"
            ),

            # Permian Basin Expansions
            Event(
                name="Whistler Pipeline In-Service",
                description="0.5 Bcf/d capacity addition relieving Waha",
                start_date="2023-09-01",
                end_date=None,
                affected_assets=["Katy hub", "Waha hub", "Permian takeaway"],
                event_type="expansion",
                causal_design="synthetic_control",
                treatment_suggestion="Waha-adjacent deliveries",
                control_suggestion="Non-Permian hubs",
                expected_impact="Modest Waha basis improvement"
            ),
            Event(
                name="Permian Highway Pipeline",
                description="0.55 Bcf/d Permian takeaway capacity",
                start_date="2023-12-01",
                end_date=None,
                affected_assets=["Texas Eastern", "Tennessee Gas", "Gulf Coast markets"],
                event_type="expansion",
                causal_design="did",
                treatment_suggestion="Katy/Agua Dulce delivery points",
                control_suggestion="Midcontinent/Midwest pipes",
                expected_impact="Structural break in Permian flows"
            ),
            Event(
                name="Gulf Coast Express Pipeline",
                description="0.6 Bcf/d additional Permian takeaway",
                start_date="2023-12-01",
                end_date=None,
                affected_assets=["Gulf Coast markets", "Texas Eastern", "NGPL"],
                event_type="expansion",
                causal_design="did",
                treatment_suggestion="Texas Eastern/NGPL delivery points near Katy",
                control_suggestion="Midwest pipes",
                expected_impact="Coordinated capacity relief with Permian Highway"
            ),
            Event(
                name="Matterhorn Express",
                description="2.5 Bcf/d major Permian expansion",
                start_date="2024-10-01",
                end_date=None,
                affected_assets=["Katy/Wharton area", "Waha hub", "Gulf Coast markets"],
                event_type="expansion",
                causal_design="synthetic_control",
                treatment_suggestion="Waha-adjacent deliveries",
                control_suggestion="Midcontinent/Midwest pipes",
                expected_impact="Flow increases, narrowing negative Waha basis"
            ),
            Event(
                name="Verde Pipeline",
                description="1.0 Bcf/d Eagle Ford to Agua Dulce",
                start_date="2024-11-01",
                end_date=None,
                affected_assets=["Eagle Ford region", "Agua Dulce hub", "Gulf Coast LNG"],
                event_type="expansion",
                causal_design="did",
                treatment_suggestion="Agua Dulce delivery points",
                control_suggestion="Non-Eagle Ford supply regions",
                expected_impact="Enhanced Eagle Ford takeaway"
            ),

            # Appalachian Expansions
            Event(
                name="Mountain Valley Pipeline Launch",
                description="New 2.0 Bcf/d pipeline connecting Appalachia to Southeast",
                start_date="2024-06-14",
                end_date=None,
                affected_assets=["Transco Z5/6", "Columbia Gas Transmission", "Tennessee Gas"],
                event_type="expansion",
                causal_design="did",
                treatment_suggestion="Transco Zone 5/6 VA/NC deliveries",
                control_suggestion="Transco Zones 3-4 or unaffected states",
                expected_impact="Step-up in flows, first sustained high-flow in winter 2024-25"
            ),
            Event(
                name="Transco Regional Energy Access",
                description="2023-2024 ramps increasing NE takeaway",
                start_date="2023-01-01",
                end_date="2024-12-31",
                affected_assets=["Transco PA/NJ zones", "Northeast markets"],
                event_type="expansion",
                causal_design="interrupted_time_series",
                treatment_suggestion="PA/NJ delivery points",
                control_suggestion="Non-REA Transco zones",
                expected_impact="Smaller but visible increases into PA/NJ"
            ),

            # West Coast/Pacific Northwest
            Event(
                name="EPNG Line 2000 Return",
                description="Major pipeline return after long outage",
                start_date="2023-02-15",
                end_date=None,
                affected_assets=["El Paso Natural Gas", "SoCal/Desert SW markets"],
                event_type="restoration",
                causal_design="interrupted_time_series",
                treatment_suggestion="SoCal/Desert SW delivery points",
                control_suggestion="Non-EPNG West Coast supplies",
                expected_impact="Improved deliveries, price relief"
            ),
            Event(
                name="PNW Cold Spell & Jackson Prairie Outage",
                description="Storage outage during cold weather tightening supply",
                start_date="2024-01-13",
                end_date="2024-01-20",
                affected_assets=["Northwest Pipeline", "Jackson Prairie storage", "PNW markets"],
                event_type="weather",
                causal_design="did",
                treatment_suggestion="Northwest Pipeline scheduled quantities",
                control_suggestion="Northern Natural/NGPL",
                expected_impact="Supply tightening, storage-adjacent delivery impacts"
            ),

            # Infrastructure Incidents
            Event(
                name="Tennessee Gas Pipeline Station 860 Rupture",
                description="Major pipeline rupture with long restoration",
                start_date="2023-08-18",
                end_date="2024-03-29",
                affected_assets=["Tennessee Gas Pipeline", "Southeast markets"],
                event_type="outage",
                causal_design="interrupted_time_series",
                treatment_suggestion="TGP southbound flows",
                control_suggestion="Transco parallel routes",
                expected_impact="Suppressed southbound flows, rerouting to alternatives"
            ),
            Event(
                name="Columbia Gas Transmission VA Rupture",
                description="Pipeline rupture reducing Cove Point LNG flows",
                start_date="2023-07-25",
                end_date="2023-08-15",
                affected_assets=["Columbia Gas Transmission", "Cove Point LNG", "MD/VA meters"],
                event_type="outage",
                causal_design="did",
                treatment_suggestion="Columbia deliveries to MD/VA LNG meters",
                control_suggestion="Transco backfill routes",
                expected_impact="Temporary LNG feed reduction, alternative routing"
            ),

            # Weather Events
            Event(
                name="Winter Storm Elliott",
                description="Extreme cold causing record outages and fuel constraints",
                start_date="2022-12-23",
                end_date="2022-12-26",
                affected_assets=["Transco", "Texas Eastern", "Tennessee Gas", "Columbia"],
                event_type="weather",
                causal_design="event_study",
                treatment_suggestion="Counties in impacted ISO footprints",
                control_suggestion="West Coast counties (unaffected)",
                expected_impact="Broad demand spikes, supply-side freeze-offs"
            ),
            Event(
                name="January 2024 Arctic Blast",
                description=">15 Bcf/d production drop week-over-week",
                start_date="2024-01-14",
                end_date="2024-01-18",
                affected_assets=["Midwest/TX/Appalachia production", "All major interstate pipelines"],
                event_type="weather",
                causal_design="event_study",
                treatment_suggestion="Counties in impacted regions",
                control_suggestion="Unaffected West Coast/South",
                expected_impact="Synchronized receipt dips, delivery spikes"
            ),
            Event(
                name="Early January 2025 Cold Wave",
                description="Near-record total gas use with supply stress",
                start_date="2025-01-06",
                end_date="2025-01-10",
                affected_assets=["All major pipelines", "Storage systems", "LDC networks"],
                event_type="weather",
                causal_design="event_study",
                treatment_suggestion="High-demand states",
                control_suggestion="Mild weather regions",
                expected_impact="Peak demand, storage withdrawals, constraint activation"
            )
        ]

        return major_events

    def get_event_by_type(self, event_type: str) -> List[Event]:
        """Get all events of a specific type"""
        return [e for e in self.events if e.event_type == event_type]

    def get_recent_events(self, after_date: str = "2023-01-01") -> List[Event]:
        """Get events after a certain date"""
        return [e for e in self.events if e.start_date and e.start_date >= after_date]

    def suggest_analysis_for_event(self, event: Event) -> Dict[str, str]:
        """Generate analysis suggestion for an event"""
        return {
            "query": f"What was the causal impact of {event.name} on {event.affected_assets[0]} flows?",
            "method": event.causal_design,
            "treatment": event.treatment_suggestion,
            "control": event.control_suggestion,
            "window": f"[{event.start_date}, +60 days]" if event.start_date else "event window",
            "expected": event.expected_impact
        }