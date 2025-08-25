"""
Agent Build Package

A modular, optimized agent workflow system using Pydantic models, decorators, and clean architecture.
"""

# Add path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Trimmed public API to only export modules that exist in this package
from agent_build.agents import (
    AgentState,
    material_hierarchy_resolver_agent,
    location_hierarchy_resolver_agent,
    p2p_spend_genie_agent,
    summary_agent,
    supervisor_agent,
)

from agent_build.config import default_config
from agent_build.utils import (
    call_get_material_hierarchy_level,
    call_get_location_hierarchy_level,
    parse_llm_extraction_output,
    build_combined_prompt,
)

__version__ = "1.0.0"
__all__ = [
    "AgentState",
    "material_hierarchy_resolver_agent",
    "location_hierarchy_resolver_agent",
    "p2p_spend_genie_agent",
    "summary_agent",
    "supervisor_agent",
    "default_config",
    "call_get_material_hierarchy_level",
    "call_get_location_hierarchy_level",
    "parse_llm_extraction_output",
    "build_combined_prompt",
]