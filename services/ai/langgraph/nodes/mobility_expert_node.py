import json
import logging
from datetime import datetime

from services.ai.ai_settings import AgentRole
from services.ai.langgraph.schemas import MobilityExpertOutputs
from services.ai.langgraph.state.training_analysis_state import TrainingAnalysisState
from services.ai.langgraph.utils.message_helper import normalize_langchain_messages
from services.ai.model_config import ModelSelector
from services.ai.tools.plotting import PlotStorage
from services.ai.utils.retry_handler import AI_ANALYSIS_CONFIG, retry_with_backoff

from .node_base import (
    configure_node_tools,
    create_cost_entry,
    create_plot_entries,
    execute_node_with_error_handling,
    log_node_completion,
)
from .prompt_components import (
    get_hitl_instructions,
    get_plotting_instructions,
    get_workflow_context,
)
from .tool_calling_helper import handle_tool_calling_in_node

logger = logging.getLogger(__name__)

MOBILITY_SYSTEM_PROMPT_BASE = """## Goal
Optimize tissue quality, joint health, and CNS recovery through precise mobility/stretching protocols.
## Principles
- Specificity: Target tissues stressed in recent activities.
- Parasympathetic: Use mobility to down-regulate the nervous system.
- Actionable: Prescribe exact stretches (PNF, foam rolling, dynamic)."""

MOBILITY_USER_PROMPT = """## Task
Analyze the mobility summary (activities and stress) to prescribe recovery protocols.

## Constraints
- Focus on **mobility, stretching, and recovery modalities**.
- Do NOT rewrite the training plan (Planner's job).
- Address systemic stress (HRV/Sleep) with parasympathetic down-regulation.

## Inputs
### Mobility Summary
{data}
### Context
- Competitions: ```json {competitions} ```
- Date: ```json {current_date} ```
- **User Context**: ``` {analysis_context} ```

## Output Requirements
Produce 3 structured fields. For EACH field, use this internal layout:
- **Signals**: what changed (concise)
- **Evidence**: numbers + date ranges
- **Implications**: constraints/opportunities for this receiver
- **Uncertainty**: gaps/low coverage if any

**Important**: Tailor content for each consumer.

### 1. `for_synthesis` (Comprehensive Report)
- **Context**: Feeds the "Whole Athlete" view.
- **Goal**: Assess general movement quality and recovery needs.

### 2. `for_season_planner` (12-24 Weeks)
- **Context**: Informs Long-Term structural decisions.
- **Goal**: Highlight chronic tightness or injury risks that need addressing.

### 3. `for_weekly_planner` (Next 28 Days)
- **Context**: Acts as the exact recovery prescription for the next block.
- **Goal**: Prescribe specific foam rolling, stretching, or PNF routines to match the incoming training load.
"""

MOBILITY_FINAL_CHECKLIST = """
## Final Checklist
- Use Signals/Evidence/Implications/Uncertainty per receiver.
- Stay within mobility/recovery domain only.
- No training structure redesign.
"""

async def mobility_expert_node(state: TrainingAnalysisState) -> dict[str, list | str | dict]:
    logger.info("Starting mobility expert analysis node")

    plot_storage = PlotStorage(state["execution_id"])
    plotting_enabled = state.get("plotting_enabled", False)
    hitl_enabled = state.get("hitl_enabled", True)

    logger.info(
        "Mobility expert: Plotting %s, HITL %s",
        "enabled" if plotting_enabled else "disabled",
        "enabled" if hitl_enabled else "disabled",
    )

    tools = configure_node_tools(
        agent_name="mobility",
        plot_storage=plot_storage,
        plotting_enabled=plotting_enabled,
    )

    system_prompt = (
        get_workflow_context("mobility")
        + MOBILITY_SYSTEM_PROMPT_BASE
        + (get_plotting_instructions("mobility") if plotting_enabled else "")
        + (get_hitl_instructions("mobility") if hitl_enabled else "")
        + MOBILITY_FINAL_CHECKLIST
    )

    base_llm = ModelSelector.get_llm(AgentRole.MOBILITY_EXPERT)
    llm_with_tools = base_llm.bind_tools(tools) if tools else base_llm
    llm_with_structure = llm_with_tools.with_structured_output(MobilityExpertOutputs)

    agent_start_time = datetime.now()

    async def call_mobility_analysis():
        qa_messages = normalize_langchain_messages(state.get("mobility_expert_messages", []))

        base_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": MOBILITY_USER_PROMPT.format(
                    data=state.get("mobility_summary", "No mobility summary available"),
                    competitions=json.dumps(state["competitions"], indent=2),
                    current_date=json.dumps(state["current_date"], indent=2),
                    analysis_context=state["analysis_context"],
                ),
            },
        ]

        return await handle_tool_calling_in_node(
            llm_with_tools=llm_with_structure,
            messages=base_messages + qa_messages,
            tools=tools,
            max_iterations=15,
        )

    async def node_execution():
        agent_output = await retry_with_backoff(
            call_mobility_analysis, AI_ANALYSIS_CONFIG, "Mobility Expert with Tools"
        )

        execution_time = (datetime.now() - agent_start_time).total_seconds()
        plots, plot_storage_data, available_plots = create_plot_entries("mobility", plot_storage)

        log_node_completion("Mobility expert analysis", execution_time, len(available_plots))

        return {
            "mobility_outputs": agent_output,
            "plots": plots,
            "plot_storage_data": plot_storage_data,
            "costs": [create_cost_entry("mobility", execution_time)],
            "available_plots": available_plots,
        }

    return await execute_node_with_error_handling(
        node_name="Mobility expert analysis",
        node_function=node_execution,
        error_message_prefix="Mobility expert analysis failed",
    )
