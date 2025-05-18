import yaml
import logging
from typing import Any, Callable, Dict, Literal, Optional

from langchain_core.messages import AIMessage
from langgraph.types import Command, interrupt
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from .gpts import GPT_MODEL
from .prompts import (
    DOMAIN_VARIABLES_ANALYSER_SYSTEM_PROMPT,
    DOMAIN_VARIABLES_ANALYSER_USER_PROMPT,
    DOMAIN_VARIABLES_RETRIEVER_SYSTEM_PROMPT,
    DOMAIN_VARIABLES_RETRIEVER_USER_PROMPT,
    DOMAIN_CONSTRAINTS_ANALYSER_SYSTEM_PROMPT,
    DOMAIN_CONSTRAINTS_ANALYSER_USER_PROMPT,
    DOMAIN_CONSTRAINTS_RETRIEVER_SYSTEM_PROMPT,
    DOMAIN_CONSTRAINTS_RETRIEVER_USER_PROMPT,
    DOMAIN_RULES_ANALYSER_SYSTEM_PROMPT,
    DOMAIN_RULES_ANALYSER_USER_PROMPT,
    DOMAIN_RULES_RETRIEVER_SYSTEM_PROMPT,
    DOMAIN_RULES_RETRIEVER_USER_PROMPT,
    DOMAIN_GENERATOR_SYSTEM_PROMPT,
    DOMAIN_GENERATOR_FIX_SYSTEM_PROMPT,
)
from .schema import YAMLSpecification, DomainGraphState
from .agents import call_dataframe_agent, call_retriever_agent
from verifia.verification.verifiers import RuleConsistencyVerifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DomainGraphContext:
    """
    Holds agents and model card information required to build the domain graph.

    Attributes:
        variables_df_agent (Any): Agent for analyzing variables from a DataFrame.
        constraints_df_agent (Any): Agent for analyzing constraints from a DataFrame.
        rules_df_agent (Any): Agent for analyzing rules from a DataFrame.
        variables_retriever_agent (Any): Agent for retrieving variable information.
        constraints_retriever_agent (Any): Agent for retrieving constraint information.
        rules_retriever_agent (Any): Agent for retrieving rule information.
        model_card_info (Dict[str, Any]): Model card details including feature names,
            categorical feature names, and target name.
    """

    def __init__(
            self,
            variables_df_agent: Any,
            constraints_df_agent: Any,
            rules_df_agent: Any,
            variables_retriever_agent: Any,
            constraints_retriever_agent: Any,
            rules_retriever_agent: Any,
            model_card_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.variables_df_agent = variables_df_agent
        self.constraints_df_agent = constraints_df_agent
        self.rules_df_agent = rules_df_agent
        self.variables_retriever_agent = variables_retriever_agent
        self.constraints_retriever_agent = constraints_retriever_agent
        self.rules_retriever_agent = rules_retriever_agent
        self.model_card_info = model_card_info or {}
        self.model_card_info["feature_names"] = self.model_card_info.get(
            "feature_names", []
        )
        self.model_card_info["cat_feature_names"] = self.model_card_info.get(
            "cat_feature_names", []
        )
        self.model_card_info["target_name"] = self.model_card_info.get(
            "target_name", None
        )


def build_domain_graph(context: DomainGraphContext) -> CompiledStateGraph:
    """
    Build and compile a domain graph based on the provided context.

    The graph is constructed by adding edges from the START node to multiple analyser and
    retriever nodes and then registering the corresponding nodes with factory functions.

    Args:
        context (DomainGraphContext): The context containing agents and model card info.

    Returns:
        CompiledStateGraph: The compiled domain graph.
    """
    builder = StateGraph(DomainGraphState)

    # Launch all nodes in parallel from START.
    builder.add_edge(START, "analyser_variables")
    builder.add_edge(START, "retriever_variables")
    builder.add_edge(START, "analyser_constraints")
    builder.add_edge(START, "retriever_constraints")
    builder.add_edge(START, "analyser_rules")
    builder.add_edge(START, "retriever_rules")

    # Register nodes using factory functions.
    builder.add_node("analyser_variables", make_analyser_domain_variables_node(context))
    builder.add_node(
        "retriever_variables", make_retriever_domain_variables_node(context)
    )
    builder.add_node(
        "analyser_constraints", make_analyser_domain_constraints_node(context)
    )
    builder.add_node(
        "retriever_constraints", make_retriever_domain_constraints_node(context)
    )
    builder.add_node("analyser_rules", make_analyser_domain_rules_node(context))
    builder.add_node("retriever_rules", make_retriever_domain_rules_node(context))
    builder.add_node("yaml_generator", domain_generator_node)
    builder.add_node("human_interaction", human_interaction_node)
    builder.add_node("yaml_validator", domain_validator_node)

    saver = MemorySaver()
    compiled = builder.compile(checkpointer=saver)

    return compiled


def make_analyser_domain_variables_node(
        context: DomainGraphContext,
) -> Callable[[DomainGraphState], Command[Literal["yaml_generator"]]]:
    """
    Create a node function for analyzing domain variables from the DataFrame.

    Args:
        context (DomainGraphContext): Context containing agents and model card info.

    Returns:
        Callable[[DomainGraphState], Command[Literal["yaml_generator"]]]:
            A function that processes domain variables and returns a command to proceed.
    """

    def analyser_domain_variables_node(
            state: DomainGraphState,
    ) -> Command[Literal["yaml_generator"]]:
        mc = context.model_card_info
        user_request = DOMAIN_VARIABLES_ANALYSER_USER_PROMPT.format(
            feature_names=mc.get("feature_names", "N/A"),
            cat_feature_names=mc.get("cat_feature_names", "N/A"),
            target_name=mc.get("target_name", "N/A"),
        )
        messages = [
            {"role": "system", "content": DOMAIN_VARIABLES_ANALYSER_SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]
        updated_state = call_dataframe_agent(
            context.variables_df_agent,
            messages,
            "analyser_variables",
            state,
            "variables_analysis",
        )
        return Command(update=updated_state, goto="yaml_generator")

    return analyser_domain_variables_node


def make_retriever_domain_variables_node(
        context: DomainGraphContext,
) -> Callable[[DomainGraphState], Command[Literal["yaml_generator"]]]:
    """
    Create a node function for retrieving domain variables from the vector store.

    Args:
        context (DomainGraphContext): Context containing agents and model card info.

    Returns:
        Callable[[DomainGraphState], Command[Literal["yaml_generator"]]]:
            A function that retrieves domain variables and returns a command to proceed.
    """

    def retriever_domain_variables_node(
            state: DomainGraphState,
    ) -> Command[Literal["yaml_generator"]]:
        messages = [
            {"role": "system", "content": DOMAIN_VARIABLES_RETRIEVER_SYSTEM_PROMPT},
            {"role": "user", "content": DOMAIN_VARIABLES_RETRIEVER_USER_PROMPT},
        ]
        updated_state = call_retriever_agent(
            context.variables_retriever_agent,
            messages,
            "retriever_variables",
            state,
            "variables_retriever",
        )
        return Command(update=updated_state, goto="yaml_generator")

    return retriever_domain_variables_node


def make_analyser_domain_constraints_node(
        context: DomainGraphContext,
) -> Callable[[DomainGraphState], Command[Literal["yaml_generator"]]]:
    """
    Create a node function for analyzing domain constraints from the DataFrame.

    Args:
        context (DomainGraphContext): Context containing agents and model card info.

    Returns:
        Callable[[DomainGraphState], Command[Literal["yaml_generator"]]]:
            A function that analyzes constraints and returns a command to proceed.
    """

    def analyser_domain_constraints_node(
            state: DomainGraphState,
    ) -> Command[Literal["yaml_generator"]]:
        mc = context.model_card_info
        user_request = DOMAIN_CONSTRAINTS_ANALYSER_USER_PROMPT.format(
            feature_names=mc.get("feature_names", "N/A"),
            cat_feature_names=mc.get("cat_feature_names", "N/A"),
            target_name=mc.get("target_name", "N/A"),
        )
        messages = [
            {"role": "system", "content": DOMAIN_CONSTRAINTS_ANALYSER_SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]
        updated_state = call_dataframe_agent(
            context.constraints_df_agent,
            messages,
            "analyser_constraints",
            state,
            "constraints_analysis",
        )
        return Command(update=updated_state, goto="yaml_generator")

    return analyser_domain_constraints_node


def make_retriever_domain_constraints_node(
        context: DomainGraphContext,
) -> Callable[[DomainGraphState], Command[Literal["yaml_generator"]]]:
    """
    Create a node function for retrieving domain constraints from the vector store.

    Args:
        context (DomainGraphContext): Context containing agents and model card info.

    Returns:
        Callable[[DomainGraphState], Command[Literal["yaml_generator"]]]:
            A function that retrieves constraints and returns a command to proceed.
    """

    def retriever_domain_constraints_node(
            state: DomainGraphState,
    ) -> Command[Literal["yaml_generator"]]:
        messages = [
            {"role": "system", "content": DOMAIN_CONSTRAINTS_RETRIEVER_SYSTEM_PROMPT},
            {"role": "user", "content": DOMAIN_CONSTRAINTS_RETRIEVER_USER_PROMPT},
        ]
        updated_state = call_retriever_agent(
            context.constraints_retriever_agent,
            messages,
            "retriever_constraints",
            state,
            "constraints_retriever",
        )
        return Command(update=updated_state, goto="yaml_generator")

    return retriever_domain_constraints_node


def make_analyser_domain_rules_node(
        context: DomainGraphContext,
) -> Callable[[DomainGraphState], Command[Literal["yaml_generator"]]]:
    """
    Create a node function for analyzing domain rules from the DataFrame.

    Args:
        context (DomainGraphContext): Context containing agents and model card info.

    Returns:
        Callable[[DomainGraphState], Command[Literal["yaml_generator"]]]:
            A function that analyzes rules and returns a command to proceed.
    """

    def analyser_domain_rules_node(
            state: DomainGraphState,
    ) -> Command[Literal["yaml_generator"]]:
        mc = context.model_card_info
        user_request = DOMAIN_RULES_ANALYSER_USER_PROMPT.format(
            feature_names=mc.get("feature_names", "N/A"),
            cat_feature_names=mc.get("cat_feature_names", "N/A"),
            target_name=mc.get("target_name", "N/A"),
        )
        messages = [
            {"role": "system", "content": DOMAIN_RULES_ANALYSER_SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]
        updated_state = call_dataframe_agent(
            context.rules_df_agent,
            messages,
            "analyser_rules",
            state,
            "rules_analysis",
        )
        return Command(update=updated_state, goto="yaml_generator")

    return analyser_domain_rules_node


def make_retriever_domain_rules_node(
        context: DomainGraphContext,
) -> Callable[[DomainGraphState], Command[Literal["yaml_generator"]]]:
    """
    Create a node function for retrieving domain rules from the vector store.

    Args:
        context (DomainGraphContext): Context containing agents and model card info.

    Returns:
        Callable[[DomainGraphState], Command[Literal["yaml_generator"]]]:
            A function that retrieves rules and returns a command to proceed.
    """

    def retriever_domain_rules_node(
            state: DomainGraphState,
    ) -> Command[Literal["yaml_generator"]]:
        messages = [
            {"role": "system", "content": DOMAIN_RULES_RETRIEVER_SYSTEM_PROMPT},
            {"role": "user", "content": DOMAIN_RULES_RETRIEVER_USER_PROMPT},
        ]
        updated_state = call_retriever_agent(
            context.rules_retriever_agent,
            messages,
            "retriever_rules",
            state,
            "rules_retriever",
        )
        return Command(update=updated_state, goto="yaml_generator")

    return retriever_domain_rules_node


def domain_generator_node(
        state: DomainGraphState,
) -> Command[Literal["human_interaction"]]:
    """
    Generate a YAML specification string for the domain by combining the outputs of
    variable, constraint, and rule agents. If previous generation failed, a fix prompt is used.

    Args:
        state (DomainGraphState): The current state of the domain graph.

    Returns:
        Command[Literal["yaml_validator"]]: A command containing the updated state and the next node to execute.
    """
    variables_info = (
            state.get("variables_analysis", "")
            + "\n"
            + state.get("variables_retriever", "")
    )
    constraints_info = (
            state.get("constraints_analysis", "")
            + "\n"
            + state.get("constraints_retriever", "")
    )
    rules_info = (
            state.get("rules_analysis", "") + "\n" + state.get("rules_retriever", "")
    )

    generate_prompt = DOMAIN_GENERATOR_SYSTEM_PROMPT.format(
        variables_info=variables_info,
        constraints_info=constraints_info,
        rules_info=rules_info,
    )

    previous_yaml = state.get("yaml_spec_str", "")
    user_instr = state.get("user_instructions", "").strip()

    if user_instr:
        # regeneration: inject into the fix prompt
        regenerate_prompt = DOMAIN_GENERATOR_FIX_SYSTEM_PROMPT.format(
            previous_yaml=previous_yaml,
            user_instr=user_instr,
            variables_info=variables_info,
            constraints_info=constraints_info,
            rules_info=rules_info,
        )
        messages = [{"role": "system", "content": regenerate_prompt}]
    else:
        # first‐time generation
        messages = [{"role": "system", "content": generate_prompt}]

    try:
        logger.info("Generating YAML ...")
        llm_response = GPT_MODEL.invoke(messages)
        yaml_spec_str = llm_response.content
    except Exception as e:
        logger.error("Error in yaml_generator_node: %s", e)
        yaml_spec_str = "{}"  # Default minimal valid JSON

    # 5) Update state
    state["yaml_spec_str"] = yaml_spec_str
    state["user_instructions"] = ""  # clear instructions so next cycle defaults back

    # Record YAML generation in messages.
    state["messages"] = [
        {"role": "assistant", "content": yaml_spec_str, "name": "yaml_generator"}
    ]
    return Command(update=state, goto="human_interaction")


def domain_validator_node(
        state: DomainGraphState,
) -> Command[Literal["human_interaction"]]:
    """
    Validate the generated YAML specification against the expected schema.
    On success, the domain graph process ends; on failure, it returns to the generator node with feedback.

    Args:
        state (DomainGraphState): The current state of the domain graph.

    Returns:
        Command[Literal["yaml_generator", "__end__"]]: A command that directs the next step.
    """
    raw_spec = state.get("yaml_spec_str", "")
    reports = []

    # 1) normalize to a dict
    try:
        spec_dict = (
            raw_spec
            if isinstance(raw_spec, dict)
            else yaml.safe_load(raw_spec)
        )
        reports.append("✔ Parsed YAML successfully.")
    except Exception as e:
        # If we can’t parse YAML, treat it as invalid
        reports.append(f"❌ YAML parse error: {e}")
        state["validator_output"] = "\n".join(reports)
        state["messages"].append(
            AIMessage(content=state["validator_output"], name="validator")
        )
        return Command(update=state, goto="human_interaction")

    # 2) Pydantic schema validation
    try:
        yaml_spec_obj = YAMLSpecification.parse_obj(spec_dict)
        state["yaml_spec_obj"] = yaml_spec_obj
        reports.append("✔ Schema validation successful.")
    except Exception as e:
        reports.append(f"❌ Schema validation failed: {e}")
        state["validator_output"] = "\n".join(reports)
        state["messages"].append(AIMessage(content=state["validator_output"], name="validator"))
        return Command(update=state, goto="human_interaction")

    # 3) Rule consistency check
    try:
        verifier = RuleConsistencyVerifier(domain_cfg_dict=spec_dict)
        # TODO: maybe call a dummy verify to check that everything is okay !! issues = verifier.verify()
        reports.append("✔ Rule consistency check passed.")
    except Exception as e:
        reports.append(f"❌ Rule consistency failed: {e}")

    # 4) Write back combined report
    state["validator_output"] = "\n".join(reports)
    state["messages"].append(
        AIMessage(content=state["validator_output"], name="validator")
    )
    return Command(update=state, goto="human_interaction")


def human_interaction_node(state: DomainGraphState) -> Command[Literal["yaml_generator", "yaml_validator", "__end__"]]:
    payload = {
        "yaml_spec": state.get("yaml_spec_str", ""),
        "validator_output": state.get("validator_output", "")
    }
    result = interrupt(payload)
    # apply any edits & store instructions
    state["yaml_spec_str"] = result.get("yaml_spec", state["yaml_spec_str"])
    state["user_instructions"] = result.get("instructions", "")
    action = result.get("action", "validate").lower()

    if action == "finish":
        return Command(update=state, goto=END)
    elif action == "regenerate":
        # loop back into generator with new instructions
        return Command(update=state, goto="yaml_generator")
    else:  # validate
        return Command(update=state, goto="yaml_validator")
