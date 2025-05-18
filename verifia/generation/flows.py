import json
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union

import yaml
import pandas as pd
from langchain_community.vectorstores import Chroma
from langgraph.types import Command

from .agents import create_dataframe_agent, create_retriever_agent
from .tools import (
    create_chroma_vectorstore_from_pdfs,
    create_chroma_vectorstore_from_string,
)
from verifia.utils.helpers.ds import read_data_file
from verifia.utils.helpers.io import read_yaml, save_yaml
from .prompts import (
    DOMAIN_GRAPH_QUERY,
    DOMAIN_VARIABLES_ANALYSER_SYSTEM_PROMPT,
    DOMAIN_VARIABLES_RETRIEVER_SYSTEM_PROMPT,
    DOMAIN_CONSTRAINTS_ANALYSER_SYSTEM_PROMPT,
    DOMAIN_CONSTRAINTS_RETRIEVER_SYSTEM_PROMPT,
    DOMAIN_RULES_ANALYSER_SYSTEM_PROMPT,
    DOMAIN_RULES_RETRIEVER_SYSTEM_PROMPT,
)
from .graphs import DomainGraphContext, build_domain_graph

# -----------------------------
# Module-level Constants
# -----------------------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860

# Configure module logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DomainGenFlow:
    """
    A class to generate a domain graph flow based on provided dataset, PDFs or vector database,
    and a model card.

    This class encapsulates the process of loading the necessary context (data, agents, and model card)
    to build and run a domain graph. It allows the context to be loaded from file paths or directly from
    provided objects. It orchestrates the domain graph generation, human interruptions, and final
    domain export.
    """

    def __init__(self) -> None:
        """
        Initialize the DomainGenFlow instance.
        """
        self.graph_context: Optional[DomainGraphContext] = None

    def load_ctx(
            self,
            data_fpath: Optional[Union[str, Path]] = None,
            dataframe: Optional[pd.DataFrame] = None,
            pdfs_dirpath: Optional[Union[str, Path]] = None,
            db_str_content: Optional[str] = None,
            vectordb: Optional[Chroma] = None,
            model_card_fpath: Optional[Union[str, Path]] = None,
            model_card: Optional[Dict[str, Any]] = None,
    ) -> "DomainGenFlow":
        """
        Load and initialize the domain context required to build the domain graph.

        At least one of 'data_fpath' or 'dataframe' must be provided, as well as one of
        'pdfs_dirpath', 'vectordb', or 'db_str_content'. Similarly, either 'model_card' or
        'model_card_fpath' must be provided.

        Args:
            data_fpath (Optional[Union[str, Path]]): Path to the CSV or data file.
            dataframe (Optional[pd.DataFrame]): DataFrame containing the data.
            pdfs_dirpath (Optional[Union[str, Path]]): Directory path containing PDF files.
            db_str_content (Optional[str]): String content to build a vector database.
            vectordb (Optional[Chroma]): Pre-initialized vector database.
            model_card_fpath (Optional[Union[str, Path]]): Path to the model card YAML file.
            model_card (Optional[Dict[str, Any]]): Model card as a dictionary.

        Returns:
            DomainGenFlow: The instance with the graph_context loaded.

        Raises:
            ValueError: If required context arguments are missing.
        """
        logger.info("Loading DomainGenFlow context.")
        if data_fpath is None and dataframe is None:
            raise ValueError(
                "You must provide either 'data_fpath' or 'dataframe' argument."
            )

        if pdfs_dirpath is None and vectordb is None and db_str_content is None:
            raise ValueError(
                "You must provide either 'pdfs_dirpath', 'vectordb', or 'db_str_content' argument."
            )

        if model_card is None and model_card_fpath is None:
            raise ValueError(
                "You must provide either 'model_card' or 'model_card_fpath' argument."
            )

        if dataframe is None:
            logger.info("Reading data from %s", data_fpath)
            dataframe = read_data_file(data_fpath)  # type: ignore

        logger.info("Creating DataFrame agents.")
        variables_df_agent = create_dataframe_agent(
            dataframe, DOMAIN_VARIABLES_ANALYSER_SYSTEM_PROMPT
        )
        constraints_df_agent = create_dataframe_agent(
            dataframe, DOMAIN_CONSTRAINTS_ANALYSER_SYSTEM_PROMPT
        )
        rules_df_agent = create_dataframe_agent(
            dataframe, DOMAIN_RULES_ANALYSER_SYSTEM_PROMPT
        )

        if vectordb is None:
            if pdfs_dirpath is not None:
                logger.info("Building vector store from PDFs at %s", pdfs_dirpath)
                vectordb = create_chroma_vectorstore_from_pdfs(pdfs_dirpath)
            elif db_str_content is not None:
                logger.info("Building vector store from string content.")
                vectordb = create_chroma_vectorstore_from_string(db_str_content)

        logger.info("Creating Retriever agents.")
        variables_retriever_agent = create_retriever_agent(
            vectordb, DOMAIN_VARIABLES_RETRIEVER_SYSTEM_PROMPT
        )
        constraints_retriever_agent = create_retriever_agent(
            vectordb, DOMAIN_CONSTRAINTS_RETRIEVER_SYSTEM_PROMPT
        )
        rules_retriever_agent = create_retriever_agent(
            vectordb, DOMAIN_RULES_RETRIEVER_SYSTEM_PROMPT
        )

        if model_card is None:
            logger.info("Reading model card from %s", model_card_fpath)
            model_card = read_yaml(model_card_fpath)  # type: ignore

        self.graph_context = DomainGraphContext(
            variables_df_agent,
            constraints_df_agent,
            rules_df_agent,
            variables_retriever_agent,
            constraints_retriever_agent,
            rules_retriever_agent,
            model_card,
        )
        logger.info("Context loaded successfully.")
        return self

    def _run(
            self,
            save: bool = False,
            local_path: Optional[Union[str, Path]] = None,
    ) -> Generator[Dict[str, Any], Any, None]:
        """
        Execute the domain graph, yielding on interrupts, then finalize and optionally save.

        Args:
            save: If True, save final domain spec to local_path.
            local_path: File path for saving when save=True.

        Yields:
            Events dicts containing "interrupt" or "complete" with payload.

        Raises:
            ValueError: If save=True but no local_path provided.
        """
        logger.info("Starting domain graph execution (save=%s).", save)
        compiled = build_domain_graph(self.graph_context)  # type: ignore
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        # initial state: inject the DOMAIN_GRAPH_QUERY to kick off graph
        state: Dict[str, Any] = {"messages": [("user", DOMAIN_GRAPH_QUERY)]}
        stream = compiled.stream(state, config=config)

        while True:
            for event in stream:
                if "__interrupt__" in event:
                    payload = event["__interrupt__"][0].value
                    yaml_spec = payload.get("yaml_spec", "")
                    val_out = payload.get("validator_output", "")
                    user_resp = yield {
                        "type": "interrupt",
                        "yaml": yaml_spec,
                        "val_out": val_out,
                    }
                    logger.info("User interrupt received, resuming graph.")
                    stream = compiled.stream(Command(resume=user_resp), config)
                    break
            else:
                logger.info("Graph run completed normally.")
                break

        final_state = compiled.get_state(config=config)
        values = getattr(final_state, "values", {})
        if "yaml_spec_obj" in values:
            domain_dict = values["yaml_spec_obj"].model_dump(mode="json")
        else:
            raw = values.get("yaml_spec_str", "")
            try:
                domain_dict = yaml.safe_load(raw) if isinstance(raw, str) else raw
            except Exception as e:
                logger.exception(f"Error parsing final YAML string.{str(e)}")
                domain_dict = {}

        if save:
            if local_path is None:
                logger.error("Save requested but local_path is missing.")
                raise ValueError("A valid local_path must be provided when save is True")
            logger.info("Saving domain spec to %s", local_path)
            save_yaml(local_path, domain_dict)

        yield {"type": "complete", "domain": domain_dict}

    @staticmethod
    def _yaml_from_event(ev: Dict[str, Any]) -> Tuple[str, str]:
        """
        Convert a stream interrupt event to YAML text and validation output.

        Args:
            ev: Event dict with raw payload under "yaml".

        Returns:
            Tuple of (yaml_text, validation_output).
        """
        raw = ev.get("yaml", "")
        try:
            parsed = raw if isinstance(raw, dict) else json.loads(raw)
            yaml_text = yaml.dump(parsed, sort_keys=False)
        except Exception as e:
            logger.warning(f"Failed to convert raw to YAML, using raw string. Exception: {str(e)}")
            yaml_text = str(raw)
        return yaml_text, ev.get("val_out", "")

    def start(self) -> Tuple[str, str, Generator[Dict[str, Any], Any, None]]:
        """
        Kick off the flow until first human interrupt, returning initial YAML and validation.

        Returns:
            init_yaml: Initial generated YAML spec.
            init_val: Initial validation output.
            gen: Generator to drive subsequent actions.
        """
        logger.info("Initiating generation flow.")
        gen = self._run(save=False)
        for ev in gen:
            if ev.get("type") == "interrupt":
                init_yaml, init_val = self._yaml_from_event(ev)
                return init_yaml, init_val, gen
        logger.info("No interrupt; generator ended with no output.")
        return "", "", gen

    def next_action(
            self,
            gen: Generator[Dict[str, Any], Any, None],
            action: str,
            yaml_str: str,
            instructions: str = "",
    ) -> Tuple[str, str, Generator[Dict[str, Any], Any, None]]:
        """
        Resume the flow based on a UI action (validate/regenerate/finish).

        Args:
            gen: Active generator from start or previous next_action.
            action: Action keyword.
            yaml_str: Current YAML shown to the user.
            instructions: User instructions for regeneration.

        Returns:
            Tuple of updated (yaml_text, validation_output, generator).
        """
        logger.info("Executing next_action: %s", action)
        try:
            spec_obj = yaml.safe_load(yaml_str)
        except Exception as e:
            logger.warning(f"YAML parsing failed. passing raw string. Exception: {str(e)}")
            spec_obj = yaml_str

        payload = {
            "action": action,
            "yaml_spec": spec_obj,
            "instructions": instructions
        }

        try:
            ev = gen.send(payload)
        except TypeError:
            ev = next(gen)

        while ev.get("type") not in ("interrupt", "complete"):
            ev = next(gen)

        if ev.get("type") == "interrupt":
            gen_yaml, val_out = self._yaml_from_event(ev)
            return gen_yaml, val_out, gen

        # On completion
        final_domain = ev.get("domain", {})
        try:
            final_yaml = yaml.dump(final_domain, sort_keys=False)
        except Exception as e:
            logger.exception(f"Failed to dump final domain dict. Exception: {str(e)}")
            final_yaml = str(final_domain)
        return final_yaml, "", gen

    def launch(
            self,
            server_name: str = DEFAULT_HOST,
            server_port: int = DEFAULT_PORT,
    ) -> None:
        """
        Launch the interactive Gradio UI tied to this flow.

        Args:
            server_name: Host/IP for the Gradio server.
            server_port: Port for the Gradio server.
        """
        logger.info("Launching Gradio UI at %s:%d", server_name, server_port)
        from verifia.generation.ui import build_ui

        app = build_ui(self)
        app.launch(server_name=server_name, server_port=server_port)
