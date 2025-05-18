import logging
from typing import Tuple, Any, Generator, Dict

import gradio as gr

from verifia.generation.flows import DomainGenFlow
from .constants import (
    LOGO_URL, APP_TITLE, YAML_PLACEHOLDER, VERIFIA_THEME, HEADER_TITLE,
    INSTRUCTIONS_TITLE, INSTRUCTIONS_PLACEHOLDER,
    CSS_PATH, LABEL_YAML, LABEL_VALIDATION, BUTTON_START,
    BUTTON_REGENERATE, BUTTON_VALIDATE, BUTTON_FINISH, FOOTER_HTML,
    YAML_LINES, YAML_MAX_LINES, INSTRUCTIONS_LINES, VALIDATION_LINES
)

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_ui(domain_flow: DomainGenFlow) -> gr.Blocks:
    """
    Build and return the Gradio Blocks app for DomainSpec generation.

    Args:
        domain_flow (DomainGenFlow): Orchestrator for domain specification generation.

    Returns:
        gr.Blocks: Configured Gradio Blocks application.
    """
    logger.info("Initializing Gradio UI...")

    # Load custom CSS
    try:
        css_text = CSS_PATH.read_text()
    except Exception as e:
        logger.error("Failed to load CSS from %s: %s", CSS_PATH, e)
        css_text = ""

    with gr.Blocks(
        title=APP_TITLE,
        theme=VERIFIA_THEME,
        css=css_text,
    ) as app:
        # Header row with logo and title
        with gr.Row(elem_classes="header-row"):
            with gr.Column(scale=1, elem_classes="logo-img"):
                gr.HTML(f"<img src='{LOGO_URL}' alt='VerifIA Logo'/>")
            with gr.Column(scale=2, elem_classes="title-col"):
                gr.Markdown(f"<h2 class='title-text'>{HEADER_TITLE}</h2>")
        gr.Markdown("---")

        # Main content: YAML view and controls
        with gr.Row():
            with gr.Column(scale=2, elem_classes="yaml-card"):
                yaml_area = gr.Code(
                    label=LABEL_YAML,
                    language="yaml",
                    interactive=False,
                    lines=YAML_LINES,
                    max_lines=YAML_MAX_LINES,
                    value=YAML_PLACEHOLDER,
                )

            with gr.Column(scale=1, elem_classes="controls-card"):
                with gr.Accordion(INSTRUCTIONS_TITLE, open=True):
                    instr_box = gr.Textbox(
                        label="How should I change the spec?",
                        placeholder=INSTRUCTIONS_PLACEHOLDER,
                        lines=INSTRUCTIONS_LINES,
                        interactive=True,
                    )

                validator_output = gr.TextArea(
                    label=LABEL_VALIDATION,
                    interactive=False,
                    lines=VALIDATION_LINES,
                    placeholder="Validation messages will appear here",
                )

                # Action buttons
                with gr.Row():
                    start_btn = gr.Button(BUTTON_START, variant="primary")
                    validate_btn = gr.Button(BUTTON_VALIDATE, interactive=False)
                with gr.Row():
                    regenerate_btn = gr.Button(BUTTON_REGENERATE, interactive=False)
                    finish_btn = gr.Button(BUTTON_FINISH, interactive=False)

        # Internal state storage for generator
        gen_state = gr.State()

        # ---------------------
        # Callback Definitions
        # ---------------------
        def on_start() -> Tuple[
            str, str, Generator[Any, None, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]
        ]:
            """
            Trigger the start of domain spec generation.

            Returns:
                Tuple containing YAML, validation text, generator object,
                and update dicts for validate, regenerate, finish buttons, and YAML area interactivity.
            """
            logger.info("Start button clicked.")
            yaml_str, val_out, gen = domain_flow.start()
            enabled = bool(yaml_str)
            logger.info(
                "Generation %s", "succeeded" if enabled else "failed"
            )
            update = gr.update(interactive=enabled)
            return yaml_str, val_out, gen, update, update, update, update

        start_btn.click(
            fn=on_start,
            outputs=[
                yaml_area,
                validator_output,
                gen_state,
                validate_btn,
                regenerate_btn,
                finish_btn,
                yaml_area,
            ],
        )

        def on_validate(
            gen: Generator[Any, None, Any],
            current_yaml: str,
        ) -> Tuple[
            str, str, Generator[Any, None, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]
        ]:
            """
            Validate the current YAML spec.

            Args:
                gen: Active generator object.
                current_yaml: User-edited YAML string.

            Returns:
                Updated YAML, validation output, updated generator,
                and update dicts for validate, regenerate, and finish.
            """
            logger.info("Validate button clicked.")
            yaml_str, val_out, new_gen = domain_flow.next_action(
                gen, "validate", current_yaml
            )
            enabled_update = gr.update(interactive=True)
            return (
                yaml_str,
                val_out,
                new_gen,
                enabled_update,
                enabled_update,
                enabled_update,
            )

        validate_btn.click(
            fn=on_validate,
            inputs=[gen_state, yaml_area],
            outputs=[
                yaml_area,
                validator_output,
                gen_state,
                validate_btn,
                regenerate_btn,
                finish_btn,
            ],
        )

        def on_regenerate(
            gen: Generator[Any, None, Any],
            current_yaml: str,
            instr: str,
        ) -> Tuple[
            str, str, Generator[Any, None, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]
        ]:
            """
            Regenerate the YAML spec based on user instructions.

            Args:
                gen: Active generator object.
                current_yaml: Last YAML string.
                instr: User instruction for regeneration.

            Returns:
                Updated YAML, validation messages, updated generator,
                and update dicts to re-enable buttons.
            """
            logger.info("Regenerate requested with instr=%r", instr)
            yaml_str, val_out, new_gen = domain_flow.next_action(
                gen, "regenerate", current_yaml, instr
            )
            enabled_update = gr.update(interactive=True)
            return (
                yaml_str,
                val_out,
                new_gen,
                enabled_update,
                enabled_update,
                enabled_update,
            )

        regenerate_btn.click(
            fn=on_regenerate,
            inputs=[gen_state, yaml_area, instr_box],
            outputs=[
                yaml_area,
                validator_output,
                gen_state,
                validate_btn,
                regenerate_btn,
                finish_btn,
            ],
        )

        def on_finish(
            gen: Generator[Any, None, Any],
            final_yaml: str,
        ) -> Tuple[
            str, str, Generator[Any, None, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]
        ]:
            """
            Finalize the generation process and disable all controls.

            Args:
                gen: Active generator object.
                final_yaml: Last YAML string.

            Returns:
                Final YAML, validation output, generator,
                and update dicts disabling all buttons and YAML area.
            """
            logger.info("Finish button clicked.")
            yaml_str, val_out, new_gen = domain_flow.next_action(
                gen, "finish", final_yaml
            )
            disabled_update = gr.update(interactive=False)
            return (
                yaml_str,
                val_out,
                new_gen,
                disabled_update,
                disabled_update,
                disabled_update,
                disabled_update,
            )

        finish_btn.click(
            fn=on_finish,
            inputs=[gen_state, yaml_area],
            outputs=[
                yaml_area,
                validator_output,
                gen_state,
                validate_btn,
                regenerate_btn,
                finish_btn,
                yaml_area,
            ],
        )

        # Footer section
        gr.Markdown("---")
        with gr.Row(elem_classes="footer"):
            gr.HTML(FOOTER_HTML)

    return app

