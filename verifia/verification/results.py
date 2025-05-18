import os
import shutil
import logging
from pathlib import Path
from itertools import chain
from collections import Counter
from datetime import datetime

from dataclasses import dataclass, asdict
from typing import Dict, List, Union, Optional, Any

from fasthtml.common import *
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import IFrame, display
from ..utils import ROUNDING_PRECISION, CHECKPOINTS_DIRPATH
from ..utils.helpers.io import mk_tmpdir, rm_tmpdir, save_json, read_json
from ..utils.helpers.sys import generate_short_uuid, generate_timestamp
from ..utils.helpers.math import compute_percentage
from ..utils.helpers.ds import read_data_file, feat_vect2dict, remove_duplicates
from ..mlops import get_configured_mlops_platform
from ..mlops import SupportedMLOpsPlatform
from ..mlops import comet_ml, mlflow, wandb
from ..models import SupportedModelTypes, ModelCard, BaseModel
from ..context.domain import Domain, Rule, Constraint

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class OriginalStatistics:
    """
    Stores statistical metrics related to original data and their computed quality metrics.
    """

    n_orig: Optional[int] = None
    n_ood: Optional[int] = None
    n_herr: Optional[int] = None
    err_thresh: Optional[float] = None
    n_miscls: Optional[int] = None
    metric_name: Optional[str] = None
    metric_score: Optional[float] = None

    def round_floating_values(self) -> None:
        """
        Rounds metric_score and err_thresh to a precision based on ROUNDING_PRECISION.
        """
        if self.metric_score is not None:
            # Rounds metric_score using double the base rounding precision.
            self.metric_score = round(self.metric_score, 2 * ROUNDING_PRECISION)
        if self.err_thresh is not None:
            self.err_thresh = round(self.err_thresh, 2 * ROUNDING_PRECISION)


@dataclass
class RunMetadata:
    """
    Stores metadata for a verification run.
    """

    run_id: Optional[str] = None
    run_ts: Optional[int] = None
    run_ckpt_dirpath: Optional[str] = None
    rules_names: List[str] = None
    constraints_names: List[str] = None
    search_algo: Optional[str] = None
    search_params: Dict = None
    pop_size: Optional[int] = None
    max_iters: Optional[int] = None

    def random(self) -> "RunMetadata":
        """
        Assigns random values to run_id and run_ts using helper functions, and returns self.
        """
        self.run_id = generate_short_uuid()
        self.run_ts = generate_timestamp()
        return self


class RulesViolationRun:
    """
    Manages a verification run by storing original data, inconsistent data, and generating
    reports (HTML and JSON) for rules violations and constraint satisfactions.
    """

    def __init__(
        self,
        model_card: ModelCard,
        domain: Domain,
        metadata: Optional[RunMetadata] = None,
    ) -> None:
        """
        Initialize a run instance with given model card, domain, and optional metadata.
        If metadata is not provided, a new RunMetadata is created with random values and
        associated paths and names are set.
        """
        self.model_card: ModelCard = model_card
        if metadata:
            self.metadata = metadata
            self.rules: List[Rule] = [
                rule for rule in domain.rules if rule.name in metadata.rules_names
            ]
            self.constraints: List[Constraint] = [
                cons
                for cons in domain.constraints
                if cons.name in metadata.constraints_names
            ]
        else:
            self.metadata = RunMetadata().random()
            self.rules = domain.rules
            self.constraints = domain.constraints
            result_dirpath = Path(CHECKPOINTS_DIRPATH) / self.model_card.full_name
            self.metadata.run_ckpt_dirpath = str(
                result_dirpath / f"run_{self.metadata.run_id}"
            )
            self.metadata.rules_names = [rule.name for rule in domain.rules]
            self.metadata.constraints_names = [cons.name for cons in domain.constraints]
        self.orig_data: List[Dict[str, Any]] = []
        self.inconsistent_data: Dict[str, List[Dict[str, Any]]] = {}
        self.curr_rule_id: Optional[str] = None
        self.orig_stats: Optional[OriginalStatistics] = None
        self.constraints_stats: Dict[str, int] = {}
        self.duplicated_count: int = 0

    def set_search_params(
        self, search_algo: str, search_params: dict, pop_size: int, max_iters: int
    ) -> None:
        """
        Set search algorithm parameters for the verification process.
        """
        self.metadata.search_algo = search_algo
        self.metadata.search_params = search_params
        self.metadata.pop_size = pop_size
        self.metadata.max_iters = max_iters

    def set_orig_stats(self, orig_stats: OriginalStatistics) -> None:
        """
        Set the original statistics and round floating point values.
        """
        self.orig_stats = orig_stats
        self.orig_stats.round_floating_values()

    def update_constraints_stats(self, constraints_dict: Dict[str, int]) -> None:
        """
        Update the constraint statistics with new counts.
        """
        for key, value in constraints_dict.items():
            self.constraints_stats[key] = self.constraints_stats.get(key, 0) + value

    def create_diff_entry(
        self, base_features: np.ndarray, input_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create a dictionary showing differences between base and input features.

        Args:
            base_features (np.ndarray): The base feature vector.
            input_features (np.ndarray): The modified feature vector.

        Returns:
            Dict[str, Any]: A mapping of feature names to their (possibly diffed) values.
        """
        diff_entry: Dict[str, Any] = {}
        base_dict = feat_vect2dict(self.model_card.feature_names, base_features)
        input_dict = feat_vect2dict(self.model_card.feature_names, input_features)
        for fname in self.model_card.feature_names:
            base_val = base_dict[fname]
            input_val = input_dict[fname]
            if isinstance(input_val, str):
                diff_entry[fname] = (
                    f"{input_val} ({base_val})" if input_val != base_val else input_val
                )
            else:
                diff = np.round(input_val - base_val, ROUNDING_PRECISION)
                rounded_input_val = np.round(input_val, ROUNDING_PRECISION)
                diff_entry[fname] = (
                    f"{rounded_input_val} ({diff:+})"
                    if diff != 0.0
                    else rounded_input_val
                )
        return diff_entry

    def add_original(self, input_features: np.ndarray, predict_value: float) -> str:
        """
        Record an original input entry with its predicted value.

        Args:
            input_features (np.ndarray): The feature vector.
            predict_value (float): The predicted value.

        Returns:
            str: A unique identifier for the recorded entry.
        """
        entry_id = generate_short_uuid()
        orig_entry: Dict[str, Any] = {"id": entry_id}
        orig_entry.update(feat_vect2dict(self.model_card.feature_names, input_features))
        orig_entry[self.model_card.target_name] = predict_value
        self.orig_data.append(orig_entry)
        return entry_id

    def add_inconsistent(
        self,
        base_features: np.ndarray,
        inputs_features: np.ndarray,
        predict_values: np.ndarray,
        parent_id: str,
        base_value: float,
        deviations: np.ndarray,
    ) -> None:
        """
        Record inconsistent entries based on the differences between base and input features.

        Args:
            base_features (np.ndarray): The base feature vector.
            inputs_features (np.ndarray): The modified feature vectors.
            predict_values (np.ndarray): Predicted values corresponding to each modified input.
            parent_id (str): Identifier of the original input.
            base_value (float): Predicted value for the base input.
            deviations (np.ndarray): Deviation values for each modified input.
        """
        for i in range(inputs_features.shape[0]):
            entry_id = generate_short_uuid()
            inconsistent_entry: Dict[str, Any] = {"id": entry_id}
            inconsistent_entry.update(
                self.create_diff_entry(base_features, inputs_features[i])
            )
            diff = np.round(
                predict_values[i] - base_value, ROUNDING_PRECISION
            ).squeeze()
            rounded_predict_value = np.round(predict_values[i], ROUNDING_PRECISION)
            target_out = (
                f"{rounded_predict_value} ({diff:+})"
                if diff != 0.0
                else f"{rounded_predict_value}"
            )
            inconsistent_entry.update(
                {
                    self.model_card.target_name: target_out,
                    "parent_id": parent_id,
                    "deviation": deviations[i],
                }
            )
            self.inconsistent_data.setdefault(self.curr_rule_id, []).append(
                inconsistent_entry
            )

    def clean_duplicated_violations(self, tol: float = 1e-6) -> None:
        """
        Remove duplicate inconsistent entries using a tolerance threshold.
        Updates the duplicated count accordingly.
        """
        for rid in list(self.inconsistent_data.keys()):
            clean_data = remove_duplicates(
                self.inconsistent_data[rid], self.model_card.feature_names, tol
            )
            self.duplicated_count += len(self.inconsistent_data[rid]) - len(clean_data)
            self.inconsistent_data[rid] = clean_data

    @property
    def rules_count(self) -> int:
        """Return the number of rules considered."""
        return len(self.rules)

    @property
    def originals_count(self) -> int:
        """Return the count of original inputs recorded."""
        return len(self.orig_data)

    @property
    def inconsistency_revealing_originals_percentage(self) -> int:
        """
        Calculate the percentage of original inputs that reveal inconsistencies.

        Returns:
            int: Percentage value.
        """
        if self.originals_count == 0:
            return 0
        unique_parents = len(
            {
                entry["parent_id"]
                for entries in self.inconsistent_data.values()
                for entry in entries
            }
        )
        return compute_percentage(unique_parents, self.originals_count)

    @property
    def total_search_evaluations(self) -> int:
        """Return the total number of search evaluations performed."""
        return self.originals_count * self.metadata.pop_size * self.metadata.max_iters

    @property
    def violated_rules_count(self) -> int:
        """Return the count of rules that have any inconsistent entries."""
        return sum(1 for entries in self.inconsistent_data.values() if entries)

    @property
    def violated_rules_percentage(self) -> float:
        """Return the percentage of rules violated."""
        return compute_percentage(self.violated_rules_count, self.rules_count)

    @property
    def verifications_count_per_original(self) -> int:
        """Return the number of verifications per original input."""
        return self.rules_count * self.metadata.pop_size * self.metadata.max_iters

    @property
    def verifications_count(self) -> int:
        """Return the total number of verifications performed."""
        return self.rules_count * self.total_search_evaluations

    @property
    def duplicated_inconsistent_inputs_percentage(self) -> float:
        """Return the percentage of duplicated inconsistent inputs removed."""
        return compute_percentage(self.duplicated_count, self.verifications_count)

    @property
    def inconsistent_inputs_percentage(self) -> float:
        """Return the percentage of inconsistent inputs relative to total verifications."""
        if self.verifications_count == 0:
            return 0
        total_inconsistent = sum(
            len(entries) for entries in self.inconsistent_data.values()
        )
        return compute_percentage(total_inconsistent, self.verifications_count)

    @property
    def avg_inconsistent_inputs_percentage_per_original(self) -> float:
        """
        Calculate the average percentage of inconsistent inputs per original input.

        Returns:
            float: Average percentage.
        """
        if self.verifications_count_per_original == 0:
            return 0
        orig_stats: Dict[str, int] = {}
        for entries in self.inconsistent_data.values():
            for entry in entries:
                pid = entry["parent_id"]
                orig_stats[pid] = orig_stats.get(pid, 0) + 1
        avg_inconsistent = np.mean(list(orig_stats.values()))
        return compute_percentage(
            avg_inconsistent, self.verifications_count_per_original
        )

    @property
    def total_constraints(self) -> int:
        """Return the total number of constraints with stats."""
        return len(self.constraints_stats)

    @property
    def unsatisfied_constraints_count(self) -> int:
        """Return the count of constraints that are unsatisfied."""
        if self.total_constraints == 0:
            return 0
        return sum(1 for count in self.constraints_stats.values() if count > 0)

    @property
    def unsatisfied_constraints_percentage(self) -> float:
        """Return the percentage of unsatisfied constraints."""
        if self.total_constraints == 0:
            return 0
        return compute_percentage(
            self.unsatisfied_constraints_count, self.total_constraints
        )

    @property
    def infeasible_inputs_count(self) -> int:
        """Return the total count of infeasible inputs across all constraints."""
        if self.total_constraints == 0:
            return 0
        return sum(self.constraints_stats.values())

    @property
    def infeasible_inputs_percentage(self) -> float:
        """Return the percentage of infeasible inputs relative to total verifications."""
        if self.verifications_count == 0:
            return 0
        return compute_percentage(
            self.infeasible_inputs_count, self.verifications_count
        )

    def infeasible_inputs_percentage_per_constraint(self, cid: str) -> float:
        """
        Calculate the infeasible inputs percentage for a given constraint.

        Args:
            cid (str): Constraint identifier.

        Returns:
            float: Infeasible inputs percentage.
        """
        if self.verifications_count == 0:
            return 0
        count = self.constraints_stats.get(cid, 0)
        return compute_percentage(count, self.verifications_count)

    def _generate_run_verif_overview_html(self) -> Any:
        """
        Generate the HTML overview section for the run.

        Returns:
            HTML content representing the run verification overview.
        """
        if self.model_card.model_type == SupportedModelTypes.CLASSIFICATION:
            perf_section = P(
                Strong("Misclassified Inputs Count: "), str(self.orig_stats.n_miscls)
            )
        elif self.model_card.model_type == SupportedModelTypes.REGRESSION:
            perf_section = P(
                Strong(f"High-Error Inputs Count (>= {self.orig_stats.err_thresh}): "),
                str(self.orig_stats.n_herr),
            )
        else:
            perf_section = P("No performance data available.")

        algo_params = (
            "; ".join(f"{k}={v}" for k, v in self.metadata.search_params.items())
            if self.metadata.search_params
            else "Default Values"
        )

        html_section = Div(
            Div(
                H5("Search Algorithm Configuration"),
                P(Strong("Algorithm: "), str(self.metadata.search_algo)),
                P(Strong("Population Size: "), str(self.metadata.pop_size)),
                P(Strong("Max Iterations: "), str(self.metadata.max_iters)),
                P(Strong("Algorithm-Specific Parameters: "), algo_params),
                Hr(),
                H5("Verification Data Overview"),
                P(Strong("Total Sampled Inputs: "), str(self.orig_stats.n_orig)),
                P(
                    Strong(f"{self.orig_stats.metric_name}: "),
                    str(self.orig_stats.metric_score),
                ),
                P(Strong("Out-of-Domain Inputs: "), str(self.orig_stats.n_ood)),
                perf_section,
                P(Strong("Seed Inputs Count: "), str(self.originals_count)),
                P(Strong("Total Rules: "), str(self.rules_count)),
                P(
                    Strong("Verifications per Seed Input: "),
                    str(self.verifications_count_per_original),
                ),
                P(
                    Strong("Total Generated Verifications: "),
                    str(self.verifications_count),
                ),
                cls="col-md-6",
            ),
            Div(
                H5("Rule Violation Metrics"),
                P(
                    Strong("Count of Violated Rules: "),
                    f"{self.violated_rules_count} ({self.violated_rules_percentage}%)",
                ),
                P(
                    Strong("Percentage of Duplicated Inconsistent Inputs Removed: "),
                    f"{self.duplicated_inconsistent_inputs_percentage}%",
                ),
                P(
                    Strong("Overall Inconsistent Inputs Percentage: "),
                    f"{self.inconsistent_inputs_percentage}%",
                ),
                P(
                    Strong("Inconsistency-Revealing Seed Inputs Percentage: "),
                    f"{self.inconsistency_revealing_originals_percentage}%",
                ),
                P(
                    Strong("Average Inconsistency Rate per Seed Input: "),
                    f"{self.avg_inconsistent_inputs_percentage_per_original}%",
                ),
                Hr(),
                H5("Constraint Satisfaction Metrics"),
                P(
                    Strong("Unsatisfied Constraints: Count (Percentage): "),
                    f"{self.unsatisfied_constraints_count} ({self.unsatisfied_constraints_percentage}%)",
                ),
                P(
                    Strong("Infeasible Inputs Percentage: "),
                    f"{self.infeasible_inputs_percentage}%",
                ),
                cls="col-md-6",
            ),
            cls="row",
        )
        return html_section

    def _generate_run_verif_overview_dict(self) -> Dict[str, Any]:
        """
        Generate a dictionary overview of the verification run.

        Returns:
            Dict[str, Any]: Overview details including search configuration and metrics.
        """
        if self.model_card.model_type == SupportedModelTypes.CLASSIFICATION:
            field_name = "misclassified_data_size"
            field_value = self.orig_stats.n_miscls
        else:
            field_name = "high_error_data_size"
            field_value = self.orig_stats.n_herr

        return {
            "search_algo_config": {
                "search_algo": self.metadata.search_algo,
                "pop_size": self.metadata.pop_size,
                "max_iters": self.metadata.max_iters,
                "algo_specific_params": self.metadata.search_params,
            },
            "verification_data": {
                "sample_data_size": self.orig_stats.n_orig,
                "out_of_domain_data_size": self.orig_stats.n_ood,
                self.orig_stats.metric_name: self.orig_stats.metric_score,
                field_name: field_value,
                "originals_count": self.originals_count,
                "rules_count": self.rules_count,
                "verifications_count_per_original": self.verifications_count_per_original,
                "verifications_count": self.verifications_count,
            },
            "rule_violation_metrics": {
                "violated_rules_count": self.violated_rules_count,
                "violated_rules_percentage": self.violated_rules_percentage,
                "inconsistent_inputs_percentage": self.inconsistent_inputs_percentage,
                "inconsistency_revealing_originals_percentage": self.inconsistency_revealing_originals_percentage,
                "avg_inconsistent_inputs_percentage_per_original": self.avg_inconsistent_inputs_percentage_per_original,
            },
            "constraint_statisfaction_metrics": {
                "unsatisfied_constraints_count": self.unsatisfied_constraints_count,
                "unsatisfied_constraints_percentage": self.unsatisfied_constraints_percentage,
                "infeasible_inputs_percentage": self.infeasible_inputs_percentage,
            },
        }

    def _generate_unsatisfied_constraint_html(
        self, constraint_id: str, infeasible_count: int, infeasible_percentage: float
    ) -> Any:
        """
        Generate HTML content for a specific unsatisfied constraint.

        Args:
            constraint_id (str): Constraint identifier.
            infeasible_count (int): Count of infeasible inputs.
            infeasible_percentage (float): Infeasible inputs percentage.

        Returns:
            HTML content for the constraint section.
        """
        constraint = next(
            (c for c in self.constraints if c.name == constraint_id), None
        )
        return (
            H4(f"Constraint: {constraint_id}"),
            P(
                Strong("Description: "),
                f"{constraint.description if constraint.description else 'No description.'}",
            ),
            P(Strong("Formula: "), f"{constraint.formula if constraint else 'N/A'}"),
            P(
                Strong("Infeasible Points Count (Percentage): "),
                f"{infeasible_count} ({infeasible_percentage}%)",
            ),
        )

    def _generate_unsatisfied_constraint_dict(
        self, constraint_id: str
    ) -> Dict[str, Any]:
        """
        Generate a dictionary representation for an unsatisfied constraint.

        Args:
            constraint_id (str): Constraint identifier.

        Returns:
            Dict[str, Any]: Details of the unsatisfied constraint.
        """
        return {
            "id": constraint_id,
            "infeasible_inputs_percentage": self.infeasible_inputs_percentage_per_constraint(
                constraint_id
            ),
        }

    def _generate_rule_violation_html(
        self, rule_id: str, rule_entries: List[Dict[str, Any]]
    ) -> Any:
        """
        Generate HTML content for a rule violation section.

        Args:
            rule_id (str): Rule identifier.
            rule_entries (List[Dict[str, Any]]): List of inconsistent entries for the rule.

        Returns:
            HTML content for the rule violation section.
        """
        rule = next((r for r in self.rules if r.name == rule_id), None)
        description = (
            H4(f"Rule: {rule_id}"),
            P(Strong("Description:"), f" {rule.description}"),
            P("Premises:"),
            Ul(*[Li(Strong("Premise:"), str(p)) for p in rule.premises]),
            P(Strong("Conclusion:"), str(rule.conclusion)),
        )

        total_entries = len(rule_entries)
        inconsistent_percentage = compute_percentage(
            total_entries, self.total_search_evaluations
        )
        unique_parents = len({entry["parent_id"] for entry in rule_entries})
        revealing_percentage = compute_percentage(unique_parents, self.originals_count)
        insights = (
            P(
                "Derived Inputs Inconsistent with this rule: ",
                Strong(f"{total_entries}"),
                f" / {self.total_search_evaluations} (",
                Strong(f"{inconsistent_percentage}%"),
                " of total verifications).",
            ),
            P(
                "These inconsistencies were revealed by ",
                Strong(f"{unique_parents}"),
                f" / {self.originals_count} (",
                Strong(f"{revealing_percentage}%"),
                " of the original seed).",
            ),
        )

        rule_df = pd.DataFrame(rule_entries)
        deviations = pd.to_numeric(rule_df["deviation"], errors="coerce")
        avg_deviation = deviations.mean() if not deviations.empty else 0
        max_deviation = deviations.max() if not deviations.empty else 0
        insights += (
            P(
                "Average / Maximum Deviations: ",
                Strong(f"{avg_deviation:.2f}"),
                " / ",
                Strong(f"{max_deviation:.2f}"),
            ),
        )

        top_entries = rule_df.nlargest(5, "deviation", keep="all")
        if "deviation" in top_entries.columns:
            top_entries = top_entries.drop(columns=["deviation"])
        insights += (
            H5("Top Inconsistent Inputs:"),
            NotStr(
                top_entries.to_html(
                    classes="table table-striped table-responsive",
                    index=False,
                    escape=False,
                )
            ),
        )
        return description + insights

    def _generate_rule_violation_dict(
        self, rule_id: str, rule_entries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a dictionary representation of a rule violation.

        Args:
            rule_id (str): Rule identifier.
            rule_entries (List[Dict[str, Any]]): List of inconsistent entries for the rule.

        Returns:
            Dict[str, Any]: Rule violation metrics and top inconsistent inputs.
        """
        total_entries = len(rule_entries)
        inconsistent_percentage = compute_percentage(
            total_entries, self.total_search_evaluations
        )
        unique_parents = len({entry["parent_id"] for entry in rule_entries})
        revealing_percentage = compute_percentage(unique_parents, self.originals_count)
        rule_df = pd.DataFrame(rule_entries)
        deviations = pd.to_numeric(rule_df["deviation"], errors="coerce")
        avg_deviation = deviations.mean() if not deviations.empty else 0
        max_deviation = deviations.max() if not deviations.empty else 0
        top_entries = rule_df.nlargest(5, "deviation", keep="all")
        if "deviation" in top_entries.columns:
            top_entries = top_entries.drop(columns=["deviation"])
        return {
            "rule_id": rule_id,
            "rule_inconsistent_entries_count": total_entries,
            "inconsistent_inputs_percentage": inconsistent_percentage,
            "inconsistency_revealing_originals_count": unique_parents,
            "inconsistency_revealing_originals_percentage": revealing_percentage,
            "inconsistent_input_avg_deviation": avg_deviation,
            "inconsistent_input_max_deviation": max_deviation,
            "top_inconsistent_inputs": top_entries.to_dict(orient="records"),
        }

    def _generate_rule_deviations_plot_html(
        self, rule_id: str, rule_entries: List[Dict[str, Any]]
    ) -> Any:
        """
        Generate HTML content for a Plotly deviation histogram for a given rule.

        Args:
            rule_id (str): Rule identifier.
            rule_entries (List[Dict[str, Any]]): List of inconsistent entries for the rule.

        Returns:
            HTML content containing the plot and a descriptive paragraph.
        """
        rule_df = pd.DataFrame(rule_entries)
        deviations = pd.to_numeric(rule_df["deviation"], errors="coerce").dropna()
        if deviations.empty:
            return ()
        fig = go.Figure(go.Histogram(x=deviations))
        fig.update_layout(
            title=f"Deviation Distribution for Rule {rule_id}",
            xaxis_title="Deviation",
            yaxis_title="Count",
            template="plotly_white",
        )
        plot_html = (
            NotStr(fig.to_html(full_html=False, include_plotlyjs="cdn")),
            P(
                "This histogram shows the distribution of deviation values for violations of this rule."
                "It provides insights into severity and frequency."
            ),
        )
        return plot_html

    def _generate_all_constraints_unsatisfaction_html(self) -> Any:
        """
        Generate HTML content for all unsatisfied constraints.

        Returns:
            HTML content for the unsatisfied constraints section.
        """
        constaints_cards = ()
        sorted_constraints_stats = sorted(
            self.constraints_stats.items(), key=lambda item: item[1], reverse=True
        )
        for constraint_id, unsatisfied_count in sorted_constraints_stats:
            if unsatisfied_count == 0:
                continue
            infeasible_inputs_percentage = (
                self.infeasible_inputs_percentage_per_constraint(constraint_id)
            )
            compliancy_rate = round(
                100.0 - infeasible_inputs_percentage, ROUNDING_PRECISION
            )
            compliance_bar = Div(
                f"Compliance rate: {compliancy_rate}%",
                cls="progress-bar bg-info",
                role="progressbar",
                style=f"width: {compliancy_rate}%;",
                aria_valuenow=f"{compliancy_rate}",
                aria_valuemin="0",
                aria_valuemax="100",
            )
            constraint_card = Div(
                Div(
                    Div(
                        Div(
                            H6(
                                Button(
                                    f"{constraint_id}",
                                    cls="btn btn-link text-dark",
                                    data_toggle="collapse",
                                    data_target=f"#collapseRun{self.metadata.run_id}{constraint_id}",
                                    aria_expanded="false",
                                    aria_controls=f"collapseRun{self.metadata.run_id}{constraint_id}",
                                )
                            ),
                            cls="col-md-6",
                        ),
                        Div(
                            Div(compliance_bar, cls="progress", style="height: 20px;"),
                            cls="col-md-6",
                        ),
                        cls="row",
                    ),
                    cls="card-header bg-white",
                    id=f"headingRun{self.metadata.run_id}{constraint_id}",
                ),
                Div(
                    Div(
                        *self._generate_unsatisfied_constraint_html(
                            constraint_id,
                            unsatisfied_count,
                            infeasible_inputs_percentage,
                        ),
                        cls="card-body",
                    ),
                    cls="collapse",
                    id=f"collapseRun{self.metadata.run_id}{constraint_id}",
                    aria_labelledby=f"headingRun{self.metadata.run_id}{constraint_id}",
                    data_parent=f"#run{self.metadata.run_id}{constraint_id}",
                ),
                cls="card shadow-sm mb-2",
            )

            constaints_cards += (
                Div(
                    constraint_card,
                    id=f"run{self.metadata.run_id}{constraint_id}",
                    cls="accordion",
                ),
            )

        html_section = Div(
            Div(
                H6(
                    Button(
                        "⚠️ Unsatisfied Constraints Details",
                        cls="btn btn-link text-dark",
                        data_toggle="collapse",
                        data_target=f"#collapseRun{self.metadata.run_id}Constraints",
                        aria_expanded="false",
                        aria_controls=f"collapseRun{self.metadata.run_id}Constraints",
                    ),
                    cls="mb-0",
                ),
                cls="card-header bg-light",
                id=f"headingRun{self.metadata.run_id}Constraints",
            ),
            Div(
                Div(*constaints_cards, cls="card-body"),
                cls="collapse",
                id=f"collapseRun{self.metadata.run_id}Constraints",
                aria_labelledby=f"headingRun{self.metadata.run_id}Constraints",
                data_parent=f"#run{self.metadata.run_id}Accordion",
            ),
            cls="card shadow-sm mb-2",
        )

        return html_section

    def _generate_all_rules_violations_html(self) -> Any:
        """
        Generate HTML content for all violated rules.

        Returns:
            HTML content for the violated rules section.
        """
        rule_cards = ()
        sorted_inconsistent_data = sorted(
            self.inconsistent_data.items(), key=lambda item: len(item[1]), reverse=True
        )
        for rule_id, rule_inconsistent_entries in sorted_inconsistent_data:
            total_rule_inconsistent_entries = len(rule_inconsistent_entries)
            inconsistent_inputs_percentage = compute_percentage(
                total_rule_inconsistent_entries, self.total_search_evaluations
            )
            consistency_rate = round(
                100.0 - inconsistent_inputs_percentage, ROUNDING_PRECISION
            )
            consistency_bar = Div(
                f"Consistency rate: {consistency_rate}%",
                cls="progress-bar bg-success",
                role="progressbar",
                style=f"width: {consistency_rate}%;",
                aria_valuenow=f"{consistency_rate}",
                aria_valuemin="0",
                aria_valuemax="100",
            )

            rule_card = Div(
                Div(
                    Div(
                        Div(
                            H6(
                                Button(
                                    f"{rule_id}",
                                    cls="btn btn-link text-dark",
                                    data_toggle="collapse",
                                    data_target=f"#collapseRun{self.metadata.run_id}{rule_id}",
                                    aria_expanded="false",
                                    aria_controls=f"collapseRun{self.metadata.run_id}{rule_id}",
                                )
                            ),
                            cls="col-md-6",
                        ),
                        Div(
                            Div(consistency_bar, cls="progress", style="height: 20px;"),
                            cls="col-md-6",
                        ),
                        cls="row",
                    ),
                    cls="card-header bg-white",
                    id=f"headingRun{self.metadata.run_id}{rule_id}",
                ),
                Div(
                    Div(
                        *self._generate_rule_violation_html(
                            rule_id, rule_inconsistent_entries
                        ),
                        *self._generate_rule_deviations_plot_html(
                            rule_id, rule_inconsistent_entries
                        ),
                        cls="card-body",
                    ),
                    cls="collapse",
                    id=f"collapseRun{self.metadata.run_id}{rule_id}",
                    aria_labelledby=f"headingRun{self.metadata.run_id}{rule_id}",
                    data_parent=f"#run{self.metadata.run_id}{rule_id}",
                ),
                cls="card shadow-sm mb-2",
            )

            rule_cards += (
                Div(
                    rule_card, id=f"run{self.metadata.run_id}{rule_id}", cls="accordion"
                ),
            )

        html_section = Div(
            Div(
                H6(
                    Button(
                        "🛑 Violated Rules Details",
                        cls="btn btn-link text-dark",
                        data_toggle="collapse",
                        data_target=f"#collapseRun{self.metadata.run_id}Rules",
                        aria_expanded="false",
                        aria_controls=f"collapseRun{self.metadata.run_id}Rules",
                    ),
                    cls="mb-0",
                ),
                cls="card-header bg-light",
                id=f"headingRun{self.metadata.run_id}Rules",
            ),
            Div(
                Div(*rule_cards, cls="card-body"),
                cls="collapse",
                id=f"collapseRun{self.metadata.run_id}Rules",
                aria_labelledby=f"headingRun{self.metadata.run_id}Rules",
                data_parent=f"#run{self.metadata.run_id}Accordion",
            ),
            cls="card shadow-sm mb-2",
        )

        return html_section

    def to_html(self) -> Any:
        """
        Generate the complete HTML report for this verification run.

        Returns:
            HTML content representing the run.
        """
        verif_accordion = Div(
            Div(
                H6(
                    Button(
                        "📊 Verification Overview",
                        cls="btn btn-link text-dark",
                        data_toggle="collapse",
                        data_target=f"#collapseRun{self.metadata.run_id}Overview",
                        aria_expanded="false",
                        aria_controls=f"collapseRun{self.metadata.run_id}Overview",
                    )
                ),
                cls="card-header bg-light",
                id=f"headingRun{self.metadata.run_id}Overview",
            ),
            Div(
                Div(self._generate_run_verif_overview_html(), cls="card-body"),
                cls="collapse",
                id=f"collapseRun{self.metadata.run_id}Overview",
                aria_labelledby=f"headingRun{self.metadata.run_id}Overview",
                data_parent=f"#run{self.metadata.run_id}Accordion",
            ),
            cls="card shadow-sm mb-2",
        )
        run_datetime = datetime.fromtimestamp(self.metadata.run_ts).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        run_card = Div(
            Div(
                H5(
                    Button(
                        f"📝 Run #{self.metadata.run_id} - {run_datetime}",
                        cls="btn btn-link text-dark font-weight-bold",
                        data_toggle="collapse",
                        data_target=f"#collapseRun{self.metadata.run_id}",
                        aria_expanded="true",
                        aria_controls=f"collapseRun{self.metadata.run_id}",
                    ),
                    cls="mb-0",
                ),
                cls="card-header bg-white",
                id=f"headingRun{self.metadata.run_id}",
            ),
            Div(
                Div(
                    Div(
                        verif_accordion,
                        self._generate_all_constraints_unsatisfaction_html(),
                        self._generate_all_rules_violations_html(),
                        id=f"run{self.metadata.run_id}Accordion",
                        cls="accordion mb-3",
                    ),
                    cls="card-body",
                ),
                cls="collapse",
                id=f"collapseRun{self.metadata.run_id}",
                aria_labelledby=f"headingRun{self.metadata.run_id}",
                data_parent="#runsAccordion",
            ),
            cls="card shadow-sm mb-3",
        )
        return run_card

    def to_dict(self) -> Dict[str, Any]:
        """
        Generate a dictionary representation of the verification run.

        Returns:
            Dict[str, Any]: Overview, unsatisfied constraints, and violated rules details.
        """
        return {
            "overview": self._generate_run_verif_overview_dict(),
            "unsatisfied_constraints_details": [
                self._generate_unsatisfied_constraint_dict(cid)
                for cid, count in self.constraints_stats.items()
                if count != 0
            ],
            "violated_rules_details": [
                self._generate_rule_violation_dict(rid, entries)
                for rid, entries in self.inconsistent_data.items()
            ],
        }

    def _save_orig_data(self) -> None:
        """
        Save the original input data to a CSV file.
        """
        csv_fpath = Path(self.metadata.run_ckpt_dirpath) / "data" / "originals.csv"
        csv_fpath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.orig_data).to_csv(csv_fpath, index=False)
        logger.info("Original data saved to %s", csv_fpath)

    def _save_inconsistent_data(self) -> None:
        """
        Save inconsistent input data for each rule to separate CSV files.
        """
        dirpath = Path(self.metadata.run_ckpt_dirpath) / "data"
        dirpath.mkdir(parents=True, exist_ok=True)
        for rid, entries in self.inconsistent_data.items():
            report_file = dirpath / f"{rid}_inconsistent_data.csv"
            pd.DataFrame(entries).to_csv(report_file, index=False)
            logger.info(
                "Inputs inconsistent with rule %s saved to %s", rid, report_file
            )

    def _save_summary(self) -> None:
        """
        Save a JSON summary of the verification run.
        """
        summary_fpath = Path(self.metadata.run_ckpt_dirpath) / "_summary.json"
        summary_fpath.parent.mkdir(parents=True, exist_ok=True)
        save_json(summary_fpath, self.to_dict())
        logger.info("Summary saved to %s", summary_fpath)

    def _save_metadata(self) -> None:
        """
        Save run metadata, original statistics, and constraints statistics to JSON files.
        """
        dirpath = Path(self.metadata.run_ckpt_dirpath) / "data"
        metadata_file = dirpath / "metadata.json"
        save_json(metadata_file, asdict(self.metadata))
        orig_stats_file = dirpath / "orig_stats.json"
        save_json(orig_stats_file, asdict(self.orig_stats))
        constraints_stats_file = dirpath / "constraints_stats.json"
        save_json(constraints_stats_file, self.constraints_stats)

    def persist(self) -> None:
        """
        Persist all run data including originals, inconsistent inputs, summary, and metadata.
        """
        self._save_orig_data()
        self._save_inconsistent_data()
        self._save_summary()
        self._save_metadata()

    @staticmethod
    def load_from_checkpoint(
        run_checkpoint_dirpath: os.PathLike, model_card: ModelCard, domain: Domain
    ) -> "RulesViolationRun":
        """
        Load a verification run from a checkpoint directory.

        Args:
            run_checkpoint_dirpath (os.PathLike): Path to the run checkpoint.
            model_card (ModelCard): The model card associated with the run.
            domain (Domain): The domain associated with the run.

        Returns:
            RulesViolationRun: The loaded run instance.
        """
        dirpath = Path(run_checkpoint_dirpath) / "data"
        metadata_file = dirpath / "metadata.json"
        run_metadata = RunMetadata(**read_json(metadata_file))
        run = RulesViolationRun(model_card, domain, run_metadata)
        original_csv_fpath = dirpath / "originals.csv"
        run.orig_data = read_data_file(original_csv_fpath).to_dict(orient="records")
        run.inconsistent_data = {}
        for rid in run_metadata.rules_names:
            data_file = dirpath / f"{rid}_inconsistent_data.csv"
            if data_file.exists():
                run.inconsistent_data[rid] = read_data_file(data_file).to_dict(
                    orient="records"
                )
        orig_stats_file = dirpath / "orig_stats.json"
        run.orig_stats = OriginalStatistics(**read_json(orig_stats_file))
        constraints_stats_file = dirpath / "constraints_stats.json"
        run.constraints_stats = read_json(constraints_stats_file)
        return run


class RulesViolationResult:
    """
    Collects, analyzes, and displays rule violation results.

    This class accumulates original inputs, rule violations, and constraint statistics
    from multiple verification runs. It generates HTML reports and JSON summaries to
    facilitate model verification analysis.
    """

    def __init__(self, model_card: "ModelCard", domain: "Domain") -> None:
        """
        Initialize with a model card and domain. Load previous runs from checkpoint.

        Args:
            model_card (ModelCard): The model card for verification.
            domain (Domain): The domain containing rules and constraints.
        """
        self.model_card: "ModelCard" = model_card
        self.domain: "Domain" = domain
        self.runs: List["RulesViolationRun"] = []
        self._load_from_checkpoint()

    def _load_from_checkpoint(self) -> None:
        """
        Load verification runs from checkpoint directories.
        """
        result_dirpath = Path(CHECKPOINTS_DIRPATH) / self.model_card.full_name
        for run_ckpt_dirpath in result_dirpath.glob("run_*"):
            if run_ckpt_dirpath.is_dir():
                self.runs.append(
                    RulesViolationRun.load_from_checkpoint(
                        run_ckpt_dirpath, self.model_card, self.domain
                    )
                )

    def clean(self) -> None:
        """
        Delete all persisted verification artifacts for this model.

        This will remove the entire directory under the global checkpoint path
        corresponding to this model's full name (CHECKPOINTS_DIRPATH/<model_full_name>).
        Use with caution, as this operation is irreversible.

        Raises:
            FileNotFoundError: If the target directory does not exist.
            PermissionError: If the directory or its contents cannot be removed due to filesystem permissions.
        """
        result_dirpath = Path(CHECKPOINTS_DIRPATH) / self.model_card.full_name
        shutil.rmtree(str(result_dirpath))

    def create_new_run(
        self, search_algo: str, search_params: dict, pop_size: int, max_iters: int
    ) -> "RulesViolationRun":
        """
        Create and register a new verification run with the specified search parameters.

        Args:
            search_algo (str): The search algorithm name.
            search_params (dict): The parameters specific to the search algorithm.
            pop_size (int): Population size used in the search.
            max_iters (int): Maximum iterations for the search.

        Returns:
            RulesViolationRun: The newly created run instance.
        """
        new_run = RulesViolationRun(self.model_card, self.domain)
        new_run.set_search_params(search_algo, search_params, pop_size, max_iters)
        self.runs.append(new_run)
        return new_run

    @property
    def constraints(self) -> List["Constraint"]:
        """Return the list of constraints defined in the domain."""
        return self.domain.constraints

    @property
    def rules(self) -> List["Rule"]:
        """Return the list of rules defined in the domain."""
        return self.domain.rules

    @property
    def constraints_stats(self) -> Dict[str, int]:
        """
        Aggregate constraint statistics from all runs.

        Returns:
            Dict[str, int]: Mapping of constraint names to aggregated counts.
        """
        total = Counter()
        for run in self.runs:
            total.update(run.constraints_stats)
        return dict(total)

    @property
    def orig_stats(self) -> OriginalStatistics:
        """
        Aggregate original statistics across all runs.

        Returns:
            OriginalStatistics: Aggregated original data statistics.
        """
        agg_orig_stats = OriginalStatistics()
        for idx, run in enumerate(self.runs):
            if idx == 0:
                agg_orig_stats.n_orig = run.orig_stats.n_orig
                agg_orig_stats.n_miscls = run.orig_stats.n_miscls
                agg_orig_stats.n_ood = run.orig_stats.n_ood
                agg_orig_stats.n_herr = run.orig_stats.n_herr
                agg_orig_stats.err_thresh = run.orig_stats.err_thresh
            else:
                agg_orig_stats.n_orig += run.orig_stats.n_orig
                agg_orig_stats.n_ood += run.orig_stats.n_ood
                if agg_orig_stats.n_miscls is not None:
                    agg_orig_stats.n_miscls += run.orig_stats.n_miscls
                if agg_orig_stats.n_herr is not None:
                    agg_orig_stats.n_herr += run.orig_stats.n_herr
        return agg_orig_stats

    def compute_percentage(self, divide: float, whole: float) -> float:
        """
        Compute the percentage (rounded by ROUNDING_PRECISION) of divide over whole.

        Args:
            divide (float): The numerator.
            whole (float): The denominator.

        Returns:
            float: The computed percentage.
        """
        if whole == 0:
            return 0
        return round(100 * divide / whole, ROUNDING_PRECISION)

    @property
    def originals_count(self) -> int:
        """
        Total count of original inputs across all runs.

        Returns:
            int: Count of original inputs.
        """
        return sum(run.originals_count for run in self.runs)

    @property
    def inconsistency_revealing_originals_percentage(self) -> int:
        """
        Percentage of original inputs that reveal inconsistencies.

        Returns:
            int: Percentage value.
        """
        total_orig = self.originals_count
        if total_orig == 0:
            return 0

        all_inconsistent_data = chain.from_iterable(
            chain.from_iterable(run.inconsistent_data.values()) for run in self.runs
        )
        n_unique_parents = len({entry["parent_id"] for entry in all_inconsistent_data})
        return compute_percentage(n_unique_parents, total_orig)

    @property
    def total_search_evaluations(self) -> int:
        """
        Total number of search evaluations across all runs.

        Returns:
            int: Total search evaluations.
        """
        return sum(run.total_search_evaluations for run in self.runs)

    @property
    def rules_count(self) -> int:
        """
        Count of unique rules across all runs.

        Returns:
            int: Unique rules count.
        """
        return len({r.name for run in self.runs for r in run.rules})

    @property
    def violated_rules_count(self) -> int:
        """
        Count unique rules that have any inconsistent entries.

        Returns:
            int: Violated rules count.
        """
        unique_rule_ids = {
            rule_id
            for run in self.runs
            for rule_id, entries in run.inconsistent_data.items()
            if entries
        }
        return len(unique_rule_ids)

    @property
    def violated_rules_percentage(self) -> int:
        """
        Percentage of rules violated.

        Returns:
            int: Violated rules percentage.
        """
        return compute_percentage(self.violated_rules_count, self.rules_count)

    @property
    def verifications_count(self) -> int:
        """
        Total number of verifications performed across all runs.

        Returns:
            int: Total verifications count.
        """
        return sum(run.verifications_count for run in self.runs)

    @property
    def inconsistent_inputs_percentage(self) -> int:
        """
        Percentage of inconsistent inputs relative to total verifications.

        Returns:
            int: Inconsistent inputs percentage.
        """
        total_verifs = self.verifications_count
        if total_verifs == 0:
            return 0
        inconsistent_inputs_count = sum(
            len(entries)
            for run in self.runs
            for entries in run.inconsistent_data.values()
        )
        return compute_percentage(inconsistent_inputs_count, total_verifs)

    @property
    def total_constraints(self) -> int:
        """
        Total number of constraints with statistics.

        Returns:
            int: Count of constraints.
        """
        return len(self.constraints_stats)

    @property
    def unsatisfied_constraints_count(self) -> int:
        """
        Count of constraints that are unsatisfied.

        Returns:
            int: Unsatisfied constraints count.
        """
        if self.total_constraints == 0:
            return 0
        return sum(1 for cv in self.constraints_stats.values() if cv > 0)

    @property
    def unsatisfied_constraints_percentage(self) -> float:
        """
        Percentage of constraints that are unsatisfied.

        Returns:
            float: Unsatisfied constraints percentage.
        """
        return compute_percentage(
            self.unsatisfied_constraints_count, self.total_constraints
        )

    @property
    def infeasible_inputs_count(self) -> int:
        """
        Count of infeasible inputs based on constraints statistics.
        Uses the maximum count among all constraints.

        Returns:
            int: Infeasible inputs count.
        """
        if self.total_constraints == 0:
            return 0
        return max(self.constraints_stats.values())

    @property
    def infeasible_inputs_percentage(self) -> float:
        """
        Percentage of infeasible inputs relative to total verifications.

        Returns:
            float: Infeasible inputs percentage.
        """
        if self.verifications_count == 0:
            return 0
        return compute_percentage(
            self.infeasible_inputs_count, self.verifications_count
        )

    def inconsistent_inputs_count(self, rid: str) -> int:
        """
        Count inconsistent inputs for a given rule by its identifier.

        Args:
            rid (str): Rule identifier.

        Returns:
            int: Count of inconsistent inputs for that rule.
        """
        rule = next((r for r in self.rules if r.name == rid), None)
        if rule is None:
            return 0
        return sum(len(run.inconsistent_data.get(rule.name, [])) for run in self.runs)

    def infeasible_inputs_percentage_per_constraint(self, cid: str) -> float:
        """
        Percentage of infeasible inputs for a given constraint.

        Args:
            cid (str): Constraint identifier.

        Returns:
            float: Infeasible inputs percentage for the constraint.
        """
        if self.verifications_count == 0:
            return 0
        count = self.constraints_stats.get(cid, 0)
        return compute_percentage(count, self.verifications_count)

    @property
    def avg_infeasible_points_percentage_per_constraint(self) -> float:
        """
        Average percentage of infeasible inputs per constraint.

        Returns:
            float: Average infeasible inputs percentage.
        """
        if self.verifications_count == 0:
            return 0
        percentages = [
            compute_percentage(cv, self.verifications_count)
            for cv in self.constraints_stats.values()
        ]
        return np.mean(percentages)

    def _generate_overview_section(self) -> "Section":
        """
        Generate the HTML overview section for the report.

        Returns:
            Section: HTML section with model and verification data metrics.
        """
        orig_stats = self.orig_stats
        model_info_card = Div(
            Div(
                H4("Model Information", cls="card-title"),
                Hr(),
                P(Strong("Name: "), self.model_card.name),
                P(Strong("Version: "), self.model_card.version),
                P(Strong("Framework: "), self.model_card.framework),
                P(Strong("Description: "), self.model_card.description),
                cls="card-body",
            ),
            cls="card shadow-sm",
        )
        col_model_info = Div(model_info_card, cls="col-md-6 mb-4")

        verif_content = [
            H4("Verification Data Metrics", cls="card-title"),
            Hr(),
            P(Strong("Total Sampled Inputs: "), str(orig_stats.n_orig)),
            P(Strong("Out-of-Domain Inputs: "), str(orig_stats.n_ood)),
        ]
        if self.model_card.model_type == SupportedModelTypes.CLASSIFICATION:
            verif_content.append(
                P(Strong("Misclassified Inputs: "), str(orig_stats.n_miscls))
            )
        elif self.model_card.model_type == SupportedModelTypes.REGRESSION:
            verif_content.append(
                P(
                    Strong(f"High-Error Inputs (>= {orig_stats.err_thresh}): "),
                    str(orig_stats.n_herr),
                )
            )
        verif_content.extend(
            [
                P(Strong("Original Seed Data Size: "), str(self.originals_count)),
                P(Strong("Total Rules: "), str(self.rules_count)),
                P(
                    Strong("Total Generated Verifications: "),
                    str(self.verifications_count),
                ),
            ]
        )
        verification_card = Div(
            Div(*verif_content, cls="card-body"), cls="card shadow-sm"
        )
        col_verification = Div(verification_card, cls="col-md-6 mb-4")
        row_div = Div(col_model_info, col_verification, cls="row")
        container_div = Div(
            Div(
                H2("Overview", cls="text-primary"),
                P(
                    "A quick summary of the verified model and the verification data metrics.",
                    cls="lead",
                ),
                cls="text-center mb-5",
            ),
            row_div,
            cls="container",
        )
        return Section(container_div, id="overview", cls="py-5")

    def _generate_rule_violation_dict(self) -> Dict[str, Any]:
        """
        Generate a dictionary summarizing rule violation metrics.

        Returns:
            Dict[str, Any]: Mapping from rule names to violation metrics.
        """
        rules_data = {}
        for rule in self.rules:
            total_entries = sum(
                len(run.inconsistent_data.get(rule.name, [])) for run in self.runs
            )
            rule_pct = compute_percentage(total_entries, self.total_search_evaluations)
            unique_parents = len(
                {
                    entry["parent_id"]
                    for run in self.runs
                    for entry in run.inconsistent_data.get(rule.name, [])
                }
            )
            revealing_pct = compute_percentage(unique_parents, self.originals_count)
            df = pd.DataFrame(
                list(
                    chain.from_iterable(
                        run.inconsistent_data.get(rule.name, []) for run in self.runs
                    )
                )
            )
            deviations = (
                pd.to_numeric(df["deviation"], errors="coerce")
                if not df.empty
                else pd.Series(dtype=float)
            )
            avg_dev = deviations.mean() if not deviations.empty else 0
            max_dev = deviations.max() if not deviations.empty else 0
            rules_data[rule.name] = {
                "rule_inconsistent_inputs_percentage": rule_pct,
                "inconsistency_revealing_originals_percentage": revealing_pct,
                "inconsistent_input_avg_deviation": avg_dev,
                "inconsistent_input_max_deviation": max_dev,
            }
        return rules_data

    def _generate_rule_violation_section(self) -> "Section":
        """
        Generate the HTML section for rule violations.

        Returns:
            Section: HTML section summarizing rule violation metrics.
        """
        rules_data = []
        sorted_rules = sorted(
            self.rules,
            key=lambda rule: self.inconsistent_inputs_count(rule.name),
            reverse=True,
        )
        for rule in sorted_rules:
            total_entries = self.inconsistent_inputs_count(rule.name)
            rule_pct = compute_percentage(total_entries, self.total_search_evaluations)
            unique_parents = len(
                {
                    entry["parent_id"]
                    for run in self.runs
                    for entry in run.inconsistent_data.get(rule.name, [])
                }
            )
            revealing_pct = compute_percentage(unique_parents, self.originals_count)
            df = pd.DataFrame(
                list(
                    chain.from_iterable(
                        run.inconsistent_data.get(rule.name, []) for run in self.runs
                    )
                )
            )
            deviations = (
                pd.to_numeric(df["deviation"], errors="coerce")
                if not df.empty
                else pd.Series(dtype=float)
            )
            avg_dev = deviations.mean() if not deviations.empty else 0
            max_dev = deviations.max() if not deviations.empty else 0

            rules_data.append(
                {
                    "Rule": rule.name,
                    "Inconsistent Inputs %": rule_pct,
                    "Inconsistency-Revealing Orig. %": revealing_pct,
                    "Avg(dev.)": avg_dev,
                    "Max(dev.)": max_dev,
                }
            )

        table_html = NotStr(
            pd.DataFrame(rules_data).to_html(
                classes="table table-bordered table-striped", index=False, escape=False
            )
        )
        section = Section(
            Div(
                Div(
                    H2("Rules Consistency", cls="text-primary"),
                    P("A summary of rules consistency metrics.", cls="lead"),
                    cls="text-center mb-4",
                ),
                Div(
                    Div(
                        Div(
                            H5(
                                Button(
                                    "🛑 Violated Rules Metrics",
                                    cls="btn btn-link text-dark font-weight-bold",
                                    data_toggle="collapse",
                                    data_target="#collapseRules",
                                    aria_expanded="true",
                                    aria_controls="collapseRules",
                                ),
                                cls="mb-0",
                            ),
                            cls="card-header bg-white",
                            id="headingRules",
                        ),
                        Div(
                            Div(
                                Div(
                                    Div(
                                        P(
                                            Strong("Violated Rules: "),
                                            f"{self.violated_rules_count} ({self.violated_rules_percentage}%)",
                                        ),
                                        cls="col-md-4",
                                    ),
                                    Div(
                                        P(
                                            Strong("Inconsistent Inputs %: "),
                                            f"{self.inconsistent_inputs_percentage}%",
                                        ),
                                        cls="col-md-4",
                                    ),
                                    Div(
                                        P(
                                            Strong("Inconsistency-Revealing Seed %: "),
                                            f"{self.inconsistency_revealing_originals_percentage}%",
                                        ),
                                        cls="col-md-4",
                                    ),
                                    cls="row",
                                ),
                                Div(table_html, cls="table-responsive mt-3"),
                                cls="card-body",
                            ),
                            cls="collapse show",
                            id="collapseRules",
                            aria_labelledby="headingRules",
                            data_parent="#rulesAccordion",
                        ),
                        cls="card shadow-sm",
                    ),
                    id="rulesAccordion",
                    cls="accordion",
                ),
                cls="container",
            ),
            id="rules",
            cls="py-5",
        )
        return section

    def _generate_unsatisfied_constraint_dict(self) -> Dict[str, Any]:
        """
        Generate a dictionary summarizing unsatisfied constraints.

        Returns:
            Dict[str, Any]: Mapping from constraint names to infeasible rates.
        """
        constraints_data = {}
        for constraint in self.constraints:
            infeasible_rate = self.infeasible_inputs_percentage_per_constraint(
                constraint.name
            )
            constraints_data[constraint.name] = {"infeasible_rate": infeasible_rate}
        return constraints_data

    def _generate_unsatisfied_constraint_section(self) -> "Section":
        """
        Generate the HTML section for unsatisfied constraints.

        Returns:
            Section: HTML section displaying constraints unsatisfaction metrics.
        """
        constraints_data = sorted(
            [
                {
                    "Constraint": c.name,
                    "Unsatisfaction %": self.infeasible_inputs_percentage_per_constraint(
                        c.name
                    ),
                }
                for c in self.constraints
            ],
            key=lambda d: d["Unsatisfaction %"],
            reverse=True,
        )
        table_html = NotStr(
            pd.DataFrame(constraints_data).to_html(
                classes="table table-bordered table-striped mt-3",
                index=False,
                escape=False,
            )
        )
        constraints_section = Section(
            Div(
                Div(
                    H2("Constraints Compliance", cls="text-primary"),
                    P("Summary of constraint unsatisfaction metrics.", cls="lead"),
                    cls="text-center mb-4",
                ),
                Div(
                    Div(
                        Div(
                            H5(
                                Button(
                                    "⚠️ Constraints Unsatisfaction Metrics",
                                    cls="btn btn-link text-dark font-weight-bold",
                                    data_toggle="collapse",
                                    data_target="#collapseConstraints",
                                    aria_expanded="true",
                                    aria_controls="collapseConstraints",
                                ),
                                cls="mb-0",
                            ),
                            cls="card-header bg-white",
                            id="headingConstraints",
                        ),
                        Div(
                            Div(
                                Div(
                                    Div(
                                        P(
                                            Strong("Unsatisfied Constraints: "),
                                            f"{self.unsatisfied_constraints_count} ({self.unsatisfied_constraints_percentage}%)",
                                        ),
                                        cls="col-md-4",
                                    ),
                                    Div(
                                        P(
                                            Strong("Infeasible Points %: "),
                                            f"{self.infeasible_inputs_percentage}%",
                                        ),
                                        cls="col-md-4",
                                    ),
                                    cls="row",
                                ),
                                Div(table_html, cls="table-responsive"),
                                cls="card-body",
                            ),
                            cls="collapse show",
                            id="collapseConstraints",
                            aria_labelledby="headingConstraints",
                            data_parent="#constraintsAccordion",
                        ),
                        cls="card shadow-sm",
                    ),
                    id="constraintsAccordion",
                    cls="accordion",
                ),
                cls="container",
            ),
            id="constraints",
            cls="py-5 bg-light",
        )
        return constraints_section

    def save_as_html(self, html_fpath: Union[str, os.PathLike]) -> None:
        """
        Save the complete verification report as an HTML file.

        Args:
            html_fpath (Union[str, os.PathLike]): Destination path for the HTML report.
        """
        html_fpath = Path(html_fpath)
        html_fpath.parent.mkdir(parents=True, exist_ok=True)

        # Define required assets for the report.
        bootstrap_css = Link(
            rel="stylesheet",
            href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css",
        )
        jquery_js = Script(src="https://code.jquery.com/jquery-3.5.1.slim.min.js")
        bootstrap_js = Script(
            src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"
        )
        custom_style_css = Style(
            """
            body { font-family: Arial, sans-serif; background-color: #f1f3f5; }
            header { display: flex; align-items: center; justify-content: space-between; padding: 15px; background-color: #eddcfd; border-bottom: 2px solid #dee2e6; }
            .logo { height: 50px; }
            .container { margin-top: 20px; margin-bottom: 20px; }
            footer { background-color: #f8f9fa; padding: 15px; text-align: center; border-top: 1px solid #dee2e6; font-size: 14px; color: #6c757d; }
            h2 { margin-top: 20px; }
            .card-header button { width: 100%; text-align: left; padding: 0; font-size: 1rem; }
        """
        )
        html_content = Html(
            Head(
                Title("VerifIA Report"),
                Meta(charset="UTF-8"),
                bootstrap_css,
                custom_style_css,
            ),
            Body(
                Header(
                    Img(src="https://www.verifia.ca/assets/logo.png", alt="VerifIA Logo", cls="logo"),
                    H2("Model Verification Report", cls="mb-0 text-secondary")
                ),
                Div(
                    self._generate_overview_section(),
                    self._generate_unsatisfied_constraint_section(),
                    self._generate_rule_violation_section(),
                    Section(
                        Div(
                            Div(
                                H2("Verification Runs", cls="text-primary"),
                                P(
                                    "Detailed information on each verification run.",
                                    cls="lead",
                                ),
                                cls="text-center mb-4",
                            ),
                            Div(
                                *[run.to_html() for run in self.runs],
                                id="runsAccordion",
                                cls="accordion",
                            ),
                            cls="container",
                        ),
                        id="verification-runs",
                        cls="py-5",
                    ),
                    cls="container",
                ),
                Footer(
                    Span(
                        f"Generated by VerifIA on {datetime.now().strftime('%B %d, %Y')}"
                    )
                ),
                jquery_js,
                bootstrap_js,
            ),
        )
        with html_fpath.open("w", encoding="utf-8") as f:
            f.write(to_xml(html_content))
        logger.info("HTML report saved to %s", html_fpath)

    def to_html(self) -> str:
        """
        Render the full verification report as an HTML document.

        This method assembles all required CSS and JavaScript assets (Bootstrap, jQuery),
        applies custom styling, and constructs the report structure.

        Returns:
            str: The complete HTML document serialized to a string.
        """
        # Define required assets for the report.
        bootstrap_css = Link(
            rel="stylesheet",
            href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css",
        )
        jquery_js = Script(src="https://code.jquery.com/jquery-3.5.1.slim.min.js")
        bootstrap_js = Script(
            src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"
        )
        custom_style_css = Style(
            """
            body { font-family: Arial, sans-serif; background-color: #f1f3f5; }
            header { display: flex; align-items: center; justify-content: space-between; padding: 15px; background-color: #eddcfd; border-bottom: 2px solid #dee2e6; }
            .logo { height: 50px; }
            .container { margin-top: 20px; margin-bottom: 20px; }
            footer { background-color: #f8f9fa; padding: 15px; text-align: center; border-top: 1px solid #dee2e6; font-size: 14px; color: #6c757d; }
            h2 { margin-top: 20px; }
            .card-header button { width: 100%; text-align: left; padding: 0; font-size: 1rem; }
        """
        )
        html_content = Html(
            Head(
                Title("VerifIA Report"),
                Meta(charset="UTF-8"),
                bootstrap_css,
                custom_style_css,
            ),
            Body(
                Header(
                    Img(src="logo.png", alt="VerifIA Logo", cls="logo"),
                    H2("Model Verification Report", cls="mb-0 text-secondary"),
                ),
                Div(
                    self._generate_overview_section(),
                    self._generate_unsatisfied_constraint_section(),
                    self._generate_rule_violation_section(),
                    Section(
                        Div(
                            Div(
                                H2("Verification Runs", cls="text-primary"),
                                P(
                                    "Detailed information on each verification run.",
                                    cls="lead",
                                ),
                                cls="text-center mb-4",
                            ),
                            Div(
                                *[run.to_html() for run in self.runs],
                                id="runsAccordion",
                                cls="accordion",
                            ),
                            cls="container",
                        ),
                        id="verification-runs",
                        cls="py-5",
                    ),
                    cls="container",
                ),
                Footer(
                    Span(
                        f"Generated by VerifIA on {datetime.now().strftime('%B %d, %Y')}"
                    )
                ),
                jquery_js,
                bootstrap_js,
            ),
        )
        return str(to_xml(html_content))

    def display(self):
        """
        Render and display the HTML report in a Jupyter Notebook.

        This method converts the HTML representation of the report (generated by self.to_html())
        to a string and uses IPython's display_html to render it. The 'raw=True' parameter ensures
        that the HTML is interpreted as raw HTML rather than plain text.
        """
        # display_html(self.to_html(), raw=True)
        tmp_dirpath = mk_tmpdir()
        local_fpath = str(Path(tmp_dirpath) / f"temp_frame.html")
        self.save_as_html(local_fpath)
        display(
            IFrame(src=local_fpath, width=700, height=600), metadata=dict(isolated=True)
        )
        rm_tmpdir(tmp_dirpath)

    def save_as_json(self, json_fpath: Union[str, os.PathLike]) -> None:
        """
        Save the verification results as a JSON file.

        Args:
            json_fpath (Union[str, os.PathLike]): Destination file path for the JSON report.
        """
        if self.model_card.model_type == SupportedModelTypes.CLASSIFICATION:
            field_name = "misclassified_data_size"
            field_value = self.orig_stats.n_miscls
        else:
            field_name = "high_error_data_size"
            field_value = self.orig_stats.n_herr

        data = {
            "overview": {
                "model": {
                    "name": self.model_card.name,
                    "version": self.model_card.version,
                    "framework": self.model_card.framework,
                    "features": self.model_card.feature_names,
                    "target": self.model_card.target_name,
                },
                "verification_data": {
                    "original_data_size": self.orig_stats.n_orig,
                    "out_of_domain_data_size": self.orig_stats.n_ood,
                    field_name: field_value,
                    "orig_data_size": self.originals_count,
                    "rules_count": self.rules_count,
                    "total_gen_verifs": self.verifications_count,
                },
            },
            "verification_summary": {
                "rules_consistency_details": {
                    "violated_rules_count": self.violated_rules_percentage,
                    "inconsistent_inputs_percentage": self.inconsistent_inputs_percentage,
                    "inconsistency_revealing_originals_percentage": self.inconsistency_revealing_originals_percentage,
                    "per_rule": self._generate_rule_violation_dict(),
                },
                "constraints_compliancy_details": {
                    "unsatisfied_constraints_count": self.unsatisfied_constraints_count,
                    "unsatisfied_constraints_percentage": self.unsatisfied_constraints_percentage,
                    "infeasible_inputs_percentage": self.infeasible_inputs_percentage,
                    "avg_infeasible_points_percentage_per_constraint": self.avg_infeasible_points_percentage_per_constraint,
                    "per_constraint": self._generate_unsatisfied_constraint_dict(),
                },
            },
            "verification_runs": {run.run_id: run.to_dict() for run in self.runs},
        }
        save_json(json_fpath, data)
        logger.info("JSON report saved to %s", json_fpath)

    def log_as_html(self, report_name: str) -> None:
        """
        Log the HTML report to the model registry via MLflow or equivalent.

        Args:
            report_name (str): The report name.
        """
        tmp_dirpath = mk_tmpdir()
        local_fpath = str(Path(tmp_dirpath) / f"{report_name}.html")
        self.save_as_html(local_fpath)
        model_name = f"{self.model_card.name}_{self.model_card.framework}"
        mlops_platform = get_configured_mlops_platform()
        if mlops_platform == SupportedMLOpsPlatform.MLFLOW:
            mlflow.log_report(model_name, self.model_card.version, local_fpath)
        elif mlops_platform == SupportedMLOpsPlatform.COMET_ML:
            comet_ml.log_report(model_name, self.model_card.version, local_fpath)
        elif mlops_platform == SupportedMLOpsPlatform.WANDB:
            wandb.log_report(model_name, self.model_card.version, local_fpath)
        else:
            raise ValueError(f"Unsupported MLOps platform: {mlops_platform}")
        rm_tmpdir(tmp_dirpath)
        logger.info("HTML report logged for model %s", model_name)

    def log_as_json(self, report_name: str) -> None:
        """
        Log the JSON report to the model registry via MLflow or equivalent.

        Args:
            report_name (str): The report name.
        """
        tmp_dirpath = mk_tmpdir()
        local_fpath = str(Path(tmp_dirpath) / f"{report_name}.json")
        self.save_as_json(local_fpath)
        model_name = f"{self.model_card.model.name}_{self.model_card.model.framework}"
        mlops_platform = get_configured_mlops_platform()
        if mlops_platform == SupportedMLOpsPlatform.MLFLOW:
            mlflow.log_report(model_name, self.model_card.model.version, local_fpath)
        elif mlops_platform == SupportedMLOpsPlatform.COMET_ML:
            comet_ml.log_report(model_name, self.model_card.model.version, local_fpath)
        elif mlops_platform == SupportedMLOpsPlatform.WANDB:
            wandb.log_report(model_name, self.model_card.model.version, local_fpath)
        else:
            raise ValueError(f"Unsupported MLOps platform: {mlops_platform}")
        rm_tmpdir(tmp_dirpath)
        logger.info("JSON report logged for model %s", model_name)
