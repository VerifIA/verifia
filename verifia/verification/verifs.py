from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Tuple
import numpy as np
import numpy.typing as npt

from ..models import BaseModel, ModelCard
from ..context.domain import Domain, VarType, ChangeType, Rule, DomainVar, Constraint
from ..utils.helpers.sys import safe_eval
from ..utils.helpers.rand import (
    randint_excluding,
    uniform_excluding,
    randint_including,
    uniform_including,
)
from ..utils import EPSILON

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class VerifTask:
    """
    Encapsulates a verification case with the metadata required to explore, compute, and retain
    specific features as defined by the domain.
    """

    def __init__(self, rule: Rule, constraints: List[Constraint]) -> None:
        """
        Initialize a VerifTask.

        Args:
            rule (Rule): The rule to verify.
            constraints (List[Constraint]): A list of domain constraints.
        """
        self.rule: Rule = rule
        self.constraints: List[Constraint] = constraints
        self.model_features_sorted: List[str] = (
            []
        )  # Ordered feature names from the model
        self.explorable_variables: List[DomainVar] = []  # Variables to be explored
        self.computed_variables: List[DomainVar] = []  # Variables computed via formulas
        self.constant_variables: List[DomainVar] = []  # Variables to remain constant

    @staticmethod
    def build(rule_id: str, domain: Domain, model_card: ModelCard) -> "VerifTask":
        """
        Build a VerifTask from a given rule identifier, domain, and model card.

        Args:
            rule_id (str): Identifier of the rule to verify.
            domain (Domain): The domain containing variables, rules, and constraints.
            model_card (ModelCard): The model card containing feature names.

        Returns:
            VerifTask: The constructed verification task.

        Raises:
            Exception: If a rule references a variable not present in the model features.
        """
        rule: Rule = domain.find_rule(rule_id)
        # Identify variables that are allowed to change (non-constant) per the rule.
        changing_vars = [
            premise.input
            for premise in rule.premises
            if premise.allowed_change != ChangeType.CST
        ]
        for var in changing_vars:
            if var.name not in model_card.feature_names:
                raise Exception(
                    f"Rule {rule_id} cannot be applied because '{var.name}' is not a feature."
                )

        # Variables computed from formulas.
        derived_vars = [
            var
            for var in domain.variables
            if var.has_formula and var.name in model_card.feature_names
        ]
        # Remaining variables are those present in the model but not derived or already changing.
        remaining_vars = [
            var
            for var in domain.variables
            if var.name in model_card.feature_names
            and var not in derived_vars
            and var not in changing_vars
        ]
        # Add variables with non-zero epsilon to the changing list.
        changing_vars += [var for var in remaining_vars if var.has_nonzero_epsilon]
        # Variables that remain constant.
        unchanged_vars = [var for var in remaining_vars if var not in changing_vars]

        verif_task = VerifTask(rule, domain.constraints)
        verif_task.model_features_sorted = model_card.feature_names
        verif_task.explorable_variables = changing_vars
        verif_task.computed_variables = derived_vars
        verif_task.constant_variables = unchanged_vars
        return verif_task

    @property
    def vect_size(self) -> int:
        """Return the total number of features defined in the model."""
        return len(self.model_features_sorted)

    @property
    def explorable_features_idxs(self) -> List[int]:
        """
        Return indices of explorable features based on their names.

        Returns:
            List[int]: List of indices corresponding to explorable variables.
        """
        return [
            self.model_features_sorted.index(var.name)
            for var in self.explorable_variables
        ]

    @property
    def constant_features_idxs(self) -> List[int]:
        """
        Return indices of constant features based on their names.

        Returns:
            List[int]: List of indices corresponding to constant variables.
        """
        return [
            self.model_features_sorted.index(var.name)
            for var in self.constant_variables
        ]

    def assemble_derived_data_features(
        self, explored_data: npt.NDArray, constant_input_features: npt.NDArray
    ) -> npt.NDArray:
        """
        Assemble a complete set of features (including derived values) for verification.

        Combines explorable data, constant input features, and computed values (via formulas)
        into one data matrix.

        Args:
            explored_data (npt.NDArray): Data from explorable variables.
            constant_input_features (npt.NDArray): Constant input feature values.

        Returns:
            npt.NDArray: A 2D array of assembled features.
        """
        n_inputs = explored_data.shape[0]
        data_features = np.empty((n_inputs, self.vect_size), dtype=object)
        # Process explorable variables.
        for exp_idx, var in enumerate(self.explorable_variables):
            ft_idx = self.model_features_sorted.index(var.name)
            if var.type == VarType.INT:
                data_features[:, ft_idx] = np.round(explored_data[:, exp_idx])
            elif var.type == VarType.CAT:
                # Map integer indices to categorical values.
                cat_mapping = np.vectorize(lambda val: var.value(int(round(val))))
                data_features[:, ft_idx] = cat_mapping(explored_data[:, exp_idx])
            else:
                data_features[:, ft_idx] = explored_data[:, exp_idx]
        # Insert constant features.
        data_features[:, self.constant_features_idxs] = np.tile(
            constant_input_features, (n_inputs, 1)
        )
        # Compute derived features using provided formulas.
        for inp_idx in range(n_inputs):
            local_vars = {
                var.name: explored_data[inp_idx, idx]
                for idx, var in enumerate(self.explorable_variables)
            }
            local_vars.update(
                {
                    var.name: constant_input_features[idx]
                    for idx, var in enumerate(self.constant_variables)
                }
            )
            for var in self.computed_variables:
                ft_idx = self.model_features_sorted.index(var.name)
                data_features[inp_idx, ft_idx] = safe_eval(var.formula, local_vars)
        return data_features

    def partition_original_input_features(
        self, original_input_features: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Partition the original input features into explorable and constant subsets.

        Args:
            original_input_features (npt.NDArray): The complete feature vector.

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: A tuple (explorable_features, constant_features).
        """
        explorable = original_input_features[self.explorable_features_idxs].copy()
        constant = original_input_features[self.constant_features_idxs].copy()
        return explorable, constant

    def assess_constraints_compliance(
        self, derived_data: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray, Dict[str, int]]:
        """
        Assess constraint compliance for each candidate in the derived data.

        For each candidate, the number of unsatisfied constraints is added to a base score of 1.
        Fully compliant candidates have a score of 1.0.

        Args:
            derived_data (npt.NDArray): Derived input data for verification.

        Returns:
            Tuple containing:
              - are_feasible (npt.NDArray): Boolean array indicating full compliance.
              - compliance_scores (npt.NDArray): Numeric scores per candidate.
              - constraints_stats (Dict[str, int]): Count of violations per constraint.
        """
        n_inputs = derived_data.shape[0]
        compliance_scores = np.ones(n_inputs)
        total_constraints = len(self.constraints)
        constraints_stats = {c.name: 0 for c in self.constraints}
        if total_constraints > 0:
            for i in range(n_inputs):
                inp_vars = {
                    ft: derived_data[i, idx]
                    for idx, ft in enumerate(self.model_features_sorted)
                }
                satisfied_count = 0
                for constraint in self.constraints:
                    if safe_eval(constraint.formula, inp_vars):
                        satisfied_count += 1
                    else:
                        constraints_stats[constraint.name] += 1
                compliance_scores[i] += total_constraints - satisfied_count
        are_feasible = compliance_scores == 1.0
        return are_feasible, compliance_scores, constraints_stats

    def filter_derived_inputs(
        self, derived_data: npt.NDArray, explorable_input_features: npt.NDArray
    ) -> npt.NDArray:
        """
        Filter derived inputs to enforce rule-specific constraints on explorable features.

        For each candidate, if a feature violates a rule (e.g. must not equal, must belong to, etc.),
        the value is replaced using the appropriate random generator.

        Args:
            derived_data (npt.NDArray): Derived candidate input data.
            explorable_input_features (npt.NDArray): Original explorable input features.

        Returns:
            npt.NDArray: The filtered derived data.
        """
        for inp_idx in range(derived_data.shape[0]):
            for ft_idx, var in enumerate(self.explorable_variables):
                premise = self.rule.find_premise(var)
                if premise is None:
                    continue
                ub, lb = var.max, var.min
                if premise.allowed_change == ChangeType.NOEQ:
                    discarded_val = premise.involved_values[0]
                    if var.type == VarType.CAT:
                        discarded_val = var.index(discarded_val)
                    if derived_data[inp_idx, ft_idx] == discarded_val:
                        if var.type in [VarType.CAT, VarType.INT]:
                            derived_data[inp_idx, ft_idx] = randint_excluding(
                                int(lb), int(ub), discarded_val
                            )
                        elif var.type == VarType.FLOAT:
                            derived_data[inp_idx, ft_idx] = uniform_excluding(
                                lb, ub, discarded_val
                            )
                elif premise.allowed_change == ChangeType.IN:
                    desired_vals = premise.involved_values
                    if var.type == VarType.CAT:
                        desired_vals = [var.index(v) for v in desired_vals]
                    if derived_data[inp_idx, ft_idx] not in desired_vals:
                        if var.type in [VarType.CAT, VarType.INT]:
                            derived_data[inp_idx, ft_idx] = randint_including(
                                int(lb), int(ub), desired_vals
                            )
                        elif var.type == VarType.FLOAT:
                            derived_data[inp_idx, ft_idx] = uniform_including(
                                lb, ub, desired_vals
                            )
                elif premise.allowed_change == ChangeType.NOIN:
                    discarded_vals = premise.involved_values
                    if var.type == VarType.CAT:
                        discarded_vals = [var.index(v) for v in discarded_vals]
                    if derived_data[inp_idx, ft_idx] in discarded_vals:
                        if var.type in [VarType.CAT, VarType.INT]:
                            derived_data[inp_idx, ft_idx] = randint_excluding(
                                int(lb), int(ub), discarded_vals
                            )
                        elif var.type == VarType.FLOAT:
                            derived_data[inp_idx, ft_idx] = uniform_excluding(
                                lb, ub, discarded_vals
                            )
        return derived_data

    def define_search_space_bounds(
        self, explorable_input_features: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Define search space bounds for explorable features based on input features and rule premises.

        For each explorable variable, upper and lower bounds are computed depending on its type,
        current value, and allowed change type. For features with an "EQ" rule, the bounds are centered
        on the desired value.

        Args:
            explorable_input_features (npt.NDArray): Original input feature values for explorable variables.

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: A tuple (lower_bound, upper_bound) for the search space.
        """
        upper_bound = np.zeros_like(explorable_input_features)
        lower_bound = np.zeros_like(explorable_input_features)
        for ft_idx, var in enumerate(self.explorable_variables):
            if var.type == VarType.CAT:
                ft_value = var.index(explorable_input_features[ft_idx])
            elif var.type in [VarType.INT, VarType.FLOAT]:
                ft_value = explorable_input_features[ft_idx]

            premise = self.rule.find_premise(var)
            if premise is None:
                # Not mentioned in rule; treat as constant (CST)
                ub = min(var.max, ft_value + var.epsilon)
                lb = max(var.min, ft_value - var.epsilon)
            else:
                if premise.allowed_change == ChangeType.CST:
                    ub = min(var.max, ft_value + var.epsilon)
                    lb = max(var.min, ft_value - var.epsilon)
                elif premise.allowed_change == ChangeType.INC:
                    ub = min(var.max, ft_value + var.max_delta)
                    lb = min(var.max, ft_value + var.min_delta)
                elif premise.allowed_change == ChangeType.DEC:
                    ub = max(var.min, ft_value - var.min_delta)
                    lb = max(var.min, ft_value - var.max_delta)
                elif premise.allowed_change == ChangeType.NOINC:
                    ub = min(var.max, ft_value + var.epsilon)
                    lb = max(var.min, ft_value - var.max_delta)
                elif premise.allowed_change == ChangeType.NODEC:
                    ub = min(var.max, ft_value + var.max_delta)
                    lb = max(var.min, ft_value - var.epsilon)
                elif premise.allowed_change in [
                    ChangeType.VAR,
                    ChangeType.NOEQ,
                    ChangeType.IN,
                    ChangeType.NOIN,
                ]:
                    ub = min(var.max, ft_value + var.max_delta)
                    lb = max(var.min, ft_value - var.max_delta)
                elif premise.allowed_change == ChangeType.EQ:
                    desired_value = premise.involved_values[0]
                    if var.type == VarType.CAT:
                        desired_idx = var.index(desired_value)
                        ub = min(var.max, desired_idx + var.epsilon)
                        lb = max(var.min, desired_idx - var.epsilon)
                    elif var.type in [VarType.INT, VarType.FLOAT]:
                        ub = min(var.max, desired_value + var.epsilon)
                        lb = max(var.min, desired_value - var.epsilon)
            upper_bound[ft_idx] = ub
            lower_bound[ft_idx] = lb
        upper_bound = np.array(upper_bound, dtype="float64") + EPSILON
        lower_bound = np.array(lower_bound, dtype="float64")
        return lower_bound, upper_bound


class VerifAssertion(ABC):
    """
    Abstract base class for verification assertions.

    A verification assertion compares a model's prediction on modified input data
    against an original prediction to compute deviation errors.
    """

    def __init__(self, model: BaseModel, insignificant_deviation: float) -> None:
        """
        Initialize the verification assertion.

        Args:
            model (BaseModel): The model to use for predictions.
            insignificant_deviation (float): The deviation threshold considered insignificant.
        """
        self.model = model
        self.insignificant_deviation = insignificant_deviation

    @abstractmethod
    def compute_deviation(
        self, original_prediction: float, predictions: npt.NDArray
    ) -> npt.NDArray:
        """
        Compute the deviation errors given the original prediction and derived predictions.

        Args:
            original_prediction (float): The original prediction value.
            predictions (npt.NDArray): Predictions for the derived input data.

        Returns:
            npt.NDArray: Computed deviation errors.
        """
        pass

    @abstractmethod
    def check_consistency(self, deviations: npt.NDArray) -> npt.NDArray:
        """
        Determine which derived inputs are consistent based on deviation errors.

        Args:
            deviations (npt.NDArray): Computed deviation errors.

        Returns:
            npt.NDArray: Boolean array indicating consistency.
        """
        pass

    def verify(
        self, derived_data: npt.NDArray, original_prediction: float
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Verify the derived input data against the original prediction.

        This method follows a template that:
          1. Computes predictions using the model.
          2. Computes deviation errors via the subclass-specific `compute_deviation`.
          3. Checks consistency using the subclass-specific `check_consistency`.

        Args:
            derived_data (npt.NDArray): Modified input data for verification.
            original_prediction (float): The original prediction value.

        Returns:
            Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
                - are_consistent: Boolean array indicating which candidates are consistent.
                - actual_deviations: Computed deviation errors.
                - derived_predictions: Predictions for each candidate.
        """
        predictions = self.model.predict_score(derived_data)
        deviations = self.compute_deviation(original_prediction, predictions)
        are_consistent = self.check_consistency(deviations)
        return are_consistent, deviations, predictions

    @staticmethod
    def build(rule_id: str, domain: Domain, model: BaseModel) -> "VerifAssertion":
        """
        Build a verification assertion based on the rule's expected change.

        Args:
            rule_id (str): The identifier of the rule.
            domain (Domain): The domain containing the rule.
            model (BaseModel): The model used for predictions.

        Returns:
            VerifAssertion: An instance of a concrete verification assertion.

        Raises:
            ValueError: If the expected change type is unsupported.
        """
        rule: Rule = domain.find_rule(rule_id)
        expected_change = rule.conclusion.expected_change
        if expected_change == ChangeType.INC:
            return PredictionIncrease(model, rule.conclusion.output.epsilon)
        elif expected_change == ChangeType.DEC:
            return PredictionDecrease(model, rule.conclusion.output.epsilon)
        elif expected_change == ChangeType.NOINC:
            return PredictionNoIncrease(model, rule.conclusion.output.epsilon)
        elif expected_change == ChangeType.NODEC:
            return PredictionNoDecrease(model, rule.conclusion.output.epsilon)
        elif expected_change == ChangeType.CST:
            return PredictionInvariance(model, rule.conclusion.output.epsilon)
        else:
            raise ValueError(f"Unsupported expected change: {expected_change}")


class PredictionInvariance(VerifAssertion):
    """
    Verification assertion for prediction invariance.

    Ensures that predictions remain effectively unchanged.
    """

    def compute_deviation(
        self, original_prediction: float, predictions: npt.NDArray
    ) -> npt.NDArray:
        # Absolute deviation between predictions and the original value.
        return np.abs(predictions - original_prediction)

    def check_consistency(self, deviations: npt.NDArray) -> npt.NDArray:
        # Consistent if deviation is within the insignificant threshold.
        return deviations <= self.insignificant_deviation


class PredictionIncrease(VerifAssertion):
    """
    Verification assertion for prediction increase.

    Ensures that predictions are at least a certain amount above the original value.
    """

    def compute_deviation(
        self, original_prediction: float, predictions: npt.NDArray
    ) -> npt.NDArray:
        required_minimum = original_prediction + self.insignificant_deviation
        # Compute shortfall: if a prediction is below the required minimum, record the difference.
        return np.maximum(required_minimum - predictions, 0)

    def check_consistency(self, deviations: npt.NDArray) -> npt.NDArray:
        # Consistent if there is no shortfall.
        return deviations == 0


class PredictionDecrease(VerifAssertion):
    """
    Verification assertion for prediction decrease.

    Ensures that predictions are at least a certain amount below the original value.
    """

    def compute_deviation(
        self, original_prediction: float, predictions: npt.NDArray
    ) -> npt.NDArray:
        required_maximum = original_prediction - self.insignificant_deviation
        # Compute excess: if a prediction is above the required maximum, record the difference.
        return np.maximum(predictions - required_maximum, 0)

    def check_consistency(self, deviations: npt.NDArray) -> npt.NDArray:
        # Consistent if there is no excess.
        return deviations == 0


class PredictionNoIncrease(VerifAssertion):
    """
    Verification assertion for no significant prediction increase.

    Allows a small increase, but flags deviations exceeding the insignificant threshold.
    """

    def compute_deviation(
        self, original_prediction: float, predictions: npt.NDArray
    ) -> npt.NDArray:
        # Compute how much predictions exceed the original.
        return np.maximum(predictions - original_prediction, 0)

    def check_consistency(self, deviations: npt.NDArray) -> npt.NDArray:
        # Consistent if the increase is within the insignificant threshold.
        return deviations <= self.insignificant_deviation


class PredictionNoDecrease(VerifAssertion):
    """
    Verification assertion for no significant prediction decrease.

    Allows a small decrease, but flags deviations exceeding the insignificant threshold.
    """

    def compute_deviation(
        self, original_prediction: float, predictions: npt.NDArray
    ) -> npt.NDArray:
        # Compute how much predictions fall below the original.
        return np.maximum(original_prediction - predictions, 0)

    def check_consistency(self, deviations: npt.NDArray) -> npt.NDArray:
        # Consistent if the decrease is within the insignificant threshold.
        return deviations <= self.insignificant_deviation
