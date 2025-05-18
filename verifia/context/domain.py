import os
import logging
import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from ..utils import EPSILON
from ..utils.helpers.io import read_yaml
from ..utils.enums.context import ChangeType, VarType

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class DomainVar:
    """
    Represents a variable in the domain with its associated metadata.

    Attributes:
        name (str): The name of the variable.
        description (Optional[str]): Optional description of the variable.
        type (VarType): The type of the variable (e.g., INT, FLOAT, CAT).
        values (Optional[Tuple[str, ...]]): Allowed categorical values (if applicable).
        min_value (Optional[Union[int, float]]): Lower bound for numeric variables.
        max_value (Optional[Union[int, float]]): Upper bound for numeric variables.
        formula (Optional[str]): Optional formula associated with the variable.
        max_variation (float): Factor indicating the maximum allowed variation (default is 1.0).
        min_variation (float): Factor indicating the minimum allowed variation (default is EPSILON).
        insignificant_variation (float): Threshold below which variation is considered insignificant (default is 0.0).
    """

    name: str
    description: Optional[str]
    type: VarType
    values: Optional[Tuple[str, ...]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    formula: Optional[str] = None
    max_variation: float = 1.0
    min_variation: float = EPSILON
    insignificant_variation: float = 0.0

    def value(self, idx: int) -> Any:
        """
        Retrieve the categorical value corresponding to the provided index.

        Args:
            idx (int): Index of the desired categorical value.

        Returns:
            Any: The categorical value at the specified index.

        Raises:
            TypeError: If the variable is not of categorical type.
            ValueError: If the index is out of range.
        """
        if self.type != VarType.CAT:
            raise TypeError(f"Variable {self.name} is not categorical.")
        if self.values is None:
            raise ValueError(
                f"Categorical values for variable {self.name} are not defined."
            )
        if 0 <= idx < len(self.values):
            return self.values[idx]
        raise ValueError(f"Index {idx} is out of range for variable '{self.name}'.")

    def index(self, val: Any) -> int:
        """
        Retrieve the index of the given categorical value.

        Args:
            val (Any): The categorical value to look up.

        Returns:
            int: The index of the value in the allowed values.

        Raises:
            TypeError: If the variable is not of categorical type.
            ValueError: If the value is not found.
        """
        if self.type != VarType.CAT:
            raise TypeError(f"Variable {self.name} is not categorical.")
        if self.values is None:
            raise ValueError(
                f"Categorical values for variable {self.name} are not defined."
            )
        try:
            return self.values.index(val)
        except ValueError:
            raise ValueError(
                f"Value '{val}' is not a valid option for variable '{self.name}'."
            )

    def __eq__(self, other: Any) -> bool:
        """
        Compare two DomainVar instances based on their name.

        Args:
            other (Any): The object to compare against.

        Returns:
            bool: True if other is a DomainVar with the same name, otherwise NotImplemented.
        """
        if isinstance(other, DomainVar):
            return self.name == other.name
        return NotImplemented

    @property
    def has_nonzero_epsilon(self) -> bool:
        """
        Indicates if the variable has a nonzero insignificant variation threshold.

        Returns:
            bool: True if insignificant_variation is greater than 0, False otherwise.
        """
        return self.insignificant_variation > 0

    @property
    def has_formula(self) -> bool:
        """
        Indicates if a formula is associated with the variable.

        Returns:
            bool: True if a formula is defined, False otherwise.
        """
        return self.formula is not None

    @property
    def min(self) -> Union[int, float]:
        """
        Provides the minimum value for the variable.

        For numeric types, returns the provided min_value.
        For categorical types, returns 0.

        Returns:
            Union[int, float]: The minimum value or index.
        """
        if self.type in [VarType.INT, VarType.FLOAT]:
            return self.min_value  # Assumes min_value is defined for numeric variables.
        elif self.type == VarType.CAT:
            return 0
        else:
            raise ValueError(f"Unsupported variable type: {self.type}")

    @property
    def max(self) -> Union[int, float]:
        """
        Provides the maximum value for the variable.

        For numeric types, returns the provided max_value.
        For categorical types, returns the maximum valid index.

        Returns:
            Union[int, float]: The maximum value or index.
        """
        if self.type in [VarType.INT, VarType.FLOAT]:
            return self.max_value  # Assumes max_value is defined for numeric variables.
        elif self.type == VarType.CAT:
            if self.values is None:
                raise ValueError(
                    f"Categorical values for variable {self.name} are not defined."
                )
            return len(self.values) - 1
        else:
            raise ValueError(f"Unsupported variable type: {self.type}")

    @property
    def min_delta(self) -> Union[int, float]:
        """
        Calculates the allowed variation (delta) for the variable.

        Returns:
            Union[int, float]: The product of the range (max - min) and the variation limit.
        """
        scaler = self.max - self.min
        return scaler * self.min_variation

    @property
    def max_delta(self) -> Union[int, float]:
        """
        Calculates the allowed variation (delta) for the variable.

        Returns:
            Union[int, float]: The product of the range (max - min) and the variation limit.
        """
        scaler = self.max - self.min
        return scaler * self.max_variation

    @property
    def epsilon(self) -> Union[int, float]:
        """
        Calculates the insignificant variation (epsilon) for the variable.

        Returns:
            Union[int, float]: The product of the range (max - min) and the insignificant variation factor.
        """
        scaler = self.max - self.min
        return scaler * self.insignificant_variation

    @staticmethod
    def parse_variables(variables_dict: Dict[str, Any]) -> List["DomainVar"]:
        """
        Parse a dictionary of variable definitions into a list of DomainVar instances.

        Args:
            variables_dict (Dict[str, Any]): Dictionary containing variable definitions.

        Returns:
            List[DomainVar]: A list of parsed DomainVar instances.

        Raises:
            ValueError: If an unsupported variable type is encountered.
        """

        def _parse_var_type(var_type_str: str) -> VarType:
            var_type_mapping = {
                "int": VarType.INT,
                "float": VarType.FLOAT,
                "cat": VarType.CAT,
            }
            try:
                return var_type_mapping[var_type_str.lower()]
            except KeyError:
                raise ValueError(f"Unsupported variable type: {var_type_str}")

        domain_vars: List[DomainVar] = []
        for var_name, var_info in variables_dict.items():
            var_type = _parse_var_type(var_info["type"])

            lower, upper = None, None
            if var_type in [VarType.INT, VarType.FLOAT] and "range" in var_info:
                lower, upper = var_info["range"]

            values = None
            if var_type == VarType.CAT and "values" in var_info:
                values = tuple(var_info["values"])

            min_variation, max_variation = EPSILON, 1.0
            if ("variation_limits" in var_info) and (
                var_info["variation_limits"] is not None
            ):
                min_variation, max_variation = var_info["variation_limits"]

            domain_var = DomainVar(
                name=var_name,
                description=var_info.get("description", ""),
                type=var_type,
                values=values,
                min_value=lower,
                max_value=upper,
                formula=var_info.get("formula", None),
                min_variation=min_variation,
                max_variation=max_variation,
                insignificant_variation=var_info.get("insignificant_variation", 0.0),
            )
            domain_vars.append(domain_var)
        return domain_vars


@dataclass
class RulePremise:
    """
    Represents a rule premise for an input variable change.

    Attributes:
        input (DomainVar): The domain variable associated with the premise.
        allowed_change (ChangeType): The allowed type of change.
        involved_values (List[Any]): List of values involved in the allowed change.
    """

    input: DomainVar
    allowed_change: ChangeType
    involved_values: List[Any]

    def _base_str(self) -> str:
        """
        Helper function to build the base string representation for the premise.
        """
        if self.allowed_change == ChangeType.DEC:
            return (
                f"Variable '{self.input.name}' should decrease by at least {(100 * self.input.min_variation)} "
                f"and up to {(100 * self.input.max_variation)}%."
            )
        elif self.allowed_change == ChangeType.INC:
            return (
                f"Variable '{self.input.name}' should increase by at least {(100 * self.input.min_variation)} "
                f"and up to {(100 * self.input.max_variation)}%."
            )
        elif self.allowed_change == ChangeType.NODEC:
            return f"""Variable '{self.input.name}' is expected to remain constant,
                        +/- {(100 * self.input.insignificant_variation)}%, 
                        or increase by up to {(100 * self.input.max_variation)}%."""
        elif self.allowed_change == ChangeType.NOINC:
            return f"""Variable '{self.input.name}' is expected to remain constant,
                        +/- {(100 * self.input.insignificant_variation)}%, 
                        or decrease by up to {(100 * self.input.max_variation)}%."""
        elif self.allowed_change == ChangeType.EQ:
            return f"Variable '{self.input.name}' should be equal to '{self.involved_values[0]}'."
        elif self.allowed_change == ChangeType.NOEQ:
            return f"Variable '{self.input.name}' should NOT be equal to '{self.involved_values[0]}'."
        elif self.allowed_change == ChangeType.IN:
            values_str = ", ".join(map(str, self.involved_values))
            return f"Variable '{self.input.name}' should belong to ({values_str})."
        elif self.allowed_change == ChangeType.NOIN:
            values_str = ", ".join(map(str, self.involved_values))
            return f"Variable '{self.input.name}' should NOT belong to ({values_str})."
        elif self.allowed_change == ChangeType.VAR:
            return f"Variable, {self.input.name}, can vary in both directions \
                    by at least {(100 * self.input.min_variation)} and up to {(100 * self.input.max_variation)}%"
        elif self.allowed_change == ChangeType.CST:
            return (
                f"Variable, {self.input.name}, is expected to remain constant, "
                f"+/- {(100 * self.input.insignificant_variation)}%"
            )
        else:
            return f"Variable '{self.input.name}' has an unsupported change type."

    def __str__(self) -> str:
        return self._base_str()

    def __repr__(self) -> str:
        return self._base_str()


@dataclass
class RuleConclusion:
    """
    Represents a rule conclusion for an output variable change.

    Attributes:
        output (DomainVar): The domain variable associated with the conclusion.
        expected_change (ChangeType): The expected type of change for the output.
    """

    output: DomainVar
    expected_change: ChangeType

    def _base_str(self) -> str:
        """
        Helper function to build the base string representation for the conclusion.
        """
        if self.expected_change == ChangeType.DEC:
            return (
                f"Variable '{self.output.name}' is expected to decrease "
                f"by more than {(100 * self.output.insignificant_variation)}%."
            )
        elif self.expected_change == ChangeType.INC:
            return (
                f"Variable '{self.output.name}' is expected to increase "
                f"by more than {(100 * self.output.insignificant_variation)}%."
            )
        elif self.expected_change == ChangeType.NODEC:
            return f"""Variable '{self.output.name}' is expected to remain constant, 
                    +/- {(100 * self.output.insignificant_variation)}%, 
                    or increase."""
        elif self.expected_change == ChangeType.NOINC:
            return f"""Variable '{self.output.name}' is expected to remain constant, 
                    +/- {(100 * self.output.insignificant_variation)}%, 
                    or decrease."""
        elif self.expected_change == ChangeType.CST:
            return (
                f"Variable '{self.output.name}' is expected to remain constant, "
                f"+/- {(100 * self.output.insignificant_variation)}%."
            )
        else:
            return f"Variable '{self.output.name}' has an unsupported expected change type."

    def __str__(self) -> str:
        return self._base_str()

    def __repr__(self) -> str:
        return self._base_str()


@dataclass
class Rule:
    """
    Represents a rule connecting premises to a conclusion.

    Attributes:
        name (str): The name of the rule.
        description (Optional[str]): Optional description of the rule.
        premises (List[RulePremise]): List of premises for the rule.
        conclusion (RuleConclusion): The conclusion of the rule.
    """

    name: str
    description: Optional[str]
    premises: List[RulePremise]
    conclusion: RuleConclusion

    def find_premise(self, var: DomainVar) -> Optional[RulePremise]:
        """
        Find and return the premise corresponding to the given variable.

        Args:
            var (DomainVar): The variable to look up in the premises.

        Returns:
            Optional[RulePremise]: The matching RulePremise if found, else None.
        """
        for premise in self.premises:
            if premise.input.name == var.name:
                return premise
        return None

    @staticmethod
    def parse_rules(
        variables: List[DomainVar], rules_dict: Dict[str, Any]
    ) -> List["Rule"]:
        """
        Parse a dictionary of rule definitions into a list of Rule instances.

        Args:
            variables (List[DomainVar]): List of DomainVar instances to reference in rules.
            rules_dict (Dict[str, Any]): Dictionary containing rule definitions.

        Returns:
            List[Rule]: A list of parsed Rule instances.

        Raises:
            ValueError: If a referenced variable does not exist or an unsupported change type is encountered.
        """

        def _find_var_by_name(var_name: str) -> DomainVar:
            for var in variables:
                if var.name == var_name:
                    return var
            raise ValueError(f"Variable '{var_name}' does not exist.")

        def _parse_change_type(change_str: str) -> ChangeType:
            change_type_mapping = {
                "inc": ChangeType.INC,
                "dec": ChangeType.DEC,
                "noinc": ChangeType.NOINC,
                "nodec": ChangeType.NODEC,
                "inc_or_cst": ChangeType.NODEC,
                "dec_or_cst": ChangeType.NOINC,
                "var": ChangeType.VAR,
                "cst": ChangeType.CST,
                "eq": ChangeType.EQ,
                "noeq": ChangeType.NOEQ,
                "in": ChangeType.IN,
                "noin": ChangeType.NOIN,
            }
            # Remove any parameters if present (e.g., "inc(0.2)" becomes "inc").
            match = re.match(r"^[a-zA-Z]+", change_str)
            change_type_token = match.group(0) if match else change_str
            try:
                return change_type_mapping[change_type_token.lower()]
            except KeyError:
                raise ValueError(f"Unsupported change type: {change_str}")

        def _parse_involved_values(change_str: str) -> List[Any]:
            """
            Extract involved values from a change string if any are specified.

            Args:
                change_str (str): The change string potentially containing values in parentheses.

            Returns:
                List[Any]: A list of involved values.
            """
            match = re.search(r"\((.*?)\)", change_str)
            if match:
                values_str = match.group(1).replace(";", ",")
                try:
                    # Convert the string representation to a list.
                    return ast.literal_eval(f"[{values_str}]")
                except Exception as e:
                    raise ValueError(
                        f"Error parsing involved values from '{change_str}': {e}"
                    )
            return []

        rules: List[Rule] = []
        for rule_name, rule_info in rules_dict.items():
            premises: List[RulePremise] = []
            for var_name, change_str in rule_info["premises"].items():
                var = _find_var_by_name(var_name)
                allowed_change = _parse_change_type(change_str)
                involved_values = _parse_involved_values(change_str)
                premises.append(
                    RulePremise(
                        input=var,
                        allowed_change=allowed_change,
                        involved_values=involved_values,
                    )
                )

            # Expect a single conclusion defined as a dictionary {variable_name: change_str}
            try:
                conclusion_var_name, conclusion_change_str = next(
                    iter(rule_info["conclusion"].items())
                )
            except StopIteration:
                raise ValueError(f"Rule '{rule_name}' has no conclusion defined.")

            conclusion_var = _find_var_by_name(conclusion_var_name)
            expected_change = _parse_change_type(conclusion_change_str)
            conclusion = RuleConclusion(
                output=conclusion_var, expected_change=expected_change
            )

            rule = Rule(
                name=rule_name,
                description=rule_info.get("description", ""),
                premises=premises,
                conclusion=conclusion,
            )
            rules.append(rule)
        return rules


@dataclass
class Constraint:
    """
    Represents a constraint in the domain.

    Attributes:
        name (str): The name of the constraint.
        description (Optional[str]): Optional description of the constraint.
        formula (str): The formula defining the constraint.
    """

    name: str
    description: Optional[str]
    formula: str

    @staticmethod
    def parse_constraints(constraints_dict: Dict[str, Any]) -> List["Constraint"]:
        """
        Parse a dictionary of constraint definitions into a list of Constraint instances.

        Args:
            constraints_dict (Dict[str, Any]): Dictionary containing constraint definitions.

        Returns:
            List[Constraint]: A list of parsed Constraint instances.
        """
        constraints: List[Constraint] = []
        for cname, cdict in constraints_dict.items():
            constraint = Constraint(
                name=cname,
                description=cdict.get("description", ""),
                formula=cdict["formula"],
            )
            constraints.append(constraint)
        return constraints


class Domain:
    """
    Represents a domain that encompasses variables, rules, and constraints.

    Provides methods to access and manipulate the domain's components.
    """

    def __init__(
        self,
        variables: List[DomainVar],
        rules: List[Rule],
        constraints: List[Constraint],
    ) -> None:
        """
        Initialize a Domain instance with variables, rules, and constraints.

        Args:
            variables (List[DomainVar]): List of domain variables.
            rules (List[Rule]): List of rules.
            constraints (List[Constraint]): List of constraints.
        """
        self.variables = variables
        self.rules = rules
        self.constraints = constraints

    def find_var(self, var_name: str) -> DomainVar:
        """
        Find and return a DomainVar by its name.

        Args:
            var_name (str): The name of the variable.

        Returns:
            DomainVar: The variable with the specified name.

        Raises:
            ValueError: If the variable does not exist in the domain.
        """
        for var in self.variables:
            if var.name == var_name:
                return var
        raise ValueError(f"Variable '{var_name}' does not exist in the Domain.")

    def find_rule(self, name: str) -> Rule:
        """
        Find and return a Rule by its name.

        Args:
            name (str): The name of the rule.

        Returns:
            Rule: The rule with the specified name.

        Raises:
            ValueError: If the rule does not exist in the domain.
        """
        for rule in self.rules:
            if rule.name == name:
                return rule
        raise ValueError(f"Rule '{name}' does not exist in the Domain.")

    def find_constraint(self, name: str) -> Constraint:
        """
        Find and return a Constraint by its name.

        Args:
            name (str): The name of the constraint.

        Returns:
            Constraint: The constraint with the specified name.

        Raises:
            ValueError: If the constraint does not exist in the domain.
        """
        for cons in self.constraints:
            if cons.name == name:
                return cons
        raise ValueError(f"Constraint '{name}' does not exist in the Domain.")

    def get_rules(self, out_name: str) -> Iterator[Rule]:
        """
        Retrieve all rules whose conclusion output matches a given variable name.

        Args:
            out_name (str): The name of the output variable.

        Yields:
            Rule: Each rule whose conclusion's output variable matches the provided name.
        """
        for rule in self.rules:
            if rule.conclusion.output.name == out_name:
                yield rule

    def get_constraints(self) -> List[str]:
        """
        Retrieve all constraint formulas defined in the domain.

        Returns:
            List[str]: A list of constraint formulas.
        """
        return [constraint.formula for constraint in self.constraints]

    @staticmethod
    def build(domain_config_input: Union[os.PathLike, Dict[str, Any]]) -> "Domain":
        """
        Build a Domain instance from a YAML configuration file.

        The YAML file must define 'variables', 'rules', and 'constraints'.

        Args:
            domain_config_input (Union[str, Path]): Path to the YAML configuration file.

        Returns:
            Domain: A constructed Domain instance.

        Raises:
            KeyError: If any required key is missing in the configuration.
        """
        if isinstance(domain_config_input, str):
            cfg_dict = read_yaml(domain_config_input)
        else:
            cfg_dict = domain_config_input

        try:
            variables_dict = cfg_dict["variables"]
            rules_dict = cfg_dict["rules"]
            constraints_dict = cfg_dict["constraints"]
        except KeyError as e:
            raise KeyError(f"Missing required key in configuration: {e}")

        variables = DomainVar.parse_variables(variables_dict)
        rules = Rule.parse_rules(variables, rules_dict)
        constraints = Constraint.parse_constraints(constraints_dict)
        return Domain(variables, rules, constraints)
