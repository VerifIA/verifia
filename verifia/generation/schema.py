import logging
from typing import Any, Dict, List, Optional, Union, Annotated

from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from verifia.utils.enums.context import VarType
from verifia.utils.helpers.io import save_yaml
from pydantic import (
    BaseModel,
    Field,
    conlist,
    field_validator,
    model_validator,
    StringConstraints,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a custom type for premise condition strings.
# Allowed patterns include literal conditions (inc, dec, var, cst, noinc, nodec)
# as well as function-like conditions (e.g., eq("value"), noeq("value"),
# in("value1"; "value2"), noin("value1"; "value2")).
PremiseCondition = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_lower=True,
        pattern=(
            r"^(?:inc|dec|var|cst|noinc|nodec|"
            r"eq\((?:'[^']+'|\"[^\"]+\"|-?\d+(?:\.\d+)?)\)|"
            r"noeq\((?:'[^']+'|\"[^\"]+\"|-?\d+(?:\.\d+)?)\)|"
            r"in\(\s*(?:'[^']+'|\"[^\"]+\"|-?\d+(?:\.\d+)?)(?:\s*;\s*(?:'[^']+'|\"[^\"]+\"|-?\d+(?:\.\d+)?))*\s*\)|"
            r"noin\(\s*(?:'[^']+'|\"[^\"]+\"|-?\d+(?:\.\d+)?)(?:\s*;\s*(?:'[^']+'|\"[^\"]+\"|-?\d+(?:\.\d+)?))*\s*\))$"
        ),
    ),
]

# Define a custom type for conclusion condition strings.
# Only literal conditions are allowed.
ConclusionCondition = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, to_lower=True, pattern=r"^(inc|dec|cst|noinc|nodec)$"
    ),
]


class Variable(BaseModel):
    """
    Model for defining a feature variable used in a dataset.

    Attributes:
        description (str): Detailed explanation of the feature variable.
        type (VarType): The type of the variable (INT, FLOAT, CAT).
        range (Optional[List[Union[int, float]]]): For numeric variables, a two-element list [min, max].
        values (Optional[List[str]]): For categorical variables, a list of allowed string values.
        variation_limits (Optional[List[float]]): Two-element list representing allowed variation percentages (0 to 1).
        insignificant_variation (Optional[float]): Threshold (0 to 1) below which variation is insignificant.
    """

    description: str = Field(
        ...,
        description=(
            "Detailed description of the feature variable. Explains what the variable represents."
        ),
    )
    type: VarType = Field(
        ...,
        description=(
            "Type of the variable. Possible values: INT, FLOAT, CAT. Determines whether the variable is integer, float"
            " or categorical."
        ),
    )
    range: Optional[
        conlist(item_type=Union[int, float], min_length=2, max_length=2)
    ] = Field(
        None,
        description=(
            "For INT/FLOAT types, a required two-element list [min, max] that defines the allowed range of values."
        ),
    )
    values: Optional[List[str]] = Field(
        None,
        description=(
            "For CAT type variables, a required list of allowed string values. Should not be provided if type "
            "is INT or FLOAT."
        ),
    )
    variation_limits: Optional[conlist(item_type=float, min_length=2, max_length=2)] = (
        Field(
            None,
            description=(
                "Two-element list [min, max] of variation limits as percentages that we use when applying rules "
                "(expressed as floats between 0 and 1)."
            ),
        )
    )
    insignificant_variation: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Insignificant variation threshold. A float between 0 and 1 indicating the percentage variation "
            "considered insignificant for this variable."
        ),
    )

    @field_validator("range")
    @classmethod
    def check_range(
        cls, v: Optional[List[Union[int, float]]]
    ) -> Optional[List[Union[int, float]]]:
        """
        Validator to ensure that the numeric range [min, max] is ordered correctly.

        Args:
            v: The list representing the range.

        Raises:
            ValueError: If the first element is greater than the second.
        """
        if v is not None and v[0] > v[1]:
            raise ValueError("Range minimum must be less than or equal to maximum.")
        return v

    @field_validator("variation_limits")
    @classmethod
    def check_variation_limits(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """
        Validator to ensure that variation_limits are non-negative, ordered, and within [0, 1].

        Args:
            v: The list representing the variation limits.

        Raises:
            ValueError: If limits are negative, out of order, or exceed 1.
        """
        if v is not None:
            if v[0] < 0 or v[1] < 0:
                raise ValueError("Variation limits must be non-negative.")
            if v[0] > v[1]:
                raise ValueError(
                    "The first variation limit must be less than or equal to the second."
                )
            if v[0] > 1.0 or v[1] > 1.0:
                raise ValueError(
                    "Variation limits should be percentages between 0 and 1."
                )
        return v

    @model_validator(mode="after")
    def validate_variable(self) -> "Variable":
        """
        Model-level validation ensuring consistency between type and provided fields.

        For INT/FLOAT variables, a 'range' is required and 'values' should not be provided.
        For CAT variables, 'values' is required and 'range' should not be provided.

        Raises:
            ValueError: If the conditions are not met.
        """
        # For INT/FLOAT, a range is required and "values" must not be provided.
        if self.type in {VarType.INT, VarType.FLOAT}:
            if self.range is None:
                raise ValueError(f"Variable of type {self.type} requires a range.")
            if self.values is not None:
                raise ValueError(
                    f"Variable of type {self.type} should not have 'values' defined."
                )
        # For CAT, values are required and "range" must not be provided.
        if self.type == VarType.CAT:
            if self.values is None:
                raise ValueError("Categorical variable requires a list of values.")
            if self.range is not None:
                raise ValueError(
                    "Categorical variable should not have a range defined."
                )
        return self


class Constraint(BaseModel):
    """
    Model representing a constraint between feature variables.

    Attributes:
        description: A detailed description explaining the constraint.
        formula: A Python expression (as a string) that must evaluate to True for the constraint to be satisfied.
    """

    description: str = Field(
        ...,
        description=(
            "A detailed explanation of the constraint, describing its purpose and the relationship it "
            "enforces among variables."
        ),
    )
    formula: str = Field(
        ...,
        description=(
            "A Python expression in string format representing the constraint. This expression should evaluate to "
            "True when the constraint is met."
        ),
    )


class Rule(BaseModel):
    """
    Model representing a rule with premises and a conclusion that should be satisfied in the dataset.

    Attributes:
        description: A rationale explaining why the rule exists.
        premises: A dictionary mapping variable names to their expected conditions (e.g., 'inc', 'dec', 'eq(value)').
        conclusion: A dictionary mapping exactly one variable name to its resulting condition,
            representing the rule's outcome.
    """

    description: str = Field(
        ...,
        description=(
            "A rationale behind the rule, explaining the logic or reasoning that connects the premises "
            "to the conclusion."
        ),
    )
    # The premises can have multiple variables with conditions.
    premises: Dict[str, PremiseCondition] = Field(
        ...,
        description=(
            "Mapping of variable names to condition strings for premises. "
            "Allowed abbreviations:\n"
            "  • 'inc': the variable is expected to increase.\n"
            "  • 'dec': the variable is expected to decrease.\n"
            "  • 'var': the variable can vary in either direction.\n"
            "  • 'cst': the variable should remain constant (default if not explicitly modified).\n"
            "  • 'noinc': the variable should not increase (i.e., either decrease or remain constant).\n"
            "  • 'nodec': the variable should not decrease (i.e., either increase or remain constant).\n"
            "Function-like conditions are supported:\n"
            "  • eq('value') or eq(value): the variable must equal the specified value.\n"
            "  • noeq('value') or noeq(value): the variable must not equal the specified value.\n"
            "  • in('value1'; 'value2') or in(value1; value2): the variable must be one of the specified values.\n"
            "  • noin('value1'; 'value2') or noin(value1; value2): the variable must not be any of "
            "the specified values."
        ),
    )
    # The conclusion must contain exactly one variable with its condition.
    conclusion: Dict[str, ConclusionCondition] = Field(
        ...,
        description=(
            "Mapping containing exactly one variable and its condition for the conclusion. "
            "Allowed abbreviations: \n"
            "  • 'inc': the variable should increase.\n"
            "  • 'dec': the variable should decrease.\n"
            "  • 'cst': the variable should remain constant.\n"
            "  • 'noinc': the variable should not increase (i.e., either decrease or remain constant).\n"
            "  • 'nodec': the variable should not decrease (i.e., either increase or remain constant)."
        ),
    )

    @model_validator(mode="after")
    def validate_conclusion(self) -> "Rule":
        """
        Validates that the conclusion contains exactly one variable.

        Raises:
            ValueError: If the conclusion mapping does not contain exactly one item.
        """
        if len(self.conclusion) != 1:
            raise ValueError(
                "The conclusion must contain exactly one variable condition."
            )
        return self


class YAMLSpecification(BaseModel):
    """
    Model representing the complete YAML specification including feature variables, constraints, and rules.

    Attributes:
        variables: A dictionary mapping feature variable names to Variable objects.
        constraints: A dictionary mapping constraint names to Constraint objects.
        rules: A dictionary mapping rule names to Rule objects.
    """

    variables: Dict[str, Variable] = Field(
        default_factory=dict,
        description=(
            "A dictionary of feature variable definitions. Each key is a feature variable name "
            "and its value is a Variable object containing all metadata and validation rules."
        ),
    )
    constraints: Dict[str, Constraint] = Field(
        default_factory=dict,
        description=(
            "A dictionary of constraint definitions. Each key is a constraint name "
            "and its value is a Constraint object that enforces relationships between variables."
        ),
    )
    rules: Dict[str, Rule] = Field(
        default_factory=dict,
        description=(
            "A dictionary of rule definitions. Each key is a rule name and its value is a Rule object "
            "that specifies logical relationships using premises and a single conclusion."
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the YAMLSpecification to a dictionary using JSON mode.

        Returns:
            Dict[str, Any]: The YAML specification as a dictionary.
        """
        return self.model_dump(mode="json")

    def export_to_yaml(self, file_path: str) -> None:
        """
        Export the YAMLSpecification object to a YAML file.

        This method converts the specification to a dictionary (including all nested models)
        and writes it to the specified file path in YAML format using PyYAML's safe_dump.

        Args:
            file_path (str): The path to the YAML file where the specification will be saved.
        """
        save_yaml(yaml_fpath=file_path, data=self.model_dump(mode="json"))


class DomainGraphState(TypedDict):
    """
    TypedDict representing the state of a domain graph.

    Keys include messages, analysis outputs, and YAML specification details.
    """

    messages: Annotated[List[AnyMessage], add_messages]
    variables_analysis: str
    variables_retriever: str
    constraints_analysis: str
    constraints_retriever: str
    rules_analysis: str
    rules_retriever: str
    yaml_spec_obj: YAMLSpecification
    yaml_spec_str: str
    yaml_retry_feedback: str
    retry_count: int
    validator_output: str
    user_instructions: str
