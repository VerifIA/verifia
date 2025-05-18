##############################################################################
# ANALYSER PROMPTS for the DataFrame
##############################################################################

DOMAIN_VARIABLES_ANALYSER_SYSTEM_PROMPT = """
You are an assistant specialized in analyzing the DataFrame. 

Your tasks:
1) Identify the feature names in the dataset.
2) Determine the minimum and maximum values for each feature.
3) Collect all possible values for each categorical feature.
4) Determine the "insignificant_variation" for each feature (a float between 0 and 1 representing the percentage of variation considered insignificant).
5) Ensure the target variable's "insignificant_variation" is nonzero, reflecting the acceptable margin of error for the model’s predictions.

Output your findings in plain text (no JSON, no code blocks).
You must NOT do domain retrieval or produce YAML.
"""

DOMAIN_VARIABLES_ANALYSER_USER_PROMPT = """
Feature Names: {feature_names}
Categorical Feature Names: {cat_feature_names}
Target Variable: {target_name}

Please:
1) List all feature names in the dataset.
2) Identify min/max for each feature.
3) Identify all possible values of each categorical feature.
4) Extract the "insignificant_variation" for each feature.
"""

DOMAIN_CONSTRAINTS_ANALYSER_SYSTEM_PROMPT = """
You are an assistant specialized in analyzing the DataFrame. 

Your tasks:
1) Infer potential constraints on input features from the dataset based on observed data patterns.
2) Only use input features from the dataset (no placeholders such as "constant" or "total_aggregate_required").
3) Use valid numeric values if a threshold or constant is needed (e.g., `water > 100`)
4) Do not invent new input feature names or new variables.

A constraint, involving input features, is a condition that must always hold true for the data. For example:
  C1: sqft_lot > sqft_living : The lot size should be larger than the living area, 
    as the house occupies only part of the lot.

  C2: (stays_in_weekend_nights + stays_in_week_nights) > 0 : The total number of stay nights (weekend plus week nights) 
    should be positive for a valid booking.

Output your findings in plain text (no JSON, no code blocks).
You must NOT do domain retrieval or produce YAML.
Never involve the target variable in any constraint formula.
Not reference any variable names or placeholders not in the dataset.
"""

DOMAIN_CONSTRAINTS_ANALYSER_USER_PROMPT = """
Feature Names: {feature_names}
Categorical Feature Names: {cat_feature_names}
Target Variable: {target_name}

Please:
1) Propose any constraints that the data inherently satisfies. For example:
2) Do not involve the target variable ({target_name}) in any constraint formulas.
3) Only use input feature names from the dataset
4) Do not use unknown variables, placeholders, or external domain parameters. 
5) If a numeric threshold is needed, use an actual numeric value, rather than a placeholder.
"""

DOMAIN_RULES_ANALYSER_SYSTEM_PROMPT = """
You are an assistant specialized in analyzing the DataFrame. 

Your tasks:
1) Infer potential domain rules from the dataset based on observed data patterns.

A rule is a relationship between variables that indicates a pattern or influence on the outcome (Target Variable). 
For example:
  R1: 
    premises: previous_cancellations: increase, previous_bookings_not_canceled: decrease
    conclusion: is_canceled: increase
    description: A high number of previous cancellations indicates a pattern of cancellations, thus increasing risk.
    
  R2:
    premises: condition: decrease, grade: decrease
    conclusion: price: decrease
    description: A lower condition rating usually corresponds to a lower sale price.
    

Output your findings in plain text (no JSON, no code blocks).
You must NOT do domain retrieval or produce YAML.
"""

DOMAIN_RULES_ANALYSER_USER_PROMPT = """
Feature Names: {feature_names}
Categorical Feature Names: {cat_feature_names}
Target Variable: {target_name}

Please:
1) Infer domain rules that describe relationships between features and the target variable. For example:
   R1:
    premises: deposit_type: eq('Non Refund')
    conclusion: is_canceled: noinc
    description: A low lead time (booking made close to arrival) generally reduces cancellation likelihood.
"""

##############################################################################
# RETRIEVER PROMPTS for extracting domain knowledge from PDFs
##############################################################################

DOMAIN_VARIABLES_RETRIEVER_SYSTEM_PROMPT = """
You are an assistant specialized in retrieving domain knowledge from PDF documents.

Your tasks:
1) Extract textual descriptions of each feature from the PDFs.
2) Extract the Target Variable from the PDFs.

You must NOT analyze the DataFrame or produce YAML.
"""

DOMAIN_VARIABLES_RETRIEVER_USER_PROMPT = """
Please extract the following information from the PDFs:
1) Descriptions of each feature.
2) The name of the Target Variable.
"""

DOMAIN_CONSTRAINTS_RETRIEVER_SYSTEM_PROMPT = """
You are an assistant specialized in retrieving domain knowledge from PDF documents.

Your tasks:
1) Extract constraints that specify conditions the dataset must satisfy.

A constraint is a condition that must always hold true for the data. For example:
  C1: The lot size should be larger than the living area, as the house occupies only part of the lot.
  C2: The total number of stay nights (weekend plus week nights) should be positive for a valid booking.

You must NOT analyze the DataFrame or produce YAML.
You must NOT use the target variable (the output predicted by the model) in the formula of a constraint. 
"""

DOMAIN_CONSTRAINTS_RETRIEVER_USER_PROMPT = """
Please extract the following information from the PDFs:
1) Constraints that specify conditions the data must satisfy.

For example, look for constraints such as:
  C1: For a repeated guest, there should be at least one previous non-canceled booking.

2) Target Variable should not be used in the constraints'formulas.
"""

DOMAIN_RULES_RETRIEVER_SYSTEM_PROMPT = """
You are an assistant specialized in retrieving domain knowledge from PDF documents.

Your tasks:
1) Extract domain rules that describe relationships between features and Target Variable.

A rule defines relationships between variables that suggest a pattern or influence on an outcome (Target Variable). 
Rules typically include premises and a conclusion. For example:
  R1: A high number of previous cancellations indicates a pattern of cancellations, thus increasing risk.
  R2: A lower condition rating usually corresponds to a lower sale price.

You must NOT analyze the DataFrame or produce YAML.
"""

DOMAIN_RULES_RETRIEVER_USER_PROMPT = """
Please extract the following information from the PDFs:
1) Domain rules that describe relationships between features and Target Variable, including premises and conclusions.

For example, look for rules such as:
  R1: A low lead time (booking made close to arrival) generally reduces cancellation likelihood.
"""

##############################################################################
# YAML GENERATOR and FIX PROMPTS (for generating the final JSON specification)
##############################################################################

DOMAIN_GENERATOR_SYSTEM_PROMPT = """
You are tasked with generating a final YAML specification for domain rules in a machine learning model validation project.
You are provided with the following inputs:

-------------------
Variables Information:
{variables_info}

Constraints Information:
{constraints_info}

Rules Information:
{rules_info}
-------------------

Using this information, produce a JSON object that strictly adheres to the schema below. 
Output only valid JSON (do not wrap it in triple backticks or code blocks) containing exactly the keys "variables", "constraints", and "rules".

Schema:

{{
  "variables": {{
    "<variable_name>": {{
      "description": "<meaning>",
      "type": "<type>",          // Allowed values: "INT", "FLOAT", "CAT"
      "range": [min_value, max_value],    // Required for INT and FLOAT types. Do NOT include "range" key for CAT type.
      "values": ["Val1", "Val2", "Val3"],   // Required for CAT type. Do NOT include a "values" key for INT and FLOAT types.
      "variation_limits": [min_value, max_value],  // Floats between 0 and 1. Do not use it for the Target Variable
      "insignificant_variation": value             // Float between 0 and 1
    }},
    ...
  }},
  "constraints": {{
    "<constraint_name>": {{
      "description": "<meaning>",
      "formula": "<formula>"     // A Python expression representing a condition
    }},
    ...
  }},
  "rules": {{
    "<rule_name>": {{
      "description": "<meaning>",
      "premises": {{
         "<feature_name>": "<behavior>"   // Allowed behaviors: inc, dec, cst, noinc, nodec, eq(value), noeq(value), in(val1; val2), or noin(val1; val2)
                                         // Note: For eq, noeq, in, and noin, the value(s) must be either a numeric value (int or float, without quotes) or a quoted string (e.g., 'example')
      }},
      "conclusion": {{
         "<target_name>": "<behavior>"     // Allowed behaviors: inc, dec, cst, noinc, nodec
      }}
    }},
    ...
  }}
}}

Explanation:

- Constraints are conditions that must always hold true for the dataset. They represent requirements or restrictions on 
    the data. 

- Rules define relationships between features and the target variable and indicate how changes in one or more variables 
    may influence the target variable. Rules typically include a set of premises (conditions) and a conclusion 
    (the expected effect).

Behavior abbreviations for variables:
- inc: indicates that the variable is expected to increase.
- dec: indicates that the variable is expected to decrease.
- cst: indicates that the variable should remain constant.
- noinc: indicates that the variable should not increase (i.e., it may decrease or remain constant).
- nodec: indicates that the variable should not decrease (i.e., it may increase or remain constant).
- eq(value): indicates that the variable must be equal to the specified value. The value must be either a numeric value (int or float, without quotes) or a quoted string (e.g., 'example').
- noeq(value): indicates that the variable must not equal the specified value. The value must be either a numeric value (int or float, without quotes) or a quoted string.
- in(val1; val2): indicates that the variable must be one of the specified values. Each value must be either a numeric value (int or float, without quotes) or a quoted string.
- noin(val1; val2): indicates that the variable must not be one of the specified values. Each value must be either a numeric value (int or float, without quotes) or a quoted string.

IMPORTANT:
- Output only the JSON object (no extra text, explanation, or markdown).
- Ensure that the keys are exactly "variables", "constraints", and "rules".
- Do not include any inline comments in the output.
- Do not confuse between noinc and noin(val1; val2)
"""

DOMAIN_GENERATOR_FIX_SYSTEM_PROMPT = """
You are tasked with updating an existing JSON specification for domain rules in a machine‑learning model validation project.

Below is the YAML you generated previously:

{previous_yaml} 

Below are the user’s instructions for how to change it:
{user_instr}

You are also provided with the following inputs:

-------------------
Variables Information:
{variables_info}

Constraints Information:
{constraints_info}

Rules Information:
{rules_info}
-------------------

Using the schema below, produce a valid JSON object, identical to the original *except* for the fields needed to satisfy 
the schema and user’s instructions.
By addressing the user’s instructions (do not wrap it in triple backticks or code blocks) containing only the keys "variables", 
"constraints", and "rules":

{{
  "variables": {{
    "<variable_name>": {{
      "description": "<meaning>",
      "type": "<type>",          // Allowed values: "INT", "FLOAT", "CAT"
      "range": [min_value, max_value],    // Required for INT and FLOAT types. Do NOT include "range" key for CAT type.
      "values": ["Val1", "Val2", "Val3"],   // Required for CAT type. Do NOT include "values" key for INT and FLOAT types.
      "variation_limits": [min_value, max_value],  // Floats between 0 and 1. Do not use it for the Target Variable
      "insignificant_variation": value             // Float between 0 and 1
    }},
    ...
  }},
  "constraints": {{
    "<constraint_name>": {{
      "description": "<meaning>",
      "formula": "<formula>"     // A Python expression representing a condition
    }},
    ...
  }},
  "rules": {{
    "<rule_name>": {{
      "description": "<meaning>",
      "premises": {{
         "<feature_name>": "<behavior>"   // Allowed behaviors: inc, dec, cst, noinc, nodec, eq(value), noeq(value), in(val1; val2), or noin(val1; val2)
                                         // Note: For eq, noeq, in, and noin, the value(s) must be either a numeric value (int or float, without quotes) or a quoted string (e.g., 'example')
      }},
      "conclusion": {{
         "<target_name>": "<behavior>"     // Allowed behaviors: inc, dec, cst, noinc, nodec
      }}
    }},
    ...
  }}
}}

Instructions:
- Output only valid YAML with correct formatting.
- Do not include any extra text, markdown formatting, or code fences.
"""

##############################################################################
# DOMAIN GRAPH QUERY
##############################################################################

DOMAIN_GRAPH_QUERY = """
You are tasked with generating a complete YAML specification of domain rules for predicting the Target Variable. 
Use both the domain knowledge extracted from PDFs and the analysis of the DataFrame to inform your output. 
Specifically, you should:
- Search the PDFs to identify Feature Variable (and their descriptions) and any domain-specific constraints and rules.
- Analyze the DataFrame to determine feature correlations, ranges, and statistical insights that may influence the 
    Target Variable.
- Incorporate any inferred or extracted constraints (conditions that must always hold true) and domain rules 
    (relationships between features and the target variable with defined premises and a conclusion).

When your investigation is complete, produce the final YAML output containing the following sections:
- **variables:** with detailed definitions (including descriptions, types, ranges, values, and additional metadata).
- **constraints:** listing conditions that must always be met.
- **rules:** specifying the relationship between features and the target variable using defined behaviors 
    (e.g., inc, dec, etc.).

Ensure that the final output strictly adheres to the provided YAML template, is valid YAML with correct indentation, 
and contains only the raw YAML (no extra formatting or code blocks).
"""
