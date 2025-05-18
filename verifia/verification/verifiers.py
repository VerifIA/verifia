import os
import logging
import warnings
from typing import List, Tuple, Optional, Dict, Union, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..context.domain import Domain, VarType
from ..context.data import Dataset
from ..models import BaseModel, build_from_model_card
from .verifs import VerifTask, VerifAssertion
from .searchers import Searcher
from .results import RulesViolationResult, RulesViolationRun, OriginalStatistics
from ..utils.helpers.ds import create_progress_bar, read_data_file
from ..utils.helpers.io import read_config_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)                
logger.addHandler(logging.NullHandler())


class RuleConsistencyVerifier:
    """
    Manages the verification of model rules based on a domain definition and search strategy.

    Evaluates a dataset against specified rules, collects statistics, and generates detailed reports.
    """

    def __init__(
        self,
        domain_cfg_dict: Optional[Dict] = None,
        domain_cfg_fpath: Optional[os.PathLike] = None,
    ) -> None:
        """
        Initialize the verifier using a domain configuration file.
        Provide either a model instance or a domain configuration file path to load the domain.

        Args:
            domain_cfg_dict (Optional[Dict]): A dictionary of a domain configuration.
            domain_cfg_fpath (Optional[os.PathLike]): Path to a domain configuration YAML file.
        """
        if domain_cfg_dict is None and domain_cfg_fpath is None:
            raise ValueError(
                "You must provide either 'domain_cfg_dict' or 'domain_cfg_fpath' argument."
            )
        self.domain: Domain = (
            Domain.build(domain_cfg_fpath)
            if domain_cfg_dict is None
            else Domain.build(domain_cfg_dict)
        )
        self.result: Optional[RulesViolationResult] = None
        self.search_algo: str = ""
        self.search_params: Dict[str, Any] = {}
        self.searcher: Optional[Searcher] = None
        self.model: Optional[BaseModel] = None
        self.dataset: Optional[Dataset] = None
        self.original_predictions: List[float] = []
        self.original_ids: List[Any] = []

    def _load_original_seed(
        self, X_filtered: npt.NDArray, curr_run: RulesViolationRun
    ) -> None:
        """
        Record the base predictions and corresponding identifiers for each sample in the filtered data.

        Args:
            X_filtered (npt.NDArray): Filtered input data.
            curr_run (RulesViolationRun): The current verification run.
        """
        self.original_predictions = []
        self.original_ids = []
        for input_features in X_filtered:
            logger.debug("Processing input features: %s", input_features)
            original_prediction = self.model.predict_score(input_features)
            self.original_predictions.append(original_prediction)
            logger.debug("Base value computed: %s", original_prediction)
            original_id = curr_run.add_original(input_features, original_prediction)
            self.original_ids.append(original_id)

    def _filter_inputs(
        self, orig_input_seed: Dataset
    ) -> Tuple[npt.NDArray, OriginalStatistics]:
        """
        Filter the dataset based on domain constraints and model predictions.

        This method removes rows:
          1. With any feature value outside its allowed domain.
          2. With predictions that deviate from ground truth (error beyond tolerance for regression,
             or misclassification for classification).

        Args:
            orig_input_seed (Dataset): The original dataset.

        Returns:
            Tuple[npt.NDArray, OriginalStatistics]:
                - Filtered feature data as a NumPy array.
                - Statistics about the filtering process.
        """

        def _filter_out_of_domain(row: pd.Series) -> bool:
            for colname, value in row.items():
                var = self.domain.find_var(colname)
                if var.type == VarType.CAT:
                    value_idx = var.index(value)
                    if value_idx < var.min or value_idx > var.max:
                        return False
                else:
                    if value < var.min or value > var.max:
                        return False
            return True

        orig_stats = OriginalStatistics()
        orig_input_df = orig_input_seed.data
        original_row_count = len(orig_input_df)
        filtered_df = orig_input_df[
            orig_input_df.apply(_filter_out_of_domain, axis=1)
        ].copy()
        filtered_row_count = len(filtered_df)
        orig_stats.n_orig = original_row_count
        orig_stats.n_ood = original_row_count - filtered_row_count
        logger.info(
            "Rows removed because they are out of domain: %s out of %s",
            orig_stats.n_ood,
            original_row_count,
        )

        features_filtered = filtered_df[self.model.feature_names].to_numpy()
        outs = self.model.predict(features_filtered)
        if self.model.is_regression:
            tolerance = self.domain.find_var(self.model.target_name).epsilon
            orig_stats.err_thresh = tolerance
            true_values = filtered_df[self.model.target_name].to_numpy()
            condition = np.abs(outs.predictions - true_values) <= tolerance
            valid_indices = np.where(condition)[0]
            filtered_df = filtered_df.iloc[valid_indices].copy()
            orig_stats.n_herr = filtered_row_count - len(filtered_df)
            logger.info(
                "In-Domain rows removed due to error > %.3f: %s out of %s",
                tolerance,
                orig_stats.n_herr,
                filtered_row_count,
            )
        elif self.model.is_classification:
            true_values = filtered_df[self.model.target_name].to_numpy()
            condition = outs.predictions == true_values
            valid_indices = np.where(condition)[0]
            filtered_df = filtered_df.iloc[valid_indices].copy()
            orig_stats.n_miscls = filtered_row_count - len(filtered_df)
            logger.info(
                "In-Domain rows removed due to misclassification: %s out of %s",
                orig_stats.n_miscls,
                filtered_row_count,
            )

        filtered_features = filtered_df[self.model.feature_names].to_numpy()
        mname, mscore = self.model.calculate_predictive_performance(orig_input_seed)
        orig_stats.metric_name = mname
        orig_stats.metric_score = mscore
        return filtered_features, orig_stats

    def calculate_dataset_statistics(self) -> OriginalStatistics:
        """
        Compute detailed statistics for the original dataset based on domain constraints and model predictions.

        This method performs the following steps:
            1. Validates that the model and dataset have been set. If not, a ValueError is raised instructing the user to call
                the appropriate setup methods (verify() for the model and on() for the dataset).
            2. Filters out rows containing any feature value that falls outside its allowed domain. This is done using an
                internal helper function that checks each row against the domain constraints.
            3. Records the total number of original rows (n_orig) and the number of rows removed because they are out-of-domain (n_ood).
            4. Computes the model's predictive performance on the entire dataset and stores the performance metric name and score.
            5. Further refines the filtered dataset based on the model's predictions:
                - For regression models:
                    * Retrieves the error tolerance (err_thresh) from the domain of the target variable.
                    * Removes rows where the absolute prediction error exceeds the tolerance.
                    * Records the count of in-domain rows removed due to high error (n_herr).
                - For classification models:
                    * Removes rows where the model's predictions do not match the true target values.
                    * Records the count of in-domain rows removed due to misclassification (n_miscls).

        Returns:
            OriginalStatistics: An object containing:
                - n_orig: Total number of rows in the original dataset.
                - n_ood: Number of rows removed because they are out-of-domain.
                - n_herr: For regression, number of rows removed due to prediction error exceeding the tolerance.
                - n_miscls: For classification, number of rows removed due to misclassification.
                - metric_name: The name of the performance metric used.
                - metric_score: The score of the performance metric.
                - err_thresh: For regression, the error tolerance threshold applied.

        Raises:
            ValueError: If the model or dataset has not been set.
        """

        def _filter_out_of_domain(row: pd.Series) -> bool:
            for colname, value in row.items():
                var = self.domain.find_var(colname)
                if var.type == VarType.CAT:
                    value_idx = var.index(value)
                    if value_idx < var.min or value_idx > var.max:
                        return False
                else:
                    if value < var.min or value > var.max:
                        return False
            return True

        # Check that required configurations are set
        if self.model is None:
            raise ValueError(
                "Model is not set. Please call verify() to set up the model before calling run()."
            )
        if self.dataset is None:
            raise ValueError(
                "Dataset is not set. Please call on() to set up the dataset before calling run()."
            )

        orig_stats = OriginalStatistics()
        mname, mscore = self.model.calculate_predictive_performance(self.dataset)
        orig_stats.metric_name = mname
        orig_stats.metric_score = mscore

        orig_input_df = self.dataset.data
        original_row_count = len(orig_input_df)
        filtered_df = orig_input_df[
            orig_input_df.apply(_filter_out_of_domain, axis=1)
        ].copy()
        filtered_row_count = len(filtered_df)
        orig_stats.n_orig = original_row_count
        orig_stats.n_ood = original_row_count - filtered_row_count
        logger.info(
            "Rows removed because they are out of domain: %s out of %s",
            orig_stats.n_ood,
            original_row_count,
        )

        features_filtered = filtered_df[self.model.feature_names].to_numpy()
        outs = self.model.predict(features_filtered)
        if self.model.is_regression:
            tolerance = self.domain.find_var(self.model.target_name).epsilon
            orig_stats.err_thresh = tolerance
            true_values = filtered_df[self.model.target_name].to_numpy()
            condition = np.abs(outs.predictions - true_values) <= tolerance
            valid_indices = np.where(condition)[0]
            filtered_df = filtered_df.iloc[valid_indices].copy()
            orig_stats.n_herr = filtered_row_count - len(filtered_df)
            logger.info(
                "In-Domain rows removed due to error > %.3f: %s out of %s",
                tolerance,
                orig_stats.n_herr,
                filtered_row_count,
            )
        elif self.model.is_classification:
            true_values = filtered_df[self.model.target_name].to_numpy()
            condition = outs.predictions == true_values
            valid_indices = np.where(condition)[0]
            filtered_df = filtered_df.iloc[valid_indices].copy()
            orig_stats.n_miscls = filtered_row_count - len(filtered_df)
            logger.info(
                "In-Domain rows removed due to misclassification: %s out of %s",
                orig_stats.n_miscls,
                filtered_row_count,
            )

        return orig_stats

    def verify(
        self,
        model: Optional[BaseModel] = None,
        model_card_fpath_or_dict: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> "RuleConsistencyVerifier":
        """
        Set up the model for verification.

        Provide either a model instance or a model card file path to build the model.

        Args:
            model (Optional[BaseModel]): An instance of a model.
            model_card_fpath (Optional[os.PathLike]): Path to a model card YAML file.

        Returns:
            RuleConsistencyVerifier: Self, to allow method chaining.

        Raises:
            ValueError: If neither model nor model_card_fpath is provided.
        """
        if model is None and model_card_fpath_or_dict is None:
            raise ValueError(
                "You must provide either 'model' or 'model_card_fpath_or_dict' argument."
            )
        self.model = (
            model
            if model is not None
            else build_from_model_card(model_card_fpath_or_dict)
        )
        self.result = RulesViolationResult(self.model.model_card, self.domain)
        return self

    def on(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        data_fpath: Optional[os.PathLike] = None,
        dataset: Optional[Dataset] = None,
    ) -> "RuleConsistencyVerifier":
        """
        Set the dataset to be verified.

        Provide either a Dataset, a DataFrame, or a file path to the data.

        Args:
            dataframe (Optional[pd.DataFrame]): A pandas DataFrame.
            data_fpath (Optional[os.PathLike]): File path to the data.
            dataset (Optional[Dataset]): A pre-constructed Dataset object.

        Returns:
            RuleConsistencyVerifier: Self, to allow method chaining.

        Raises:
            ValueError: If none of the Dataset, DataFrame, or file path is provided.
                        If the model is not set, instructs the user to call verify() first.
        """
        if self.model is None:
            raise ValueError(
                "Model is not set. Please call verify() to set up the model before calling on()."
            )

        if dataset is None and dataframe is None and data_fpath is None:
            raise ValueError(
                "You must provide either 'dataset', 'dataframe' or 'data_fpath' argument."
            )

        if dataset is None:
            if dataframe is None:
                dataframe = read_data_file(data_fpath)
            dataset = Dataset(
                dataframe,
                self.model.target_name,
                self.model.feature_names,
                self.model.cat_feature_names,
            )
        self.dataset = dataset
        return self

    def using(
        self,
        search_algo: str,
        search_params: Optional[dict] = None,
        search_params_fpath: Optional[os.PathLike] = None,
    ) -> "RuleConsistencyVerifier":
        """
        Specify the search algorithm and parameters for verification.

        Args:
            search_algo (str): The identifier of the search algorithm.
            search_params (Optional[dict]): A dictionary of search parameters.
            search_params_fpath (Optional[os.PathLike]): Path to a configuration file for search parameters.

        Returns:
            RuleConsistencyVerifier: Self, to allow method chaining.

        Warns:
            UserWarning: If no search parameters are provided.
        """
        if search_params is None and search_params_fpath is None:
            warnings.warn(
                "No search parameters provided. Using default settings.", UserWarning
            )
        if search_params is None:
            search_params = (
                read_config_file(search_params_fpath)
                if search_params_fpath is not None
                else {}
            )
        self.search_algo = search_algo
        self.search_params = search_params
        self.searcher = Searcher.build(search_algo, search_params)
        return self

    def run(
        self,
        pop_size: int,
        max_iters: int,
        orig_seed_ratio: Optional[float] = None,
        orig_seed_size: Optional[int] = None,
        persistance: bool = True,
    ) -> RulesViolationResult:
        """
        Execute the verification run.

        The method performs the following steps:
            - Validates input parameters.
            - Samples the original dataset.
            - Filters out rows violating domain constraints.
            - Loads original seed predictions.
            - Iterates over rules and original inputs to search for rule violations.
            - Records any inconsistent candidates.
            - Persists the results if requested.

        Args:
            pop_size (int): Population size for the search algorithm.
            max_iters (int): Maximum iterations for the search.
            orig_seed_ratio (Optional[float]): Ratio of original seed samples to use.
            orig_seed_size (Optional[int]): Number of original seed samples to use.
            persistance (bool, optional): Whether to persist the run results. Defaults to True.

        Returns:
            RulesViolationResult: The final verification result.

        Raises:
            TypeError: If pop_size or max_iters are not integers, or if seed parameters have incorrect types.
            ValueError: If pop_size or max_iters are out of valid ranges, or if neither seed parameter is provided.
                        Also if the model, dataset, or searcher have not been set.
        """
        # Check that required configurations are set
        if self.model is None:
            raise ValueError(
                "Model is not set. Please call verify() to set up the model before calling run()."
            )
        if self.dataset is None:
            raise ValueError(
                "Dataset is not set. Please call on() to set up the dataset before calling run()."
            )
        if self.searcher is None:
            raise ValueError(
                "Searcher is not set. Please call using() to set up the searcher before calling run()."
            )

        # Validate input parameters
        if not isinstance(pop_size, int) or not isinstance(max_iters, int):
            raise TypeError("Both pop_size and max_iters must be integers.")
        if pop_size < 2:
            raise ValueError(f"pop_size must be at least 2, got {pop_size}.")
        if max_iters < 1:
            raise ValueError(f"max_iters must be at least 1, got {max_iters}.")
        if orig_seed_size is not None and not isinstance(orig_seed_size, int):
            raise TypeError("orig_seed_size must be an integer.")
        if orig_seed_ratio is not None and not isinstance(orig_seed_ratio, float):
            raise TypeError("orig_seed_ratio must be a float.")
        if orig_seed_ratio is None and orig_seed_size is None:
            logger.info(
                "No orig_seed_ratio or orig_seed_size provided; \
                        using the entire original dataset as seed (orig_seed_ratio=1.0)."
            )
            orig_seed_ratio = 1.0

        if not self.model.has_model:
            try:
                self.model.load_model_from_registry()
            except Exception as e:
                logger.warning(
                    "Failed to load model from registry (%s). Falling back to load_model locally.",
                    e,
                )
                self.model.load_model()

        run_result: RulesViolationRun = self.result.create_new_run(
            self.search_algo, self.search_params, pop_size, max_iters
        )

        orig_seed_ds = self.dataset.sample(orig_seed_size, orig_seed_ratio)
        X_filtered, orig_stats = self._filter_inputs(orig_seed_ds)
        run_result.set_orig_stats(orig_stats)

        # Return early if no valid data remains.
        if X_filtered.size == 0:
            return self.result

        self._load_original_seed(X_filtered, run_result)
        rule_names = [
            rule.name for rule in self.domain.get_rules(self.model.target_name)
        ]
        for rule_id in create_progress_bar(rule_names, desc="Processing Rules"):
            run_result.curr_rule_id = rule_id
            verif_task = VerifTask.build(rule_id, self.domain, self.model.model_card)
            verif_assertion = VerifAssertion.build(rule_id, self.domain, self.model)
            for idx, input_features in create_progress_bar(
                enumerate(X_filtered), desc="Processing Original Inputs"
            ):
                exp_ft_inp, cst_ft_inp = verif_task.partition_original_input_features(
                    input_features
                )
                lower_bound, upper_bound = verif_task.define_search_space_bounds(
                    exp_ft_inp
                )
                logger.debug("Local search bounds: lower=%s, upper=%s", lower_bound, upper_bound)
                original_id, original_prediction = (
                    self.original_ids[idx],
                    self.original_predictions[idx],
                )
                derived_candidates = self.searcher.init(
                    pop_size, max_iters, lower_bound, upper_bound
                )
                while derived_candidates.size > 0:
                    derived_candidates = verif_task.filter_derived_inputs(
                        derived_candidates, exp_ft_inp
                    )
                    derived_data = verif_task.assemble_derived_data_features(
                        derived_candidates, cst_ft_inp
                    )
                    are_feasible, compliancy_scores, constraints_stats = (
                        verif_task.assess_constraints_compliance(derived_data)
                    )
                    logger.debug("Domain-Compliant: feasible=%s, compliancy-scores=%s", are_feasible, compliancy_scores)
                    run_result.update_constraints_stats(constraints_stats)
                    are_consistent, deviations, predictions = verif_assertion.verify(
                        derived_data, original_prediction
                    )
                    logger.debug("Verification: are_consistent=%s, deviations=%s, predictions=%s",
                                 are_consistent, deviations, predictions)
                    are_inconsistent = np.logical_and(
                        are_feasible, np.logical_not(are_consistent)
                    )
                    logger.info("Detected Rule %s Violations: %s", rule_id, deviations[are_inconsistent])
                    run_result.add_inconsistent(
                        input_features,
                        derived_data[are_inconsistent],
                        predictions[are_inconsistent],
                        original_id,
                        original_prediction,
                        deviations[are_inconsistent],
                    )
                    derived_candidates = self.searcher.next_gen(
                        derived_candidates, compliancy_scores, deviations
                    )
        run_result.clean_duplicated_violations()
        if persistance:
            run_result.persist()
        return self.result

    def clean_results(self) -> None:
        """
        Remove all previously generated verification results.

        This will delete the directory where rule‐violation reports and checkpoints
        have been stored. You must have already run a verification (i.e., called verify()) before cleaning, 
        otherwise no results will be available.

        Raises:
            ValueError: If no results are available (i.e., verify() has not been called).
        """
        if self.result is None:
            raise ValueError(
                "Results are not available. Please call verify() to set up the model and results before cleaning."
            )
        self.result.clean()
