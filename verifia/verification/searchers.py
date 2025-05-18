from abc import ABC, abstractmethod
from enum import Enum
import random as rand
import logging
import math

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import numpy.typing as npt

from ..utils.helpers.math import softmax, roulette_wheel_selection, normalization
from ..utils.enums.verification import SearchAlgorithm

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Searcher(ABC):
    """
    Abstract base class for search algorithms.

    Provides common functionality for population initialization, normalization,
    evolution, and bounds management.
    """

    def __init__(self) -> None:
        self.pop_size: int = 0
        self.max_iters: int = 0
        self.curr_iter: int = 0
        self.vect_size: int = 0
        self.orig_lower_bound: Optional[npt.NDArray] = None
        self.orig_upper_bound: Optional[npt.NDArray] = None
        self.lower_bound: Optional[npt.NDArray] = None
        self.upper_bound: Optional[npt.NDArray] = None

    def init(
        self,
        pop_size: int,
        max_iters: int,
        lower_bound: npt.NDArray,
        upper_bound: npt.NDArray,
    ) -> npt.NDArray:
        """
        Initialize a random population within the given bounds.

        Args:
            pop_size (int): Population size.
            max_iters (int): Maximum number of iterations.
            lower_bound (npt.NDArray): Original lower bounds (vector).
            upper_bound (npt.NDArray): Original upper bounds (vector).

        Returns:
            npt.NDArray: Denormalized initial population.
        """
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.set_orig_bounds(lower_bound, upper_bound)
        self.curr_iter = 0
        population = self.get_rand_population()
        return self.denormalize_population(population)

    def next_gen(
        self,
        Pcurrent: npt.NDArray,
        valid_scores: npt.NDArray,
        derivation_errors: npt.NDArray,
    ) -> npt.NDArray:
        """
        Generate the next population based on current population, scores, and errors.

        Fitness is computed as the negative product of valid_scores and derivation_errors
        (so that higher scores and lower errors yield higher fitness).

        Args:
            Pcurrent (npt.NDArray): Current normalized population.
            valid_scores (npt.NDArray): Validity scores (fitness) for each candidate.
            derivation_errors (npt.NDArray): Derivation errors for each candidate.

        Returns:
            npt.NDArray: Denormalized next population.
        """
        fitness = -1 * (valid_scores * derivation_errors)
        Pcurrent = self.normalize_population(Pcurrent)
        Pnext = self.derive_next_population(Pcurrent, fitness)
        if Pnext.size == 0:
            return Pnext
        return self.denormalize_population(Pnext)

    @abstractmethod
    def derive_next_population(
        self, Pcurrent: npt.NDArray, fitness: npt.NDArray
    ) -> npt.NDArray:
        """
        Generate the next generation from the current population and fitness.

        Args:
            Pcurrent (npt.NDArray): Current normalized population.
            fitness (npt.NDArray): Fitness scores for each candidate.

        Returns:
            npt.NDArray: Next normalized population.
        """
        raise NotImplementedError(
            "Subclasses should implement derive_next_population()."
        )

    def set_orig_bounds(
        self, original_lower_bound: npt.NDArray, original_upper_bound: npt.NDArray
    ) -> None:
        """
        Set the original bounds and initialize normalized bounds.

        Args:
            original_lower_bound (npt.NDArray): Original lower bounds.
            original_upper_bound (npt.NDArray): Original upper bounds.

        Raises:
            ValueError: If shapes mismatch, lower_bound > upper_bound, or bounds are identical.
        """
        if original_lower_bound.shape != original_upper_bound.shape:
            raise ValueError(
                "Original lower and upper bounds must have the same shape."
            )
        if np.any(original_lower_bound > original_upper_bound):
            raise ValueError(
                "Each element in original_lower_bound must be ≤ corresponding element in original_upper_bound."
            )
        if np.array_equal(original_lower_bound, original_upper_bound):
            raise ValueError("Original lower and upper bounds must not be identical.")

        self.orig_lower_bound = original_lower_bound
        self.orig_upper_bound = original_upper_bound
        self.vect_size = len(original_upper_bound)
        self.lower_bound = -np.ones_like(original_lower_bound)
        self.upper_bound = np.ones_like(original_upper_bound)

    def bound_candidate(self, candidate: npt.NDArray) -> npt.NDArray:
        """
        Clip a candidate's values to the normalized bounds.

        Args:
            candidate (npt.NDArray): Candidate solution vector.

        Returns:
            npt.NDArray: Candidate vector clipped to [lower_bound, upper_bound].
        """
        return np.clip(candidate, self.lower_bound, self.upper_bound)

    def bound_population(self, population: npt.NDArray) -> npt.NDArray:
        """
        Clip an entire population's values to the normalized bounds.

        Args:
            population (npt.NDArray): Population array.

        Returns:
            npt.NDArray: Population array with each candidate clipped to [lower_bound, upper_bound].
        """
        L_b = np.repeat(np.expand_dims(self.lower_bound, axis=0), self.pop_size, axis=0)
        U_b = np.repeat(np.expand_dims(self.upper_bound, axis=0), self.pop_size, axis=0)
        return np.clip(population, L_b, U_b)

    def normalize_population(self, population: npt.NDArray) -> npt.NDArray:
        """
        Normalize the population from original scale to normalized scale.

        Args:
            population (npt.NDArray): Population on original scale.

        Returns:
            npt.NDArray: Normalized population.
        """
        divider = self.orig_upper_bound - self.orig_lower_bound
        multiplier = self.upper_bound - self.lower_bound
        normalized_pop = (population - self.orig_lower_bound) / divider
        normalized_pop = normalized_pop * multiplier + self.lower_bound
        return normalized_pop

    def denormalize_population(self, normalized_pop: npt.NDArray) -> npt.NDArray:
        """
        Convert the population from normalized scale back to original scale.

        Args:
            normalized_pop (npt.NDArray): Normalized population.

        Returns:
            npt.NDArray: Denormalized population.
        """
        divider = self.upper_bound - self.lower_bound
        multiplier = self.orig_upper_bound - self.orig_lower_bound
        denormalized_pop = (normalized_pop - self.lower_bound) / divider
        denormalized_pop = denormalized_pop * multiplier + self.orig_lower_bound
        return denormalized_pop

    def get_rand_candidate(self) -> npt.NDArray:
        """
        Generate a random candidate within normalized bounds.

        Returns:
            npt.NDArray: A random candidate vector.
        """
        return np.random.uniform(low=self.lower_bound, high=self.upper_bound)

    def get_rand_population(self) -> npt.NDArray:
        """
        Generate a random population of candidates.

        Returns:
            npt.NDArray: Random population array of shape (pop_size, vect_size).
        """
        population = np.zeros((self.pop_size, self.vect_size))
        for i in range(self.pop_size):
            population[i, :] = self.get_rand_candidate()
        return population

    @staticmethod
    def build(
        search_algo: str, search_params: Optional[Dict[str, Any]] = None
    ) -> "Searcher":
        """
        Build a Searcher instance based on the specified algorithm.

        Args:
            search_algo (str): Identifier of the search algorithm.
            search_params (Optional[Dict[str, Any]]): Parameters for the search algorithm.

        Returns:
            Searcher: An instance of a search algorithm.

        Raises:
            ValueError: If the specified search algorithm is not supported.
        """
        algorithm = SearchAlgorithm(search_algo.upper())
        if algorithm == SearchAlgorithm.RS:
            searcher = RS(search_params)
        elif algorithm == SearchAlgorithm.FFA:
            searcher = FFA(search_params)
        elif algorithm == SearchAlgorithm.MFO:
            searcher = MFO(search_params)
        elif algorithm == SearchAlgorithm.GWO:
            searcher = GWO(search_params)
        elif algorithm == SearchAlgorithm.MVO:
            searcher = MVO(search_params)
        elif algorithm == SearchAlgorithm.PSO:
            searcher = PSO(search_params)
        elif algorithm == SearchAlgorithm.GA:
            searcher = GA(search_params)
        elif algorithm == SearchAlgorithm.WOA:
            searcher = WOA(search_params)
        elif algorithm == SearchAlgorithm.SSA:
            searcher = SSA(search_params)
        else:
            raise ValueError(f"The searching algorithm {search_algo} is not supported.")
        logger.info("Searcher built with algorithm: %s", search_algo)
        return searcher


# --- Concrete Search Algorithms --- #


class RS(Searcher):
    """
    Random Sampler (RS) search algorithm.

    Generates the next population entirely at random.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()

    def derive_next_population(
        self, Pcurrent: npt.NDArray, fitness: npt.NDArray
    ) -> npt.NDArray:
        """
        Derive the next population by generating new random candidates.

        Args:
            Pcurrent (npt.NDArray): Current normalized population.
            fitness (npt.NDArray): Fitness values (unused in RS).

        Returns:
            npt.NDArray: Next normalized population.
        """
        self.curr_iter += 1
        if self.curr_iter >= self.max_iters:
            return np.empty((0, self.vect_size))
        new_population = self.get_rand_population()
        return self.bound_population(new_population)


class FFA(Searcher):
    """
    Firefly Algorithm (FFA) search algorithm.
    """

    @dataclass
    class Params:
        beta_min: float = 1.0
        gamma: float = 1.0
        theta: float = 0.97
        time: float = 0.5
        alpha: float = 1.0
        _moving_alpha: float = 1.0

        @property
        def moving_alpha(self) -> float:
            return self._moving_alpha * self.theta

        def from_dict(self, params_dict: Dict[str, Any]) -> "FFA.Params":
            self.beta_min = params_dict.get("beta_min", self.beta_min)
            self.gamma = params_dict.get("gamma", self.gamma)
            self.theta = params_dict.get("theta", self.theta)
            self.time = params_dict.get("time", self.time)
            self.alpha = params_dict.get("alpha", self.alpha)
            return self

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params: FFA.Params = FFA.Params().from_dict(params)

    def derive_next_population(
        self, Pcurrent: npt.NDArray, fitness: npt.NDArray
    ) -> npt.NDArray:
        """
        Derive the next population using the Firefly Algorithm.

        Args:
            Pcurrent (npt.NDArray): Current normalized population.
            fitness (npt.NDArray): Fitness scores.

        Returns:
            npt.NDArray: Next normalized population.
        """
        self.curr_iter += 1
        if self.curr_iter >= self.max_iters:
            return np.empty((0, self.vect_size))

        sorted_indices = np.argsort(-fitness)
        light = fitness[sorted_indices]
        Psorted = Pcurrent[sorted_indices]
        Pnext = np.copy(Pcurrent)
        alpha = self.params.moving_alpha
        scale = np.abs(self.upper_bound - self.lower_bound)
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                if light[i] < light[j]:
                    r_squared = np.sum((Pnext[i] - Pnext[j]) ** 2)
                    beta = self.params.beta_min * np.exp(-self.params.gamma * r_squared)
                    step = (
                        alpha
                        * (np.random.randn(self.vect_size) - self.params.time)
                        * scale
                    )
                    Pnext[i] += beta * (Psorted[j] - Pnext[i]) + step
                    Pnext[i] = self.bound_candidate(Pnext[i])
        return Pnext


class GA(Searcher):
    """
    Genetic Algorithm (GA) search algorithm.
    """

    @dataclass
    class Params:
        class CrossoverType(Enum):
            UNIFORM = "uniform"
            ONE_POINT = "one_point"
            TWO_POINT = "two_point"

        mutation_prob: float = 0.5
        elite_ratio: float = 0.01
        crossover_prob: float = 0.5
        parents_ratio: float = 0.3
        crossover_op: "GA.Params.CrossoverType" = CrossoverType.UNIFORM

        def parent_size(self, pop_size: int) -> int:
            parent_size = int(self.parents_ratio * pop_size)
            if (pop_size - parent_size) % 2 != 0:
                parent_size += 1
            return parent_size

        def elite_size(self, pop_size: int) -> int:
            if self.elite_ratio <= 0:
                return 0
            return max(1, int(pop_size * self.elite_ratio))

        def from_dict(self, params_dict: Dict[str, Any]) -> "GA.Params":
            self.mutation_prob = params_dict.get("mutation_prob", self.mutation_prob)
            self.elite_ratio = params_dict.get("elite_ratio", self.elite_ratio)
            self.crossover_prob = params_dict.get("crossover_prob", self.crossover_prob)
            self.parents_ratio = params_dict.get("parents_ratio", self.parents_ratio)
            self.crossover_op = self.CrossoverType(
                params_dict.get("crossover_op", self.crossover_op)
            )
            return self

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params: GA.Params = GA.Params().from_dict(params)

    def cross(
        self, parent1: npt.NDArray, parent2: npt.NDArray, crossover_op: str
    ) -> npt.NDArray:
        """
        Perform crossover between two parent candidates.

        Args:
            parent1 (npt.NDArray): First parent's data.
            parent2 (npt.NDArray): Second parent's data.
            crossover_op (str): Crossover operator ('one_point', 'two_point', 'uniform').

        Returns:
            npt.NDArray: Array containing two offspring.
        """
        ofs1 = parent1.copy()
        ofs2 = parent2.copy()
        if crossover_op == GA.Params.CrossoverType.ONE_POINT:
            cp = np.random.randint(0, self.vect_size)
            ofs1[:cp] = parent2[:cp]
            ofs2[:cp] = parent1[:cp]
        elif crossover_op == GA.Params.CrossoverType.TWO_POINT:
            cp1 = np.random.randint(0, self.vect_size)
            cp2 = np.random.randint(cp1, self.vect_size)
            ofs1[cp1:cp2] = parent2[cp1:cp2]
            ofs2[cp1:cp2] = parent1[cp1:cp2]
        elif crossover_op == GA.Params.CrossoverType.UNIFORM:
            mask = np.random.choice([True, False], size=parent1.shape, p=[0.5, 0.5])
            ofs1[mask] = parent2[mask]
            ofs2[mask] = parent1[mask]
        else:
            raise ValueError(f"Unsupported crossover operator: {crossover_op}")
        return np.array([ofs1, ofs2])

    def mutate(self, candidate: npt.NDArray) -> npt.NDArray:
        """
        Mutate a candidate solution by randomly perturbing its elements.

        Args:
            candidate (npt.NDArray): Candidate solution vector.

        Returns:
            npt.NDArray: Mutated candidate.
        """
        mask = np.random.choice(
            [True, False],
            size=candidate.shape,
            p=[self.params.mutation_prob, 1 - self.params.mutation_prob],
        )
        for i in range(self.vect_size):
            indices = np.where(mask[:, i])[0]
            scaler = self.upper_bound[i] - self.lower_bound[i]
            candidate[indices, i] = self.lower_bound[i] + scaler * np.random.random(
                size=len(indices)
            )
        return candidate

    def derive_next_population(
        self, Pcurrent: npt.NDArray, fitness: npt.NDArray
    ) -> npt.NDArray:
        """
        Derive the next population using Genetic Algorithm operators.

        Args:
            Pcurrent (npt.NDArray): Current normalized population.
            fitness (npt.NDArray): Fitness scores for each candidate.

        Returns:
            npt.NDArray: Next normalized population.
        """
        self.curr_iter += 1
        if self.curr_iter >= self.max_iters:
            return np.empty((0, self.vect_size))
        parent_size = self.params.parent_size(self.pop_size)
        elite_size = self.params.elite_size(self.pop_size)
        Pnext = np.zeros_like(Pcurrent)
        best_indices = np.argsort(fitness)[::-1]
        Pnext[:elite_size] = Pcurrent[best_indices[:elite_size]].copy()
        probes = softmax(fitness[best_indices])
        cum_probes = np.cumsum(probes)
        num_parents = parent_size - elite_size
        rand_vals = np.random.random(size=num_parents)
        selected_indices = np.searchsorted(cum_probes, rand_vals)
        Pnext[elite_size:parent_size] = Pcurrent[best_indices][selected_indices].copy()
        parents = Pnext[:parent_size]
        par_count = max(
            np.count_nonzero(
                np.random.choice(
                    [True, False],
                    size=parent_size,
                    p=[self.params.crossover_prob, 1 - self.params.crossover_prob],
                )
            ),
            1,
        )
        crossover_parent = parents[
            np.random.choice(np.arange(parent_size), size=par_count, replace=True)
        ]
        num_offspring = self.pop_size - parent_size
        offspring = []
        for _ in range(num_offspring // 2):
            idx1, idx2 = np.random.randint(0, par_count, size=2)
            children = self.cross(
                crossover_parent[idx1 : idx1 + 1],
                crossover_parent[idx2 : idx2 + 1],
                self.params.crossover_op,
            )
            offspring.append(self.mutate(children[0]))
            offspring.append(self.mutate(children[1]))
        offspring = np.concatenate(offspring, axis=0)
        Pnext[parent_size : parent_size + offspring.shape[0]] = offspring
        return Pnext


class GWO(Searcher):
    """
    Grey Wolf Optimizer (GWO) search algorithm.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()

    def derive_next_population(
        self, Pcurrent: npt.NDArray, fitness: npt.NDArray
    ) -> npt.NDArray:
        """
        Derive the next population using GWO logic.

        Args:
            Pcurrent (npt.NDArray): Current normalized population.
            fitness (npt.NDArray): Fitness scores.

        Returns:
            npt.NDArray: Next normalized population.
        """
        self.curr_iter += 1
        if self.curr_iter >= self.max_iters:
            return np.empty((0, self.vect_size))
        sorted_indices = np.argsort(fitness)
        alpha_pos = Pcurrent[sorted_indices[0]]
        beta_pos = Pcurrent[sorted_indices[1]]
        delta_pos = Pcurrent[sorted_indices[2]]
        leader_pos = np.array([alpha_pos, beta_pos, delta_pos])
        a = 2 - self.curr_iter * (2 / self.max_iters)
        r1 = np.random.rand(3 * self.pop_size, self.vect_size)
        r2 = np.random.rand(3 * self.pop_size, self.vect_size)
        A = 2 * a * r1 - a
        C = 2 * r2
        repeated_leader = np.tile(leader_pos, (self.pop_size, 1))
        D = np.abs(C * repeated_leader - np.repeat(Pcurrent, 3, axis=0))
        X = repeated_leader - A * D
        Pnext = np.mean(X.reshape(self.pop_size, 3, self.vect_size), axis=1)
        return self.bound_population(Pnext)


class MFO(Searcher):
    """
    Moth-Flame Optimization (MFO) search algorithm.
    """

    @dataclass
    class Params:
        b: float = 1.0

        def from_dict(self, params_dict: Dict[str, Any]) -> "MFO.Params":
            self.b = params_dict.get("b", self.b)
            return self

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params: MFO.Params = MFO.Params().from_dict(params)

    def derive_next_population(
        self, Pcurrent: npt.NDArray, fitness: npt.NDArray
    ) -> npt.NDArray:
        """
        Derive the next population using MFO.

        Args:
            Pcurrent (npt.NDArray): Current normalized population.
            fitness (npt.NDArray): Fitness values.

        Returns:
            npt.NDArray: Next normalized population.
        """
        self.curr_iter += 1
        if self.curr_iter >= self.max_iters:
            return np.empty((0, self.vect_size))
        sorted_indices = np.argsort(-fitness)
        Psorted = Pcurrent[sorted_indices]
        n_flames = int(
            round(
                self.pop_size - self.curr_iter * ((self.pop_size - 1) / self.max_iters)
            )
        )
        a = -1 + self.curr_iter * (-1 / self.max_iters)
        t = (a - 1) * np.random.rand(self.pop_size, self.vect_size) + 1
        distance = np.abs(Psorted - Pcurrent)
        Pnext = np.copy(Pcurrent)
        if n_flames > 0:
            Pnext[:n_flames] = (
                distance[:n_flames]
                * np.exp(self.params.b * t[:n_flames])
                * np.cos(2 * t[:n_flames] * math.pi)
                + Psorted[:n_flames]
            )
        if n_flames < self.pop_size:
            Pnext[n_flames:] = (
                distance[n_flames:]
                * np.exp(self.params.b * t[n_flames:])
                * np.cos(2 * t[n_flames:] * math.pi)
                + Psorted[n_flames:]
            )
        return self.bound_population(Pnext)


class MVO(Searcher):
    """
    Multi-Verse Optimizer (MVO) search algorithm.
    """

    @dataclass
    class Params:
        WEP_max: float = 1.0
        WEP_min: float = 0.2
        TDR_p: float = 6.0

        def from_dict(self, params_dict: Dict[str, Any]) -> "MVO.Params":
            self.WEP_max = params_dict.get("WEP_max", self.WEP_max)
            self.WEP_min = params_dict.get("WEP_min", self.WEP_min)
            self.TDR_p = params_dict.get("TDR_p", self.TDR_p)
            return self

        def compute_WEP(self, curr_iter: int, max_iters: int) -> float:
            scaler = self.WEP_max - self.WEP_min
            return self.WEP_min + scaler * (curr_iter / max_iters)

        def compute_TDR(self, curr_iter: int, max_iters: int) -> float:
            pow_coef = 1 / self.TDR_p
            return 1 - (curr_iter**pow_coef / max_iters**pow_coef)

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params: MVO.Params = MVO.Params().from_dict(params)

    def derive_next_population(
        self, Pcurrent: npt.NDArray, fitness: npt.NDArray
    ) -> npt.NDArray:
        """
        Derive the next population using MVO.

        Args:
            Pcurrent (npt.NDArray): Current normalized population.
            fitness (npt.NDArray): Fitness values.

        Returns:
            npt.NDArray: Next normalized population.
        """
        self.curr_iter += 1
        if self.curr_iter >= self.max_iters:
            return np.empty((0, self.vect_size))
        WEP = self.params.compute_WEP(self.curr_iter, self.max_iters)
        TDR = self.params.compute_TDR(self.curr_iter, self.max_iters)
        sorted_indices = np.argsort(-fitness)
        sorted_Pcurrent = Pcurrent[sorted_indices]
        best_universe = sorted_Pcurrent[0]
        normalized_fitness = normalization(fitness[sorted_indices])
        Pnext = np.zeros_like(Pcurrent)
        Pnext[0] = sorted_Pcurrent[0]
        for i in range(self.pop_size):
            for j in range(self.vect_size):
                if rand.random() < normalized_fitness[i]:
                    idx = roulette_wheel_selection(-fitness)
                    idx = idx if idx != -1 else 0
                    Pnext[i, j] = sorted_Pcurrent[idx, j]
                if rand.random() < WEP:
                    multiplier = self.upper_bound[j] - self.lower_bound[j]
                    scaler = multiplier * rand.random() + self.lower_bound[j]
                    if rand.random() < 0.5:
                        Pnext[i, j] = best_universe[j] + scaler * TDR
                    else:
                        Pnext[i, j] = best_universe[j] - scaler * TDR
        return self.bound_population(Pnext)


class PSO(Searcher):
    """
    Particle Swarm Optimization (PSO) search algorithm.
    """

    @dataclass
    class Params:
        c_1: float = 2.0
        c_2: float = 2.0
        w_max: float = 0.9
        w_min: float = 0.2
        V_max: Optional[npt.NDArray] = None
        V: Optional[npt.NDArray] = None
        p_best_fit: Optional[npt.NDArray] = None
        p_best_Pos: Optional[npt.NDArray] = None

        def from_dict(self, params_dict: Dict[str, Any]) -> "PSO.Params":
            self.c_1 = params_dict.get("c_1", self.c_1)
            self.c_2 = params_dict.get("c_2", self.c_2)
            self.w_max = params_dict.get("w_max", self.w_max)
            self.w_min = params_dict.get("w_min", self.w_min)
            return self

        def init_moving_params(self, pop_size: int, vect_size: int) -> None:
            shape = (pop_size, vect_size)
            self.V = np.zeros(shape)
            self.V_max = np.ones(shape)
            self.p_best_fit = np.full(pop_size, -np.inf)
            self.p_best_Pos = np.zeros(shape)

        def compute_w(self, curr_iter: int, max_iters: int) -> float:
            scaler = self.w_max - self.w_min
            return self.w_max - scaler * (curr_iter / max_iters)

        def update_p_best_params(
            self, Pcurrent: npt.NDArray, fitness: npt.NDArray
        ) -> None:
            mask = fitness > self.p_best_fit
            self.p_best_fit = np.where(mask, fitness, self.p_best_fit)
            mask2 = np.repeat(mask[:, None], self.p_best_Pos.shape[1], axis=1)
            self.p_best_Pos = np.where(mask2, Pcurrent, self.p_best_Pos)

        def update_V(self, V: npt.NDArray) -> None:
            self.V = np.clip(V, -self.V_max, self.V_max)

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params: PSO.Params = PSO.Params().from_dict(params)

    def init(
        self,
        pop_size: int,
        max_iters: int,
        lower_bound: npt.NDArray,
        upper_bound: npt.NDArray,
    ) -> npt.NDArray:
        result = super().init(pop_size, max_iters, lower_bound, upper_bound)
        self.params.init_moving_params(self.pop_size, self.vect_size)
        return result

    def derive_next_population(
        self, Pcurrent: npt.NDArray, fitness: npt.NDArray
    ) -> npt.NDArray:
        """
        Derive the next population using PSO.

        Args:
            Pcurrent (npt.NDArray): Current normalized population.
            fitness (npt.NDArray): Fitness scores for each candidate.

        Returns:
            npt.NDArray: Next normalized population.
        """
        self.curr_iter += 1
        if self.curr_iter >= self.max_iters:
            return np.empty((0, self.vect_size))
        self.params.update_p_best_params(Pcurrent, fitness)
        w = self.params.compute_w(self.curr_iter, self.max_iters)
        r_1 = np.random.rand(self.pop_size, self.vect_size)
        r_2 = np.random.rand(self.pop_size, self.vect_size)
        g_best_pos = self.params.p_best_Pos[np.argmax(self.params.p_best_fit)]
        diff_1 = self.params.p_best_Pos - Pcurrent
        diff_2 = g_best_pos - Pcurrent
        new_V = (
            w * self.params.V
            + self.params.c_1 * r_1 * diff_1
            + self.params.c_2 * r_2 * diff_2
        )
        Pnext = Pcurrent + new_V
        self.params.update_V(new_V)
        return self.bound_population(Pnext)


class SSA(Searcher):
    """
    Salp Swarm Algorithm (SSA) search algorithm.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()

    def derive_next_population(
        self, Pcurrent: npt.NDArray, fitness: npt.NDArray
    ) -> npt.NDArray:
        """
        Derive the next population using SSA.

        Args:
            Pcurrent (npt.NDArray): Current normalized population.
            fitness (npt.NDArray): Fitness scores.

        Returns:
            npt.NDArray: Next normalized population.
        """
        self.curr_iter += 1
        if self.curr_iter >= self.max_iters:
            return np.empty((0, self.vect_size))
        food_idx = np.argmax(fitness)
        food_pos = Pcurrent[food_idx]
        Pnext = Pcurrent.copy()
        c_1 = 2 * np.exp(-((4 * self.curr_iter / self.max_iters) ** 2))
        c_2 = np.random.rand(1, self.vect_size)
        c_3 = np.random.rand(1, self.vect_size)
        sign = 2 * (c_3 >= 0.5).astype(np.int32) - 1
        scaler = self.upper_bound - self.lower_bound
        Pnext[0] = food_pos + sign * c_1 * (scaler * c_2 + self.lower_bound)
        for s_idx in range(1, self.pop_size):
            Pnext[s_idx] = np.mean(Pnext[s_idx - 1 : s_idx + 1], axis=0)
        return self.bound_population(Pnext)


class WOA(Searcher):
    """
    Whale Optimization Algorithm (WOA) search algorithm.
    """

    @dataclass
    class Params:
        b: float = 1.0

        def from_dict(self, params_dict: Dict[str, Any]) -> "WOA.Params":
            self.b = params_dict.get("b", self.b)
            return self

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params: WOA.Params = WOA.Params().from_dict(params)

    def get_moving_params(self) -> float:
        return 2 - self.curr_iter * (2 / self.max_iters)

    def derive_next_population(
        self, Pcurrent: npt.NDArray, fitness: npt.NDArray
    ) -> npt.NDArray:
        """
        Derive the next population using WOA.

        Args:
            Pcurrent (npt.NDArray): Current normalized population.
            fitness (npt.NDArray): Fitness scores.

        Returns:
            npt.NDArray: Next normalized population.
        """
        self.curr_iter += 1
        if self.curr_iter >= self.max_iters:
            return np.empty((0, self.vect_size))
        sorted_indices = np.argsort(fitness)[::-1]
        ranked_pos = Pcurrent[sorted_indices]
        leader_pos = ranked_pos[0]
        Pnext = ranked_pos.copy()
        a = 2 * (1 - self.curr_iter / self.max_iters)
        for i in range(1, self.pop_size):
            if rand.random() < 0.5:
                r_a = np.random.rand(self.vect_size)
                A = 2 * a * r_a - a
                r_2 = np.random.rand(self.vect_size)
                C = 2 * r_2
                if np.linalg.norm(A) >= 1:
                    rand_leader_idx = int(math.floor(self.pop_size * rand.random()))
                    rand_leader_pos = Pcurrent[rand_leader_idx]
                    D = np.linalg.norm(C * rand_leader_pos - ranked_pos[i])
                    Pnext[i] = rand_leader_pos - A * D
                else:
                    D = np.linalg.norm(C * leader_pos - ranked_pos[i])
                    Pnext[i] = leader_pos - A * D
            else:
                l = np.random.uniform(-1.0, 1.0)
                D = np.linalg.norm(leader_pos - ranked_pos[i])
                Pnext[i] = (
                    D * np.exp(self.params.b * l) * np.cos(2 * l * math.pi) + leader_pos
                )
        return self.bound_population(Pnext)
