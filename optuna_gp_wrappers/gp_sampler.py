import skopt

from optuna import distributions
from optuna.samplers import BaseSampler
from optuna.trial import FrozenTrial
from optuna.structs import TrialState


class GPSampler(BaseSampler):
    def __init__(self):
        # type: () -> None

        self.optimizer = None
        self.search_space = {}
        self.param_names = []
        self.told_trials = set()

    def sample_relative(self, trial, search_space):
        """
        # type:(FrozenTrial, Dict[str, distributions.BaseDistribution]) -> Dict[str, float]
        """

        if len(search_space) == 0:
            return {}

        # [1] Initialize optimizer.
        if len(self.search_space) == 0:
            self.search_space = search_space

            # Convert optuna's search space to skopt's one.
            dimensions = []
            for name, distribution in search_space.items():
                if isinstance(distribution, distributions.UniformDistribution):
                    dimensions.append((distribution.low, distribution.high))
                    self.param_names.append(name)
                elif isinstance(distribution, distributions.IntUniformDistribution):
                    dimensions.append((distribution.low, distribution.high))
                    self.param_names.append(name)
                elif isinstance(distribution, distributions.CategoricalDistribution):
                    dimensions.append(distribution.choices)
                    self.param_names.append(name)
                else:
                    # TODO: Support other distributions.
                    pass

            self.optimizer = skopt.Optimizer(dimensions, "GP")

        # TODO: Support search space change (e.g., re-create optimizer instance).
        assert self.search_space == search_space

        # [2] Tell previous results.
        for trial in self.study.trials:
            if trial.number in self.told_trials or trial.state != TrialState.COMPLETE:
                continue
            self.told_trials.add(trial.number)

            params = []
            for name in self.param_names:
                assert name in trial.params_in_internal_repr  # TODO: Add error handling.

                internal = trial.params_in_internal_repr[name]
                value = self.search_space[name].to_external_repr(internal)
                params.append(value)

            self.optimizer.tell(params, trial.value)  # TODO: Support 'maximize' direction.

        # [3] Ask next parameters.
        params = self.optimizer.ask()
        return dict((n, self.search_space[n].to_internal_repr(p)) for n, p in zip(self.param_names, params))
