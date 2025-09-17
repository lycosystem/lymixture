"""Implements the `EM algorithm`_ for the mixture model.

Using the class :py:class:`.models.LymphMixture` and its methods, this module provides
functions to compute the expectation and maximization steps of the `EM algorithm`_.

.. _EM algorithm: https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
"""

import logging
from collections.abc import Callable, Sequence
from multiprocessing import Pool

import emcee
import numpy as np
import pandas as pd
from scipy import optimize as opt

from lymixture import models, utils

logger = logging.getLogger(__name__)

RNG = np.random.default_rng(seed=42)
"""Random number generator for reproducibility."""


def _get_params(model: models.LymphMixture) -> np.ndarray:
    """Return the params of ``model``.

    .. seealso::
        This function is very similar to the :py:meth:`.models.LymphMixture.get_params`
        method. Except that it does not accept a dictionary of parameters, but only a 1D
        array.
    """
    params = []
    if model.universal_p:
        for comp in model.components:
            params += list(comp.get_spread_params(as_dict=False))

        params += list(model.get_distribution_params(as_dict=False))
    else:
        for comp in model.components:
            params += list(comp.get_params(as_dict=False))
    return params


def _set_params(model: models.LymphMixture, params: np.ndarray) -> None:
    """Set the params of ``model`` from ``params``.

    .. seealso::
        This function is very similar to the :py:meth:`.models.LymphMixture.set_params`
        method. Except that it does not accept a dictionary of parameters, but only a 1D
        array.
    """
    if model.universal_p:
        for comp in model.components:
            params = comp.set_spread_params(*params)
        params = np.array(model.set_distribution_params(*params))
    else:
        for comp in model.components:
            params = comp.set_params(*params)
        params = np.array(params)


def expectation(
    model: models.LymphMixture,
    params: dict[str, float],
    *,
    log: bool = False,
) -> np.ndarray:
    """Compute expected value of latent ``model`` variables given the ``params``.

    This marks the E-step of the famous `EM algorithm`_. The returned expected values
    are also often called responsibilities.

    If ``log`` is set to ``True``, the function returns the logarithm of the
    responsibilities.

    .. _EM algorithm: https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
    """
    model.set_params(**params)
    llhs = model.patient_mixture_likelihoods(log=log, marginalize=False)
    if log:
        return utils.log_normalize(llhs.T, axis=0).T

    return utils.normalize(llhs.T, axis=0).T


def init_callback() -> Callable:
    """Return a function that logs the optimization progress."""
    iteration = 0

    def log_optimization(xk) -> None:  # noqa: ANN001
        nonlocal iteration
        logger.debug(f"Iteration {iteration} with params: {xk}")
        iteration += 1

    return log_optimization


def _neg_complete_component_llh(
    params: np.ndarray,
    model: models.LymphMixture,
    component: int,
) -> float:
    """Return the negative complete log likelihood of ``component`` in ``model``.

    This function is used in the M-step of the EM algorithm.
    """
    try:
        model.components[component].set_params(*params)
    except ValueError:
        return np.inf

    result = -model.complete_data_likelihood(component=component)
    logger.debug(f"Component {component} with params {params} has llh {result}")
    return result

def _neg_complete_component_llh_shared(
    params: dict[str, float],
    model: models.LymphMixture,
) -> float:
    """Return the negative complete log likelihood of ``component`` in ``model``.

    This function is used in the M-step of the EM algorithm.
    """
    try:
        params_dict = model.get_params(model_params_only=True)
        params_dict_new = dict(zip(params_dict.keys(), params))
        model.set_params(**params_dict_new)
    except ValueError:
        return np.inf

    result = -model.complete_data_likelihood()
    logger.debug(f"Mixture model with params {params} has llh {result}")
    return result


def maximization(
    model: models.LymphMixture,
    log_resps: np.ndarray,
) -> dict[str, float]:
    """Maximize ``model`` params given expectation of ``latent`` variables.

    This is the corresponding M-step to the :py:func:`.expectation` of the
    `EM algorithm`_. It first maximizes the mixture coefficients analytically and then
    optimizes the model parameters of all components sequentially.

    .. _EM algorithm: https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
    """
    log_maxed_mix_coefs = model.infer_mixture_coefs(new_resps=log_resps, log=True)
    log_maxed_mix_coefs = utils.log_normalize(log_maxed_mix_coefs, axis=0)
    model.set_mixture_coefs(np.exp(log_maxed_mix_coefs))
    if model.shared_transmission or model.universal_p:
        current_params = list(model.get_params(as_dict=False, model_params_only=True))
        lb = np.zeros(shape=len(current_params))
        ub = np.ones(shape=len(current_params))
        result = opt.minimize(
            fun=_neg_complete_component_llh_shared,
            args=(model),
            x0=current_params,
            bounds=opt.Bounds(lb=lb, ub=ub),
            method="Powell",
            callback=init_callback(),
        )

        if result.success:
            params_dict = model.get_params(model_params_only=True)
            params_dict_new = dict(zip(params_dict.keys(), result.x))
            model.set_params(**params_dict_new)
        else:
            msg = f"Optimization failed: {result}"
            raise RuntimeError(msg)

    else:
        for i, component in enumerate(model.components):
            current_params = list(component.get_params(as_dict=False))
            lb = np.zeros(shape=len(current_params))
            ub = np.ones(shape=len(current_params))

            result = opt.minimize(
                fun=_neg_complete_component_llh,
                args=(model, i),
                x0=current_params,
                bounds=opt.Bounds(lb=lb, ub=ub),
                method="Powell",
                callback=init_callback(),
            )

            if result.success:
                component.set_params(*result.x)
            else:
                msg = f"Optimization failed: {result}"
                raise RuntimeError(msg)

    return model.get_params(as_dict=True)


def log_prob_fn_fixed_mixture(
    theta: Sequence[float],
    model: models.LymphMixture,
) -> float:
    """Compute the model's log-prob, given its params, excluding mixture coefficients.

    This function calculates the log-probability of a mixture ``model`` based on the
    provided parameters (``theta``), assuming that mixture coefficients remain fixed.
    It ensures that the parameter values are within the valid range [0, 1], and returns
    negative infinity (``-inf``) if any parameter is out of bounds.

    Returns:
        float: The log-probability of the model if parameters are valid, or ``-inf`` if
        parameters are out of bounds.

    .. note::

        - This function does not modify or include mixture coefficients in ``theta``;
          these are assumed to remain unchanged.
        - The `_set_params` function is used to update the model parameters before
          computing the likelihood.

    """
    lower_bounds = np.zeros(len(theta))
    upper_bounds = np.ones(len(theta))
    # Check if the parameters are within bounds
    if np.any(theta < lower_bounds) or np.any(theta > upper_bounds):
        return -np.inf  # Return -infinity if out of bounds
    _set_params(model, theta)
    return model.likelihood(log=True, use_complete=False)


def log_prob_fn(theta: Sequence[float], model: models.LymphMixture) -> float:
    """Compute the log-probability of the model given its parameters.

    This function returns the log-probability of the provided mixture ``model`` based
    on the given parameter values (``theta``). It ensures that parameters stay within
    predefined bounds (0 to 1). If any parameter is out of bounds, the function
    returns negative infinity (``-inf``).

    .. note::

        The `theta` array includes mixture parameters, which are not sampled from a
        simplex. This behavior could be extended to enforce simplex constraints if
        required.

    """
    lower_bounds = np.zeros(len(theta))
    upper_bounds = np.ones(len(theta))

    # Check if the parameters are within bounds
    if np.any(theta < lower_bounds) or np.any(theta > upper_bounds):
        return -np.inf  # Return -infinity if out of bounds
    model.set_params(*theta)
    return model.likelihood(log=True, use_complete=True)


def sample_fixed_mixture(
    model: models.LymphMixture,
    steps: int = 100,
    latent: pd.DataFrame | None = None,
    filename: str = "chain_fixed_mix.hdf5",
    *,
    continue_sampling: bool = False,
) -> tuple[emcee.backends.HDFBackend, np.ndarray]:
    """Sample the parameters of a mixture model, excluding mixture coefficients.

    This function performs MCMC sampling for the parameters of a mixture ``model`` while
    keeping the mixture coefficients fixed. It allows the specification of ``latent``
    parameters and offers options to either start a new sampling session or
    ``continue_sampling`` from an existing HDF5 backend file (named ``filename``).

    .. note::

        - The model's responsibilities (``resps``) and mixture coefficients are updated
          based on the provided or computed latent parameters.
        - Mixture coefficients are fixed during the sampling process.
        - The function initializes an :py:class:`emcee.EnsembleSampler` with a fixed
          mixture coefficient log-probability function (``log_prob_fn_fixed_mixture``)
          and uses multiprocessing to parallelize sampling.
    """
    if latent is None:
        latent = model.get_resps()
    model.set_resps(latent)
    maximized_mixture_coefs = model.infer_mixture_coefs(new_resps=latent)
    model.set_mixture_coefs(maximized_mixture_coefs)
    current_params = _get_params(model)

    ndim = len(current_params)
    nwalkers = 5 * ndim
    perturbation = 1e-6 * RNG.randn(nwalkers, ndim)
    backend = emcee.backends.HDFBackend(filename)
    if continue_sampling is False:
        starting_points = np.ones((nwalkers, ndim)) * current_params + perturbation
        backend.reset(nwalkers, ndim)
    else:
        starting_points = None
    # Pass model as an additional argument to log_prob_fn
    with Pool() as pool:
        logger.info(f"Number of cores used by the sampler: {pool._processes}")  # noqa: SLF001

        original_sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob_fn_fixed_mixture,
            args=(model,),  # Pass model here
            pool=pool,
            backend=backend,
        )
        original_sampler.run_mcmc(
            initial_state=starting_points,
            nsteps=steps,
            progress=True,
        )

    return backend, original_sampler.get_chain(discard=0, thin=1, flat=True)


def sample_model_params(
    model: models.LymphMixture,
    steps: int = 100,
    latent: pd.DataFrame | None = None,
    filename: str = "chain_fixed_latent.hdf5",
    *,
    continue_sampling: bool = False,
) -> tuple[emcee.backends.HDFBackend, np.ndarray]:
    """Sample the parameters of a mixture model given expectations of latent variables.

    This function performs Markov Chain Monte Carlo (MCMC) sampling of the parameters
    of a provided mixture ``model``. It allows setting ``latent`` parameters and
    provides options to either start sampling from scratch or ``continue_sampling``
    from a previous state stored in an HDF5 file named ``filename``.

    .. note::

        - The model's responsibilities (``resps``) and mixture coefficients are
          updated based on the provided or computed latent parameters.
        - The function initializes an `emcee.EnsembleSampler` for MCMC sampling and
          uses a multiprocessing pool to parallelize the computations.
    """
    latent = latent or model.get_resps()

    model.set_resps(latent)
    model.set_mixture_coefs(model.infer_mixture_coefs())
    current_params = list(model.get_params(as_dict=False))

    ndim = len(current_params)
    nwalkers = 5 * ndim
    perturbation = 1e-6 * abs(RNG.randn(nwalkers, ndim))
    backend = emcee.backends.HDFBackend(filename)

    if continue_sampling is False:
        starting_points = np.ones((nwalkers, ndim)) * current_params + perturbation
        starting_points[starting_points > 1] = 1 - perturbation[starting_points > 1]
        backend.reset(nwalkers, ndim)
    else:
        starting_points = None

    with Pool() as pool:
        original_sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob_fn,
            args=(model,),
            pool=pool,
            backend=backend,
        )
        original_sampler.run_mcmc(
            initial_state=starting_points,
            nsteps=steps,
            progress=True,
        )

    return backend, original_sampler.get_chain(discard=0, thin=1, flat=True)


def complete_latent_likelihood(
    theta: Sequence[float],
    model: models.LymphMixture,
) -> float:
    """Compute the complete data log-llh for mixture ``model``, given latent variables.

    This function evaluates the log-likelihood of the mixture ``model`` using a
    provided set of latent variable assignments (``theta``). The assignments are set
    as the responsibilities (``resps``) of the model before computing the likelihood.
    """
    resps = model.get_resps()
    sampled_df = pd.DataFrame(theta, index=resps.index, columns=resps.columns)
    model.set_resps(sampled_df)
    return model.likelihood(log=True, use_complete=True)


def mh_latent_sampler_per_patient_2_component(
    model: models.LymphMixture,
    temp: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Perform Metropolis-Hastings for latent variables per-patient for 2 components.

    This function implements a basic Metropolis-Hastings (MH) sampler to update the
    latent variables (responsibilities) of a mixture ``model`` for individual patients.
    It swaps the latent variable assignments for two components, evaluates the
    log-acceptance ratio, and accepts or rejects the proposed changes based on the
    Metropolis criterion.

    It returns the latent variable responsibilities before the sampling step and the
    log-probability of the model before the sampling step.

    .. note::

        - The sampler works by proposing a swap of responsibilities between two
          components for each patient and calculating the acceptance ratio using
          the patient-specific mixture likelihoods.
        - Accepted swaps are updated in the latent variable matrix under
          the header ``accepted_position``.
        - The current and new log-probabilities are computed using
          the provided ``log_prob_fn``.
        - This function is designed for a full AIP algorithm but is not used
          due to long computation times.
    """
    temp = temp or 0.5
    current_position = model.get_resps()
    new_position = current_position.copy()
    accepted_position = current_position.copy()
    current_log_prob = complete_latent_likelihood(current_position, model)
    new_position.iloc[:, [0, 1]] = current_position.iloc[:, [1, 0]].to_numpy()

    current_assignments = np.argmax(np.array(current_position), axis=1)
    new_assignments = np.argmax(np.array(new_position), axis=1)

    log_acceptance_ratio = (
        model.patient_mixture_likelihoods(log=True)[
            np.arange(len(new_assignments)),
            new_assignments,
        ]
        - model.patient_mixture_likelihoods(log=True)[
            np.arange(len(current_assignments)),
            current_assignments,
        ]
    ) / temp
    accept_ratio = np.exp(log_acceptance_ratio)
    accept_thresholds = RNG.rand(len(accept_ratio))
    accepted_indices = np.where(accept_thresholds < accept_ratio)[0]
    accepted_position.iloc[accepted_indices, [0, 1]] = current_position.iloc[
        accepted_indices,
        [1, 0],
    ].to_numpy()
    model.set_resps(accepted_position)
    logger.info(f"{len(accepted_indices)} swaps accepted")
    return current_position, current_log_prob


def aip_sampling_algorithm(
    model: models.LymphMixture,
    ip_rounds: int = 4000,
    n_steps_params: int = 1,
    temperature_schedule: Callable[[int], float] | None = None,
    params_filename: str = "../../params_samples.hdf5",
) -> dict[str, list]:
    """Perform Alternating Iterative Posterior (AIP) sampling for a mixture model.

    This function alternates between sampling latent variables and ``model`` parameters
    to approximate the posterior distribution of a mixture model. The AIP algorithm
    integrates Metropolis-Hastings (MH) sampling for latent variables and a parameter
    sampler initialized with ``emcee``. This is computationally intensive and may take
    a long time to converge and is therefore only used for toy problems.

    Returns:
        A dictionary containing:
        - "params_samples" (list): Samples of model parameters.
        - "latent_samples" (list): Samples of latent variables.
        - "complete_likelihoods" (list): Complete data log-llhs across iterations.
        - "incomplete_likelihoods" (list): Incomplete data log-llhs across iterations.
        - "number_of_swaps" (list): Number of swaps in latent variables btw. iterations.

    """
    # Initialization
    n_dim_params = len(model.get_params())

    # Lists to store results
    params_samples = []
    latent_samples = []
    complete_likelihoods = []
    incomplete_likelihoods = []
    number_of_swaps = []

    # Initialize latent variables
    starting_latent = model.get_resps()
    starting_latent.iloc[:, 0] = RNG.choice([0, 1], len(starting_latent))
    starting_latent.iloc[:, 1] = 1 - starting_latent.iloc[:, 0]
    model.set_resps(starting_latent)

    # Initialize parameter sampler
    backend_params, params_samples = sample_model_params(
        model,
        steps=1,
        filename=params_filename,
        continue_sampling=False,
    )

    # Initial samples
    latent_samples.append(model.get_resps())
    params_samples.append(model.get_params(as_dict=False))

    for ip_round in range(ip_rounds):
        # Determine temperature
        if temperature_schedule is None:
            temperature = 1 - ip_round / ip_rounds + 0.05
        else:
            temperature = temperature_schedule(ip_round)

        # Latent sampling
        new_latent, current_prob = mh_latent_sampler_per_patient_2_component(
            model,
            complete_latent_likelihood,
            temperature,
        )
        latent_samples.append(new_latent)

        # Parameter sampling
        backend_params, params_samples = sample_model_params(
            model,
            steps=n_steps_params,
            filename=params_filename,
            continue_sampling=True,
        )
        new_params_samples = backend_params.get_chain(discard=0, thin=1, flat=False)

        # Extract the last parameter sample and update
        samples_flat = new_params_samples.reshape(-1, n_dim_params)  # Flatten correctly
        params_samples.append(samples_flat[-1])
        model.set_params(*samples_flat[-1])

        # Compute likelihoods for diagnostics
        complete_likelihoods.append(model.likelihood(use_complete=True))
        incomplete_likelihoods.append(model.likelihood(use_complete=False))

        if ip_round != 0:
            number_of_swaps.append(
                abs(latent_samples[-1] - latent_samples[-2]).sum().sum() / 2,
            )

        logger.debug(
            f"Complete likelihood: {complete_likelihoods[-1]}, "
            f"Incomplete likelihood: {incomplete_likelihoods[-1]}",
        )

    return {
        "params_samples": params_samples,
        "latent_samples": latent_samples,
        "complete_likelihoods": complete_likelihoods,
        "incomplete_likelihoods": incomplete_likelihoods,
        "number_of_swaps": number_of_swaps,
    }