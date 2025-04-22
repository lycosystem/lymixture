"""Provides the :py:class:`LymphMixture` class for wrapping multiple lymph models.

Each component and subgroup of the mixture model is a
:py:class:`~lymph.models.Unilateral` instance. Its properties, parametrization, and
data are orchestrated by the :py:class:`LymphMixture` class. It provides the methods
and computations necessary to use the expectation-maximization algorithm to fit the
model to data.
"""

import logging
import warnings
from collections.abc import Iterable
from typing import Any, TypeVar

import lymph
import numpy as np
import pandas as pd
from lymph import diagnosis_times, modalities, types
from lymph.utils import flatten, popfirst, unflatten_and_split

from lymixture.utils import (
    RESP_COLS,
    T_STAGE_COL,
    join_with_resps,
    normalize,
    one_slice,
)

pd.options.mode.copy_on_write = True
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)

# Ignore a warning that appears due to self.t_stage when each component has a different
# t_stage (If we set components with different t_stages, i.e. not all of them are early
# and late, but some have others, then this wont work anymore and we need to reconsider
# the code structure)
warnings.filterwarnings(
    action="ignore",
    message="Not all distributions are equal. Returning the first one.",
)


ModelType = TypeVar("ModelType", bound=types.Model)


class LymphMixture(
    diagnosis_times.Composite,  # NOTE: The order of inheritance must be the same as the
    modalities.Composite,  #       order in which the respective __init__ methods
    types.Model,  #       are called.
):
    """Class that handles the individual components of the mixture model."""

    def __init__(
        self,
        model_cls: type[ModelType] = lymph.models.Unilateral,
        model_kwargs: dict[str, Any] | None = None,
        num_components: int = 2,
        *,
        universal_p: bool = False,
    ) -> None:
        """Initialize the mixture model.

        The mixture will be based on the given ``model_cls`` (which is instantiated with
        the ``model_kwargs``), and will have ``num_components``. ``universal_p``
        indicates whether the model shares the time prior distribution over all
        components.
        """
        if model_kwargs is None:
            model_kwargs = {
                "graph_dict": {
                    ("tumor", "T"): ["II", "III"],
                    ("lnl", "II"): ["III"],
                    ("lnl", "III"): [],
                },
            }

        if not issubclass(model_cls, lymph.models.Unilateral):
            msg = "Mixture model only implemented for `Unilateral` model."
            raise NotImplementedError(msg)

        self._model_cls = model_cls
        self._model_kwargs = model_kwargs
        self._mixture_coefs = None
        self.universal_p = universal_p

        self.subgroups: dict[str, ModelType] = {}
        self.components: list[ModelType] = self._init_components(num_components)

        diagnosis_times.Composite.__init__(
            self,
            distribution_children=dict(enumerate(self.components)),
            is_distribution_leaf=False,
        )
        logger.info(
            f"Created LymphMixtureModel based on {model_cls} model with "
            f"{num_components} components.",
        )

    def _init_components(self, num_components: int) -> list[Any]:
        """Initialize the component parameters and assignments."""
        if num_components < 2:
            msg = f"A mixture of {num_components} does not make sense."
            raise ValueError(msg)

        return [self._model_cls(**self._model_kwargs) for _ in range(num_components)]

    @property
    def is_trinary(self) -> bool:
        """Check if the model is trinary."""
        if all(sub.is_trinary for sub in self.subgroups.values()) != all(
            comp.is_trinary for comp in self.components
        ):
            msg = "Subgroups & components not all trinary/not all binary."
            raise ValueError(msg)

        return self.components[0].is_trinary

    def _init_mixture_coefs(self) -> pd.DataFrame:
        """Initialize the mixture coefficients for the model."""
        nan_array = np.empty((len(self.components), len(self.subgroups)))
        nan_array[:] = np.nan
        return pd.DataFrame(
            nan_array,
            index=range(len(self.components)),
            columns=self.subgroups.keys(),
        )

    def get_mixture_coefs(
        self,
        component: int | None = None,
        subgroup: str | None = None,
        *,
        norm: bool = True,
    ) -> float | pd.Series | pd.DataFrame:
        """Get mixture coefficients for the given ``subgroup`` and ``component``.

        The mixture coefficients are sliced by the given ``subgroup`` and ``component``
        which means that if no subgroup and/or component is given, multiple mixture
        coefficients are returned.

        If ``norm`` is set to ``True``, the mixture coefficients are normalized along
        the component axis before being returned.
        """
        if getattr(self, "_mixture_coefs", None) is None:
            self._mixture_coefs = self._init_mixture_coefs()

        if norm:
            self.normalize_mixture_coefs()

        component = slice(None) if component is None else component
        subgroup = slice(None) if subgroup is None else subgroup
        return self._mixture_coefs.loc[component, subgroup]

    def set_mixture_coefs(
        self,
        new_mixture_coefs: float | np.ndarray,
        component: int | None = None,
        subgroup: str | None = None,
    ) -> None:
        """Assign new mixture coefficients to the model.

        As in :py:meth:`~get_mixture_coefs`, ``subgroup`` and ``component`` can be used
        to slice the mixture coefficients and therefore assign entirely new coefs to
        the entire model, to one subgroup, to one component, or to one component of one
        subgroup.

        .. note::
            After setting, these coefficients are not normalized.
        """
        if getattr(self, "_mixture_coefs", None) is None:
            self._mixture_coefs = self._init_mixture_coefs()

        component = slice(None) if component is None else component
        subgroup = slice(None) if subgroup is None else subgroup
        self._mixture_coefs.loc[component, subgroup] = new_mixture_coefs

    def normalize_mixture_coefs(self) -> None:
        """Normalize the mixture coefficients to sum to one."""
        if getattr(self, "_mixture_coefs", None) is not None:
            self._mixture_coefs = normalize(self._mixture_coefs, axis=0)

    def repeat_mixture_coefs(
        self,
        t_stage: str | None = None,
        subgroup: str | None = None,
        *,
        log: bool = False,
    ) -> np.ndarray:
        """Repeat mixture coefficients.

        The result will match the number of patients with tumors of ``t_stage`` that
        are in the specified ``subgroup`` (or all if it is set to ``None``). The
        mixture coefficients are returned in log-space if ``log`` is set to ``True``

        This method enables easy multiplication of the mixture coefficients with the
        likelihoods of the patients under the components as in the method
        :py:meth:`.patient_mixture_likelihoods`.
        """
        result = np.empty(shape=(0, len(self.components)))

        if subgroup is not None:
            subgroups = {subgroup: self.subgroups[subgroup]}
        else:
            subgroups = self.subgroups

        for label, subgroup in subgroups.items():  # noqa: PLR1704
            is_t_stage = subgroup.patient_data[T_STAGE_COL] == t_stage
            num_patients = is_t_stage.sum() if t_stage is not None else len(is_t_stage)
            result = np.vstack(
                [
                    result,
                    np.tile(self.get_mixture_coefs(subgroup=label), (num_patients, 1)),
                ],
            )
        with np.errstate(divide="ignore"):
            return np.log(result) if log else result

    def infer_mixture_coefs(
        self,
        new_resps: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Infer the optimal mixture parameters given the mixture's responsibilities.

        This method can be seen as part of the M-step of the EM algorithm, since the
        mixture's likelihood is maximized with respect to the mixture coefficients when
        they are simply the average of the corresponding responsibilities.

        If set to ``None``, the responsibilities already stored in the model are used.
        Otherwise, they are first set via the :py:meth:`.set_resps` method.

        The returned ``DataFrame`` has the shape ``(num_components, num_subgroups)`` and
        can be used to set the mixture coefficients via the
        :py:meth:`.set_mixture_coefs` method.
        """
        mixture_coefs = np.zeros(self.get_mixture_coefs().shape).T

        if new_resps is not None:
            self.set_resps(new_resps)

        for i, subgroup in enumerate(self.subgroups.keys()):
            num_in_subgroup = len(self.subgroups[subgroup].patient_data)
            mixture_coefs[i] = (
                self.get_resps(subgroup=subgroup).sum(axis=0) / num_in_subgroup
            )

        return pd.DataFrame(mixture_coefs.T, columns=self.subgroups.keys())

    def get_params(
        self,
        *,
        as_dict: bool = True,
        as_flat: bool = True,
    ) -> Iterable[float] | dict[str, float]:
        """Get the parameters of the mixture model.

        This includes both the parameters of the individual components and the mixture
        coefficients. If a dictionary is returned (i.e. if ``as_dict`` is set to
        ``True``), the components' parameters are nested under keys that simply
        enumerate them. While the mixture coefficients are returned under keys of the
        form ``<subgroup>from<component>_coef``.

        The parameters are returned as a dictionary if ``as_dict`` is True, and as
        an iterable of floats otherwise. The argument ``as_flat`` determines whether
        the returned dict is flat or nested.

        .. seealso::
            In the :py:mod:`lymph` package, the model parameters are also set and get
            using the :py:meth:`~lymph.types.Model.get_params` and the
            :py:meth:`~lymph.types.Model.set_params` methods. We tried to keep the
            interface as similar as possible.

        >>> graph_dict = {
        ...     ("tumor", "T"): ["II", "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... }
        >>> mixture = LymphMixture(
        ...     model_kwargs={"graph_dict": graph_dict},
        ...     num_components=2,
        ... )
        >>> mixture.get_params(as_dict=True)     # doctest: +NORMALIZE_WHITESPACE
        {'0_TtoII_spread': 0.0,
         '0_TtoIII_spread': 0.0,
         '0_IItoIII_spread': 0.0,
         '1_TtoII_spread': 0.0,
         '1_TtoIII_spread': 0.0,
         '1_IItoIII_spread': 0.0}
        """
        params = {}
        for c, component in enumerate(self.components):
            if self.universal_p:
                params[str(c)] = component.get_spread_params(as_flat=as_flat)
            else:
                params[str(c)] = component.get_params(as_flat=as_flat)

            for label in self.subgroups:
                params[str(c)].update(
                    {f"{label}_coef": self.get_mixture_coefs(c, label)},
                )

        if self.universal_p:
            params.update(self.get_distribution_params(as_flat=as_flat))

        if as_flat or not as_dict:
            params = flatten(params)

        return params if as_dict else params.values()

    def set_params(self, *args: float, **kwargs: float) -> tuple[float]:
        """Assign new params to the component models.

        This includes both the spread parameters for the component's models (if
        provided as positional arguments, they are used up first), as well as the
        mixture coefficients for the subgroups.

        .. seealso::
            In the :py:mod:`lymph` package, the model parameters are also set and get
            using the :py:meth:`~lymph.types.Model.get_params` and the
            :py:meth:`~lymph.types.Model.set_params` methods. We tried to keep the
            interface as similar as possible.

        .. important::
            After setting all parameters, the mixture coefficients are normalized and
            may thus not be the same as the ones provided in the arguments.

        >>> graph_dict = {
        ...     ("tumor", "T"): ["II", "III"],
        ...     ("lnl", "II"): ["III"],
        ...     ("lnl", "III"): [],
        ... }
        >>> mixture = LymphMixture(
        ...     model_kwargs={"graph_dict": graph_dict},
        ...     num_components=2,
        ... )
        >>> mixture.set_params(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        (0.7,)
        >>> mixture.get_params(as_dict=True)   # doctest: +NORMALIZE_WHITESPACE
        {'0_TtoII_spread': 0.1,
         '0_TtoIII_spread': 0.2,
         '0_IItoIII_spread': 0.3,
         '1_TtoII_spread': 0.4,
         '1_TtoIII_spread': 0.5,
         '1_IItoIII_spread': 0.6}

        """
        kwargs, global_kwargs = unflatten_and_split(
            kwargs,
            expected_keys=[str(c) for c, _ in enumerate(self.components)],
        )

        for c, component in enumerate(self.components):
            component_kwargs = global_kwargs.copy()
            component_kwargs.update(kwargs.get(str(c), {}))
            args = component.set_spread_params(*args, **component_kwargs)

            if not self.universal_p:
                args = component.set_distribution_params(*args, **component_kwargs)

            for label in self.subgroups:
                first, args = popfirst(args)
                value = component_kwargs.get(f"{label}_coef", first)
                if value is not None:
                    self.set_mixture_coefs(value, component=c, subgroup=label)

        if self.universal_p:
            args = self.set_distribution_params(*args, **global_kwargs)

        self.normalize_mixture_coefs()
        return args

    def get_resps(
        self,
        patient: int | None = None,
        subgroup: str | None = None,
        component: int | None = None,
        t_stage: str | None = None,
        *,
        norm: bool = True,
    ) -> float | pd.Series | pd.DataFrame:
        """Get the responsibility of a ``patient`` for a ``component``.

        The ``patient`` index enumerates all patients in the mixture model unless
        ``subgroup`` is given, in which case the index runs over the patients in the
        given subgroup.

        Omitting ``component`` or ``patient`` (or both) will return corresponding slices
        of the responsibility table.

        The ``filter_by`` argument can be used to filter the responsibility table by
        any ``filter_value`` in the patient data. Most commonly, this is used to filter
        the responsibilities by T-stage.
        """
        if subgroup is not None:
            resp_table = self.subgroups[subgroup].patient_data[RESP_COLS]
        else:
            resp_table = self.patient_data[RESP_COLS]

        if norm:
            # double transpose, because pandas has weird broadcasting behavior
            resp_table = normalize(resp_table.T, axis=0).T

        if t_stage is not None:
            idx = resp_table["t_stage"] == t_stage
            if patient is not None and not idx[patient]:
                msg = f"Patient {patient} does not have T-stage {t_stage}."
                raise ValueError(msg)
            if patient is not None:
                idx = patient
        else:
            idx = slice(None) if patient is None else patient

        component = slice(None) if component is None else component
        return resp_table.loc[idx, component]

    def set_resps(
        self,
        new_resps: float | np.ndarray,
        patient: int | None = None,
        subgroup: str | None = None,
        component: int | None = None,
    ) -> None:
        """Assign ``new_resps`` (responsibilities) to the model.

        They should have the shape ``(num_patients, num_components)``, where
        ``num_patients`` is either the total number of patients in the model or only
        the number of patients in the ``subgroup`` (if that argument is not ``None``)
        and summing them along the last axis should yield a vector of ones.

        Note that these responsibilities essentially become the latent variables
        of the model or the expectation values of the latent variables (depending on
        whether or not they are "hardened", see :py:meth:`.harden_responsibilities`).

        .. note::
            Also, like in the :py:meth:`.set_mixtures_coefs` method, the
            responsibilities are not normalized after setting them.
        """
        if isinstance(new_resps, pd.DataFrame):
            new_resps = new_resps.to_numpy()

        pat_slice = slice(None) if patient is None else patient
        comp_slice = (*RESP_COLS, slice(None) if component is None else component)

        if subgroup is not None:
            sub_data = self.subgroups[subgroup].patient_data
            sub_data.loc[pat_slice, comp_slice] = new_resps
            return

        patient_idx = 0
        for subgroup in self.subgroups.values():  # noqa: PLR1704
            sub_data = subgroup.patient_data
            patient_idx += len(sub_data)

            if patient is not None:
                if patient_idx > patient:
                    sub_data.loc[pat_slice, comp_slice] = new_resps
                    break
            else:
                sub_resp = new_resps[: len(sub_data)]
                sub_data.loc[pat_slice, comp_slice] = sub_resp
                new_resps = new_resps[len(sub_data) :]

    def load_patient_data(
        self,
        patient_data: pd.DataFrame,
        split_by: tuple[str, str, str],
        **kwargs,
    ) -> None:
        """Split the ``patient_data`` into subgroups and load it into the model.

        This amounts to computing the diagnosis matrices for the individual subgroups.
        The ``split_by`` tuple should contain the three-level header of the LyProX-style
        data. Any additional keyword arguments are passed to the
        :py:meth:`~lymph.models.Unilateral.load_patient_data` method.
        """
        self._mixture_coefs = None
        grouped = patient_data.groupby(split_by)

        for label, data in grouped:
            if label not in self.subgroups:
                self.subgroups[label] = self._model_cls(**self._model_kwargs)
            joined_data = join_with_resps(data, num_components=len(self.components))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=types.DataWarning)
                self.subgroups[label].load_patient_data(joined_data, **kwargs)

        modalities.Composite.__init__(
            self,
            modality_children=self.subgroups,
            is_modality_leaf=False,
        )

    @property
    def patient_data(self) -> pd.DataFrame:
        """Return all patients stored in the individual subgroups."""
        return pd.concat(
            [subgroup.patient_data for subgroup in self.subgroups.values()],
            ignore_index=True,
        )

    def patient_component_likelihoods(
        self,
        t_stage: str | None = None,
        subgroup: str | None = None,
        component: int | None = None,
        *,
        log: bool = True,
    ) -> np.ndarray:
        """Compute the (log-)likelihood of all patients, given the components.

        The returned array has shape ``(num_patients, num_components)`` and contains
        the likelihood of each patient with ``t_stage`` under each component. If ``log``
        is set to ``True``, the likelihoods are returned in log-space.
        """
        t_stages = [t_stage] if t_stage is not None else self.t_stages

        sg_keys = list(self.subgroups.keys())
        sg_idx = slice(None) if subgroup is None else one_slice(sg_keys.index(subgroup))
        subgroups = list(self.subgroups.values())[sg_idx]

        comp_idx = slice(None) if component is None else one_slice(component)
        components = self.components[comp_idx]

        llhs = np.empty(shape=(0, len(components)))

        for subgroup in subgroups:  # noqa: PLR1704
            shape = (len(subgroup.patient_data), len(components))
            sub_llhs = np.empty(shape=shape)

            for t in t_stages:
                # use the index to align the likelihoods with the patients
                t_idx = subgroup.patient_data[T_STAGE_COL] == t
                sub_llhs[t_idx] = np.stack(
                    [  # TODO: Precompute the state-dists for better performance
                        comp.state_dist(t) @ subgroup.diagnosis_matrix(t).T
                        for comp in components
                    ],
                    axis=-1,
                )
            llhs = np.vstack([llhs, sub_llhs])

        if component is not None:
            llhs = llhs[:, 0]

        return np.log(llhs) if log else llhs

    def patient_mixture_likelihoods(
        self,
        t_stage: str | None = None,
        subgroup: str | None = None,
        component: int | None = None,
        *,
        log: bool = True,
        marginalize: bool = False,
    ) -> np.ndarray:
        """Compute the (log-)likelihood of all patients under the mixture model.

        This is essentially the (log-)likelihood of all patients given the individual
        components as computed by :py:meth:`.patient_component_likelihoods`, but
        weighted by the mixture coefficients. This means that the returned array when
        ``marginalize`` is set to ``False`` represents the unnormalized expected
        responsibilities of the patients for the components.

        If ``marginalize`` is set to ``True``, the likelihoods are summed
        over the components, effectively marginalizing the components out of the
        likelihoods and yielding the incomplete data likelihood per patient.
        """
        component_patient_likelihood = self.patient_component_likelihoods(
            t_stage=t_stage,
            subgroup=subgroup,
            component=component,
            log=log,
        )
        full_mixture_coefs = self.repeat_mixture_coefs(
            t_stage=t_stage,
            subgroup=subgroup,
            log=log,
        )

        component = slice(None) if component is None else component
        matching_mixture_coefs = full_mixture_coefs[:, component]

        if len(component_patient_likelihood.shape) != len(matching_mixture_coefs.shape):
            msg = "Mismatch btw. num components and num mixture coefficients."
            raise ValueError(msg)

        if log:
            llh = matching_mixture_coefs + component_patient_likelihood
        else:
            llh = matching_mixture_coefs * component_patient_likelihood

        if marginalize:
            return np.logaddexp.reduce(llh, axis=1) if log else np.sum(llh, axis=1)

        return llh

    def incomplete_data_likelihood(
        self,
        t_stage: str | None = None,
        subgroup: str | None = None,
        component: int | None = None,
        *,
        log: bool = True,
    ) -> float:
        """Compute the incomplete data likelihood of the model."""
        llhs = self.patient_mixture_likelihoods(
            t_stage=t_stage,
            subgroup=subgroup,
            component=component,
            log=log,
            marginalize=True,
        )
        return np.sum(llhs) if log else np.prod(llhs)

    def complete_data_likelihood(
        self,
        t_stage: str | None = None,
        subgroup: str | None = None,
        component: int | None = None,
        *,
        log: bool = True,
    ) -> float:
        """Compute the complete data likelihood of the model."""
        llhs = self.patient_mixture_likelihoods(
            t_stage=t_stage,
            subgroup=subgroup,
            component=component,
            log=log,
        )
        resps = self.get_resps(
            t_stage=t_stage,
            subgroup=subgroup,
            component=component,
        ).to_numpy()
        if log:
            with np.errstate(invalid="ignore"):
                final_llh = resps * llhs
            final_llh[np.isnan(final_llh)] = 0
            final_llh[np.isinf(final_llh)] = 0
            return np.sum(final_llh)
        return np.prod(llhs**resps)

    def likelihood(
        self,
        given_params: Iterable[float] | dict[str, float] | None = None,
        given_resps: np.ndarray | None = None,
        *,
        log: bool = True,
        use_complete: bool = True,
    ) -> float:
        """Compute the (in-)complete data likelihood of the model.

        The likelihood is computed for the ``given_params``. If no parameters are given,
        the currently set parameters of the model are used.

        If responsibilities for each patient and component are given via
        ``given_resps``, they are used to compute the complete data likelihood.
        Otherwise, the incomplete data likelihood is computed, which marginalizes over
        the responsibilities.

        The likelihood is returned in log-space if ``log`` is set to ``True``.
        """
        try:
            # all functions and methods called here should raise a ValueError if the
            # given parameters are invalid...
            if given_params is None:
                pass
            elif isinstance(given_params, dict):
                self.set_params(**given_params)
            else:
                self.set_params(*given_params)
        except ValueError:
            return -np.inf if log else 0.0

        if use_complete:
            if given_resps is not None:
                self.set_resps(given_resps)

            if np.any(self.get_resps().isna()):
                msg = "Responsibilities contain NaNs."
                raise ValueError(msg)

            return self.complete_data_likelihood(log=log)

        return self.incomplete_data_likelihood(log=log)

    def state_dist(
        self,
        t_stage: str = "early",
        subgroup: str | None = None,
    ) -> np.ndarray:
        """Compute the distribution over possible states.

        Do this for a given ``t_stage`` and ``subgroup``. If no subgroup is given, the
        distribution is computed for all subgroups. The result is a matrix with shape
        ``(num_subgroups, num_states)``.
        """
        comp_state_dist_size = len(self.components[0].state_dist(t_stage))
        comp_state_dists = np.zeros((len(self.components), comp_state_dist_size))
        for i, component in enumerate(self.components):
            comp_state_dists[i] = component.state_dist(t_stage)

        if subgroup is not None:
            state_dist = self.get_mixture_coefs(subgroup=subgroup) @ comp_state_dists
        else:
            state_dist = np.zeros((len(self.subgroups), comp_state_dist_size))
            for i, mixer in enumerate(self._mixture_coefs.items()):
                state_dist[i] = mixer[1] @ comp_state_dists

        return state_dist

    def posterior_state_dist(
        self,
        subgroup: str | None = None,
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnosis: types.DiagnosisType | None = None,
        t_stage: str | int = "early",
    ) -> np.ndarray:
        """Compute the posterior distribution over hidden states given a diagnosis.

        The ``given_diagnosis`` is a dictionary of diagnosis for each modality. E.g.,
        this could look like this:

        .. code-block:: python

            given_diagnosis = {
                "MRI": {"II": True, "III": False, "IV": False},
                "PET": {"II": True, "III": True, "IV": None},
            }

        The ``t_stage`` parameter determines the T-stage for which the posterior is
        computed.
        """
        if given_state_dist is None:
            # in contrast to when computing the likelihood, we do want to raise an error
            # here if the parameters are invalid, since we want to know if the user
            # provided invalid parameters.
            if given_params is not None:
                if isinstance(given_params, dict):
                    self.set_params(**given_params)
                else:
                    self.set_params(*given_params)
            given_state_dist = self.state_dist(t_stage, subgroup)

        if given_diagnosis is None:
            return given_state_dist
        diagnosis_encoding = self.subgroups[subgroup].compute_encoding(given_diagnosis)
        # vector containing P(Z=z|X). Essentially a data matrix for one patient
        diagnosis_given_state = (
            diagnosis_encoding @ self.subgroups[subgroup].observation_matrix().T
        )

        # multiply P(Z=z|X) * P(X) element-wise to get vector of joint probs P(Z=z,X)
        joint_diagnosis_and_state = given_state_dist * diagnosis_given_state

        # compute vector of probabilities for all possible involvements given the
        # specified diagnosis P(X|Z=z) = P(Z=z,X) / P(X), where P(X) = sum_z P(Z=z,X)
        return joint_diagnosis_and_state / np.sum(joint_diagnosis_and_state)

    def risk(
        self,
        subgroup: str,
        involvement: types.PatternType,
        given_params: types.ParamsType | None = None,
        given_state_dist: np.ndarray | None = None,
        given_diagnosis: dict[str, types.PatternType] | None = None,
        t_stage: str = "early",
    ) -> float:
        """Compute risk of a certain ``involvement``, using the ``given_diagnosis``.

        If an ``involvement`` pattern of interest is provided, this method computes
        the risk of seeing just that pattern for the set of given parameters and a
        dictionary of diagnosis for each modality.

        If no ``involvement`` is provided, this will simply return the posterior
        distribution over hidden states, given the diagnosis, as computed by the
        :py:meth:`.posterior_state_dist` method. See its documentation for more
        details about the arguments and the return value.
        """
        posterior_state_dist = self.posterior_state_dist(
            subgroup,
            given_params=given_params,
            given_state_dist=given_state_dist,
            given_diagnosis=given_diagnosis,
            t_stage=t_stage,
        )

        # if a specific involvement of interest is provided, marginalize the
        # resulting vector of hidden states to match that involvement of
        # interest
        return self.components[0].marginalize(involvement, posterior_state_dist)
