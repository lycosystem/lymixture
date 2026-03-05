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
from lymph.models.unilateral import RAW_T_COL_NEW
import numpy as np
import pandas as pd
from lymph import diagnosis_times, modalities, types
from lymph.utils import flatten, popfirst, unflatten_and_split, get_item

from lymixture.utils import (
    RESP_COLS,
    join_with_resps,
    normalize,
    one_slice,
)

MAP_T_COL = ("_model", "core", "t_stage")
MAP_EXT_COL = ("_model", "core", "extension")
MAP_SUBGROUP_COL = ("_mixture", "core", "subsite")
RAW_SUBGROUP_OLD = ("tumor", "1", "subsite")
RAW_SUBGROUP_NEW = ("tumor", "core", "subsite")
# RAW_T_COL_OLD = ("tumor", "1", "t_stage")
# RAW_T_COL_NEW = ("tumor", "core", "t_stage")

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


def _set_resps(
    data: pd.DataFrame,
    resps: np.ndarray,
    columns: pd.MultiIndex,
    t_stage: str | None = None,
) -> None:
    """Help setting ``resps`` in the ``data``."""
    if t_stage is not None:
        is_t_stage = data[MAP_T_COL] == t_stage
    else:
        is_t_stage = np.ones(len(data), dtype=bool)

    data.loc[is_t_stage, columns] = resps


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
        shared_transmission: bool = False,
        split_midext: bool = False,
    ) -> None:
        """Initialize the mixture model.

        The mixture will be based on the given ``model_cls`` (which is instantiated with
        the ``model_kwargs``), and will have ``num_components``. ``universal_p``
        indicates whether the model shares the time prior distribution over all
        components.
        """
        model_kwargs = model_kwargs or {
            "graph_dict": {
                ("tumor", "T"): ["II", "III"],
                ("lnl", "II"): ["III"],
                ("lnl", "III"): [],
            },
        }

        if not (
            issubclass(model_cls, lymph.models.Unilateral)
            or issubclass(model_cls, lymph.models.Bilateral)
            or issubclass(model_cls, lymph.models.Midline)
        ):
            msg = "Mixture model only implemented for `Unilateral`, `Bilateral`, and `Midline` models."
            raise NotImplementedError(msg)
        if model_kwargs.get("central"):
            msg = "Central tumors not implemented in mixture model."
            raise NotImplementedError(msg)
        if split_midext and not issubclass(model_cls, lymph.models.Midline):
            msg = "Splitting midline extension only relevant for `Midline` models."
            raise NotImplementedError(msg)
        if split_midext:
            use_evo = model_kwargs.get("use_midext_evo", True)
            if use_evo:
                msg = "Splitting midline extension only implemented for the non evolution version. Requires `use_midext_evo=False`."
                raise ValueError(msg)

        self._model_cls: type[ModelType] = model_cls
        self._model_kwargs: dict = model_kwargs
        self._mixture_coefs: pd.DataFrame | None = None
        self._split_by: tuple[str, str, str] | None = None
        self.universal_p: bool = universal_p
        self.shared_transmission: bool = shared_transmission
        self.split_midext = split_midext

        self.subgroups: dict[str, ModelType] = {}
        self.components: list[ModelType] = self._init_components(num_components)
        self.transmission_param_names = list(self.components[0].get_lnl_spread_params().keys())

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
    
    def midext_prob_builder(self) -> np.ndarray:
        """Build an array of midext probabilities for each patient and component.
        The result will match the number of patients in the model and assign for each patient
        the correct midext/1-midext probability in column 0 if there is an extension and in column 1 if there is no extension.
        if the extension is NaN both columns will have the midext and 1-midextprobability.
        """
        self.all_midext_probs = {}
        for subgroup_key, subgroup in self.subgroups.items():
            self.all_midext_probs[subgroup_key] = subgroup.patient_data[MAP_EXT_COL].sum()/subgroup.patient_data[MAP_EXT_COL].notna().sum()

        prob_array = np.empty(shape=(len(self.patient_data),2))
        for i, patient in self.patient_data.iterrows():
            prob_array[i,0] = self.all_midext_probs[patient[MAP_SUBGROUP_COL]]
            prob_array[i,1] = 1 - prob_array[i,0]
        mult_array = np.zeros(prob_array.shape)
        extension_col = self.patient_data[MAP_EXT_COL]
        is_nan = extension_col.isna()

        mult_array[:,0] = np.where(is_nan, 1, extension_col.fillna(0).astype(int))
        mult_array[:,1] = np.where(is_nan, 1, (~extension_col.astype(bool).fillna(True)).astype(int))
        self.midext_prob_array = mult_array*prob_array

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
            is_t_stage = subgroup.patient_data[MAP_T_COL] == t_stage
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
        *,
        log: bool = False,
    ) -> pd.DataFrame:
        """Infer optimal mixture coefficients based on responsibilities.

        This method updates the mixture coefficients by averaging the corresponding
        responsibilities, which can be provided via ``new_resps`` or taken from the
        model if ``new_resps`` is ``None``.

        The result is a ``DataFrame`` of shape ``(num_components, num_subgroups)``,
        which can be used to update the mixture coefficients via
        ``set_mixture_coefs``.

        If ``log`` is ``True``, both the input ``new_resps`` and the output
        coefficients are in log-space for numerical stability.
        """
        mixture_coefs = np.zeros(self.get_mixture_coefs().shape).T

        if log:
            log_resps = new_resps
            new_resps = np.exp(log_resps)

        for i, subgroup in enumerate(self.subgroups.keys()):
            len_subgroup = len(self.subgroups[subgroup].patient_data)
            idx = self.get_resp_indices(subgroup=subgroup)
            if log:
                log_sum = np.logaddexp.reduce(log_resps[idx], axis=0)
                mixture_coefs[i] = log_sum - np.log(len_subgroup)
            else:
                mixture_coefs[i] = np.sum(new_resps[idx], axis=0) / len_subgroup

        return pd.DataFrame(mixture_coefs.T, columns=self.subgroups.keys())

    def get_params(
        self,
        *,
        as_dict: bool = True,
        as_flat: bool = True,
        model_params_only: bool = False,
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
        transmission_params = {}
        for c, component in enumerate(self.components):
            if self.shared_transmission:
                params[str(c)] = component.get_tumor_spread_params(as_flat=as_flat)
                transmission_params[str(c)] = component.get_lnl_spread_params(as_flat=as_flat)
            else:
                params[str(c)] = component.get_spread_params(as_flat=as_flat)

            if not self.universal_p:
                params[str(c)].update(component.get_distribution_params(as_flat=as_flat))
            ## add potential additional values like midext evolution
            spread_params = component.get_spread_params()
            all_params = component.get_params()
            distribution_params = component.get_distribution_params()
            unique_keys = set(all_params.keys()) - set(spread_params.keys()) - set(distribution_params.keys())
            params[str(c)].update({key: all_params[key] for key in unique_keys})
            
            if not model_params_only:
                for label in self.subgroups:
                    params[str(c)].update(
                        {f"{label}_coef": self.get_mixture_coefs(c, label)},
                    )
            if self.split_midext:
                # remove midext_prob from params as it is set per subgroup
                if 'midext_prob' in params[str(c)]:
                    del params[str(c)]['midext_prob']
        # Check if transmission parameters are the same for all components
        if self.shared_transmission:
            first_transmission = self.components[0].get_lnl_spread_params(as_flat=as_flat)
            for c, component in enumerate(self.components[1:], start=1):
                if component.get_lnl_spread_params(as_flat=as_flat) != first_transmission:
                    warnings.warn(
                        "The transmission parameters are different between components. " \
                        "Returning parameters for the first component.",
                    )
            params.update(first_transmission)

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
        # If shared_transmission is True, remove transmission params from component-specific kwargs
        if self.shared_transmission:
            for c_str in kwargs:
                component_kwargs = kwargs[c_str]
                for param_name in list(component_kwargs.keys()):
                    if param_name in self.transmission_param_names:
                        del component_kwargs[param_name]
        

        for c, component in enumerate(self.components):
            component_kwargs = global_kwargs.copy()
            component_kwargs.update(kwargs.get(str(c), {}))
            args = component.set_spread_params(*args, **component_kwargs)

            if not self.universal_p:
                args = component.set_distribution_params(*args, **component_kwargs)
            ## set potential additional values like midext evolution
            spread_params = component.get_spread_params()
            all_params = component.get_params()
            distribution_params = component.get_distribution_params()
            unique_keys = list(set(all_params.keys()) - set(spread_params.keys()) - set(distribution_params.keys()))
            if unique_keys != [] and component_kwargs != {}:
                unique_dict = {}
                # Only include keys that are actually present in component_kwargs
                unique_dict = {key: component_kwargs[key] for key in unique_keys if key in component_kwargs}
                component.set_params(**unique_dict)

            for label in self.subgroups:
                first, args = popfirst(args)
                value = component_kwargs.get(f"{label}_coef", first)
                if value is not None:
                    self.set_mixture_coefs(value, component=c, subgroup=label)

        if self.universal_p:
            args = self.set_distribution_params(*args, **global_kwargs)

        self.normalize_mixture_coefs()
        return args

    def get_resp_indices(
        self,
        subgroup: str | None = None,
        t_stage: str | None = None,
    ) -> np.ndarray:
        """Get the indices of the responsibilities.

        Returns a boolean array of shape ``(num_patients,)`` that is ``True`` for each
        patient that has the given ``t_stage`` and belongs to the given ``subgroup``.

        Both ``subgroup`` and ``t_stage`` are optional.
        """
        if subgroup is not None:
            is_subgroup = self.patient_data[self._split_by] == subgroup
        else:
            is_subgroup = np.ones(len(self.patient_data), dtype=bool)

        if t_stage is not None:
            has_t_stage = self.patient_data[MAP_T_COL] == t_stage
        else:
            has_t_stage = np.ones(len(self.patient_data), dtype=bool)

        return is_subgroup & has_t_stage

    def get_resps(
        self,
        subgroup: str | None = None,
        component: int | None = None,
        t_stage: str | None = None,
        *,
        norm: bool = True,
    ) -> pd.Series | pd.DataFrame:
        """Get the responsibilities of each patient for a component.

        One can filter the returned table of responsibilities by the patient's subgroup
        and T-stage. If ``norm`` is set to ``True``, the responsibilities are normalized
        to sum to one along the component axis.
        """
        resp_table = self.patient_data[RESP_COLS]

        if norm:
            # double transpose, because pandas has weird broadcasting behavior
            resp_table = normalize(resp_table.T, axis=0).T

        idx = self.get_resp_indices(subgroup=subgroup, t_stage=t_stage)
        component = slice(None) if component is None else component
        return resp_table.loc[idx, component]

    def set_resps(
        self,
        new_resps: float | np.ndarray,
        subgroup: str | None = None,
        component: int | None = None,
        t_stage: str | None = None,
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

        comp_slice = (*RESP_COLS, slice(None) if component is None else component)
        _kwargs = {"t_stage": t_stage, "columns": comp_slice}

        if subgroup is not None:
            sub_data = self.subgroups[subgroup].patient_data
            _set_resps(data=sub_data, resps=new_resps, **_kwargs)
            return

        for subgroup in self.subgroups.values():  # noqa: PLR1704
            sub_data = subgroup.patient_data
            sub_resp = new_resps[: len(sub_data)]
            _set_resps(data=sub_data, resps=sub_resp, **_kwargs)
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
        self._split_by = split_by
        grouped = patient_data.groupby(self._split_by)

        for label, data in grouped:
            if label not in self.subgroups:
                self.subgroups[label] = self._model_cls(**self._model_kwargs)
            joined_data = join_with_resps(data, num_components=len(self.components))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=types.DataWarning)
                self.subgroups[label].load_patient_data(joined_data, **kwargs)
                self.subgroups[label].patient_data[MAP_SUBGROUP_COL] = get_item(
                mapping=self.subgroups[label].patient_data,keys=[RAW_SUBGROUP_OLD, RAW_SUBGROUP_NEW])
        all_patients = pd.concat(
            [subgroup.patient_data for subgroup in self.subgroups.values()],
            ignore_index=True
        )
        # Remove _model and _mixture columns from MultiIndex DataFrame

        all_patients = all_patients.drop(columns=['_model', '_mixture'], errors='ignore')
        # Remove unused levels from MultiIndex
        if hasattr(all_patients.columns, 'remove_unused_levels'):
            all_patients.columns = all_patients.columns.remove_unused_levels()
        all_patients = all_patients.reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=types.DataWarning)
            for component in self.components: # load the data in the correct order
                component.load_patient_data(all_patients, **kwargs)

        #need the component dict to set modalities. We can probably do this more elegantly
        component_dict = {i: comp for i, comp in enumerate(self.components)}
        combined_dict = {**self.subgroups, **component_dict}
        modalities.Composite.__init__(
            self,
            modality_children=combined_dict,
            is_modality_leaf=False,
        )
        t_stage_unique = self.patient_data[MAP_T_COL].unique()
        self.t_stage_indices = {stage: self.patient_data[MAP_T_COL] == stage for stage in t_stage_unique}
        
        # store all midext_probs for each ICD code
        if issubclass(self._model_cls, lymph.models.Midline):
            self.midext_prob_builder()
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
        comp_idx = slice(None) if component is None else one_slice(component)
        components = self.components[comp_idx]
        shape_llhs = (len(self.patient_data), len(components))
        llhs = np.empty(shape_llhs)
        if issubclass(self._model_cls, lymph.models.Midline) and self.split_midext:
            for i, comp in enumerate(components):
                component_llhs = np.empty((len(self.patient_data), 2))
                if t_stage is None:
                    sub_llhs = comp.patient_likelihoods(ext_noext_arrays=True)
                    component_llhs = sub_llhs
                else:
                    for t in t_stages:
                        t_idx = self.t_stage_indices[t]
                        sub_llhs = comp.patient_likelihoods(t_stage = t, ext_noext_arrays=True)
                        component_llhs[t_idx, ] = sub_llhs
                component_llhs = component_llhs*self.midext_prob_array
                component_llhs = np.sum(component_llhs, axis=1)
                llhs[:, i] = component_llhs
        else:
            for i, comp in enumerate(components):
                for t in t_stages:
                    t_idx = self.t_stage_indices[t]
                    sub_llhs = comp.patient_likelihoods(t_stage = t)
                    llhs[t_idx, i] = sub_llhs
        if component is not None:
            llhs = llhs[:, 0]

        return np.log(llhs) if log else llhs

    def patient_mixture_likelihoods(
        self,
        t_stage: str | None = None,
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
            component=component,
            log=log,
        )
        full_mixture_coefs = self.repeat_mixture_coefs(
            t_stage=t_stage,
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
        component: int | None = None,
        *,
        log: bool = True,
    ) -> float:
        """Compute the incomplete data likelihood of the model."""
        llhs = self.patient_mixture_likelihoods(
            t_stage=t_stage,
            component=component,
            log=log,
            marginalize=True,
        )
        return np.sum(llhs) if log else np.prod(llhs)

    def complete_data_likelihood(
        self,
        t_stage: str | None = None,
        component: int | None = None,
        *,
        log: bool = True,
    ) -> float:
        """Compute the complete data likelihood of the model."""
        llhs = self.patient_mixture_likelihoods(
            t_stage=t_stage,
            component=component,
            log=log,
        )
        if component is not None:
            llhs[(np.isinf(llhs)) & (self.repeat_mixture_coefs()[:,component] == 0)] = 0
        else:
            llhs[(np.isinf(llhs)) & (self.repeat_mixture_coefs() == 0)] = 0
        resps = self.get_resps(
            t_stage=t_stage,
            component=component,
        ).to_numpy()
        if log:
            with np.errstate(invalid="ignore"):
                final_llh = resps * llhs
            nan_condition = np.isnan(final_llh) & np.isinf(llhs) & (resps == 0)
            final_llh[nan_condition] = 0
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
    ) -> np.ndarray:
        """Compute the distribution over possible states for all components.

        Do this for a given ``t_stage``. The result is a matrix with shape
        ``(num_components, num_states)``.
        """
        comp_state_dist_size = self.components[0].state_dist(t_stage).shape
        comp_state_dists = np.zeros((len(self.components), *comp_state_dist_size))
        for i, component in enumerate(self.components):
            comp_state_dists[i] = component.state_dist(t_stage)
        return comp_state_dists

    def posterior_state_dist(
        self,
        subgroup: str | None = None,
        given_params: types.ParamsType | None = None,
        given_diagnosis: types.DiagnosisType | None = None,
        t_stage: str | int = "early",
        midext: bool = None,
        central: bool = None,
    ) -> np.ndarray:
        """Compute a *fixed-weight mixture* posterior distribution over hidden states.

        This method returns the posterior distribution over the hidden state space
        conditioned on an observed clinical diagnosis. For mixture models, we use a
        **non–fully Bayesian** combination rule: component mixture weights are treated as
        **fixed, subsite-dependent population weights** and are **not updated** based on
        the patient's diagnosis.

        Concretely, let component index :math:`m=1,\\dots,M` have subsite-specific weight
        :math:`\\pi_m^s` (with :math:`\\sum_m \\pi_m^s = 1`). Each component defines its
        own prior hidden-state distribution :math:`p_m(\\mathbf{X}\\mid T)` (via the HMM)
        and a diagnosis model :math:`p(\\mathbf{Z}\\mid\\mathbf{X})` (via sensitivity and
        specificity, shared across components). For a given diagnosis :math:`\\mathbf{Z}`
        and T-stage :math:`T`, we compute the component-wise posterior state distribution

        .. math::

            p_m(\\mathbf{X}\\mid\\mathbf{Z}, T)
            \\propto
            p(\\mathbf{Z}\\mid\\mathbf{X})\\,p_m(\\mathbf{X}\\mid T),

        and then form the mixture posterior by averaging component posteriors using the
        fixed weights:

        .. math::

            p(\\mathbf{X}\\mid\\mathbf{Z}, T, s)
            =
            \\sum_{m=1}^M \\pi_m^s\\, p_m(\\mathbf{X}\\mid\\mathbf{Z}, T).

        Importantly, this differs from a fully Bayesian latent-class mixture posterior,
        which would replace :math:`\\pi_m^s` with diagnosis-dependent responsibilities
        :math:`p(m\\mid\\mathbf{Z},T,s)`. The fixed-weight averaging is intentional: it
        encodes the assumption that each patient belongs to a subsite-specific *mixture
        population*, and the mixture coefficients represent stable subsite composition
        rather than a patient-specific latent class to be inferred from :math:`\\mathbf{Z}`.

        Parameters
        ----------
        subgroup:
            Subgroup identifier used to select subsite-dependent mixture coefficients and,
            where applicable, subgroup-specific state distributions.
        given_params:
            Optional parameters to set on the model before computing the posterior.
            If provided as a dict, it is passed to ``set_params(**given_params)``;
            otherwise to ``set_params(*given_params)``.
        given_diagnosis:
            Observed diagnosis per modality. Example:

            .. code-block:: python

                given_diagnosis = {
                    "MRI": {"II": True, "III": False, "IV": False},
                    "PET": {"II": True, "III": True, "IV": None},
                }

            If ``None``, the method returns the (mixture) prior state distribution.
        t_stage:
            T-stage for which to compute the prior / posterior.
        midext:
            For midline-capable components, optionally condition on midline extension state.
            If ``None``, averages over extension status; otherwise selects the requested
            extension branch and renormalizes.
        central:
            Placeholder for future support of central tumors.

        Returns
        -------
        np.ndarray
            Posterior distribution over hidden states with the same shape as the state
            distribution returned by ``state_dist`` (after any midline aggregation/selection).
        """
        if given_params is not None:
            if isinstance(given_params, dict):
                self.set_params(**given_params)
            else:
                self.set_params(*given_params)
        if type(self.components[0]) == lymph.models.Midline:
            given_state_dists = self.state_dist(
                    t_stage=t_stage
                )
            if central:
                raise ValueError("Central not implemented yet")

            if midext is None:
                for i, given_state_dist in enumerate(given_state_dists):
                    given_state_dists[i] = given_state_dist[0] * (1 - self.all_midext_probs[subgroup]) + given_state_dist[1] * self.all_midext_probs[subgroup]
            else:
                given_state_dists = given_state_dists[:, int(midext), :, :]   # shape: (n, 32, 32)
                given_state_dists = given_state_dists / given_state_dists.sum(axis=(1, 2), keepdims=True)

        else:
            given_state_dists = self.state_dist(
            t_stage=t_stage
                )
            
        if given_diagnosis is None:
            combined_state_dist = np.sum(given_state_dists * self.get_mixture_coefs()[subgroup][:, np.newaxis], axis=0)
            return combined_state_dist
        combined_state_dist = np.zeros_like(given_state_dists[0])
        for i, state_dist in enumerate(given_state_dists):
            combined_state_dist += self.components[i].posterior_state_dist(given_state_dist = state_dist, given_diagnosis=given_diagnosis) * self.get_mixture_coefs()[subgroup][i]
        return combined_state_dist

    def risk(
        self,
        subgroup: str,
        involvement: types.PatternType,
        given_params: types.ParamsType | None = None,
        given_diagnosis: dict[str, types.PatternType] | None = None,
        t_stage: str = "early",
        midext: bool = None,
    ) -> float:
        """Compute the risk of a given ``involvement`` pattern given a clinical diagnosis.

        This method evaluates the probability of a specified lymph node involvement pattern
        conditional on an observed diagnosis and the model parameters.

        For mixture models, the risk is computed using a **fixed-weight mixture formulation**.
        First, the posterior risk is computed independently for each component HMM using the
        component-specific parameters. The final risk is then obtained by averaging these
        component risks using the subsite-specific mixture coefficients.

        Mathematically, the risk is computed as

        .. math::

            P(X_v = 1 \mid Z, T, s; \\theta, \\pi)
            =
            \\sum_{m=1}^{M} \\pi_m^{s}
            P(X_v = 1 \mid Z, T; \\theta_m),

        where :math:`\\pi_m^{s}` are the subsite-dependent mixture weights and
        :math:`P(X_v = 1 \mid Z, T; \\theta_m)` is the posterior risk computed from
        component :math:`m`.

        Importantly, the mixture weights remain **fixed** and are not updated based on the
        observed diagnosis. This reflects the interpretation that the mixture components
        represent population-level spread patterns rather than patient-specific latent
        classes.

        Parameters
        ----------
        subgroup :
            Subsite or subgroup identifier used to select the appropriate mixture
            coefficients.
        involvement :
            Dictionary specifying the lymph node involvement pattern of interest.
        given_params :
            Optional model parameters that will be set before computing the risk.
        given_diagnosis :
            Dictionary containing the observed diagnosis for each imaging modality.
        t_stage :
            Tumor T-stage used to compute the underlying hidden state distribution.
        midext :
            Optional flag specifying the midline extension status for midline models.

        Returns
        -------
        float
            The posterior risk of the specified involvement pattern given the diagnosis.
        """
        if (midext is not None) or (type(self.components[0]) != lymph.models.Midline) or (not self.split_midext):
            risk = 0
            if given_params is not None:
                if isinstance(given_params, dict):
                    self.set_params(**given_params)
                else:
                    self.set_params(*given_params)
            for i, component in enumerate(self.components):
                risk += component.risk(involvement = involvement, given_diagnosis= given_diagnosis, t_stage = t_stage, midext = midext)*self.get_mixture_coefs()[subgroup][i]
            return risk
                
        elif type(self.components[0]) == lymph.models.Midline and self.split_midext:
            comp_state_dist_size = self.components[0].noext.state_dist(t_stage).shape
            comp_state_dists = np.zeros((len(self.components), *comp_state_dist_size))
            for i, component in enumerate(self.components):
                diag_time_matrix = np.diag(component.get_distribution(t_stage).pmf)
                ipsi_dist_evo = component.ext.ipsi.state_dist_evo()
                contra_dist_evo = {}
                contra_dist_evo["noext"], contra_dist_evo["ext"] = component.contra_state_dist_evo(
                    unmodified_midext_prob=True
                )
                contra_dist_evo_marginalized = (
                    contra_dist_evo["noext"] * (1 - self.all_midext_probs[subgroup])
                    + contra_dist_evo["ext"] * self.all_midext_probs[subgroup]
                )
                comp_state_dists[i] = ipsi_dist_evo.T @ diag_time_matrix @ contra_dist_evo_marginalized

            # PAPER VERSION (non bayesian): posterior per component, then mix posteriors with fixed π
            weights = self.get_mixture_coefs()[subgroup]
            posterior_state_dist = np.zeros_like(comp_state_dists[0])
            for i, component in enumerate(self.components):
                post_i = component.posterior_state_dist(
                    given_state_dist=comp_state_dists[i],
                    given_diagnosis=given_diagnosis,
                )
                posterior_state_dist += weights[i] * post_i

            return self.components[0].marginalize(involvement, posterior_state_dist)