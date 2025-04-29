"""Test the functionality of the mixture model class."""

import unittest
import warnings

import numpy as np
import pandas as pd
from lymph.models import Unilateral

from lymixture import LymphMixture
from lymixture.utils import RESP_COLS

from . import fixtures

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class TestMixtureModel(fixtures.MixtureModelFixture):
    """Unit test the mixture model class."""

    def setUp(self) -> None:
        """Create intermediate and helper objects for the tests."""
        self.setup_rng(seed=42)
        self.setup_mixture_model(
            model_cls=Unilateral,
            num_components=3,
            graph_size="small",
            load_data=True,
        )
        self.setup_responsibilities()
        return super().setUp()

    def test_init(self):
        """Test the initialization of the mixture model."""
        self.assertIsInstance(self.mixture_model, LymphMixture)
        self.assertEqual(len(self.mixture_model.components), self.num_components)

    def test_load_patient_data(self):
        """Test the loading of patient data."""
        total_num_patients = 0
        for subgroup in self.mixture_model.subgroups.values():
            total_num_patients += len(subgroup.patient_data)

        self.assertEqual(total_num_patients, len(self.patient_data))

    def test_responsibility_indices(self):
        """Check if the returned index array is correct."""
        self.assertTrue(
            np.all(
                self.mixture_model.get_resp_indices(subgroup="C09")
                == (self.mixture_model.patient_data[fixtures.SIMPLE_SUBSITE] == "C09"),
            )
        )

    def test_set_responsibilities(self):
        """Test the assignment of responsibilities."""
        self.mixture_model.set_resps(self.resp)

        stored_resp = np.empty(shape=(0, self.num_components))
        for subgroup in self.mixture_model.subgroups.values():
            self.assertIn(RESP_COLS, subgroup.patient_data)
            stored_resp = np.vstack(
                [stored_resp, subgroup.patient_data[RESP_COLS].to_numpy()],
            )
        np.testing.assert_array_equal(self.resp, stored_resp)
        stored_resp = self.mixture_model.patient_data[RESP_COLS]
        np.testing.assert_array_equal(self.resp, stored_resp)
        stored_resp = self.mixture_model.get_resps()
        np.testing.assert_allclose(self.resp, stored_resp)

    def test_get_responsibilities(self):
        """Test accessing the responsibilities."""
        self.mixture_model.set_resps(self.resp)
        p_idx = self.rng.integers(low=0, high=len(self.patient_data))
        c_idx = self.rng.integers(low=0, high=self.num_components)
        self.assertEqual(
            self.resp[p_idx, c_idx],
            self.mixture_model.get_resps(component=c_idx).loc[p_idx],
        )


class DistributionsTestCase(fixtures.MixtureModelFixture):
    """Test the functionality provided by the diagnosis time distribution composite."""

    def setUp(self) -> None:
        """Create intermediate and helper objects for the tests."""
        self.setup_rng(seed=42)
        self.setup_mixture_model(
            model_cls=Unilateral,
            num_components=3,
            graph_size="small",
            load_data=True,
        )
        self.dists = {
            "early": fixtures.create_random_dist("frozen", max_time=10, rng=self.rng),
            "late": fixtures.create_random_dist(
                "parametric",
                max_time=10,
                rng=self.rng,
            ),
        }
        self.mixture_model.set_distribution("early", self.dists["early"])
        self.mixture_model.set_distribution("late", self.dists["late"])
        return super().setUp()

    def test_get_all_distributions(self) -> None:
        """Check if the distributions are returned correctly."""
        self.assertEqual(
            self.dists.keys(),
            self.mixture_model.get_all_distributions().keys(),
        )
        self.assertTrue(
            np.all(
                self.dists["early"] == self.mixture_model.get_distribution("early").pmf,
            ),
        )


class GetAndSetParamsTestCase(fixtures.MixtureModelFixture, unittest.TestCase):
    """Check the setters and getters of the model params."""

    def setUp(self) -> None:
        """Create intermediate and helper objects for the tests."""
        self.setup_rng(seed=42)
        self.setup_mixture_model(
            model_cls=Unilateral,
            num_components=3,
            graph_size="small",
            load_data=False,
        )
        self.dists = {
            "early": fixtures.create_random_dist("frozen", max_time=10, rng=self.rng),
            "late": fixtures.create_random_dist(
                "parametric",
                max_time=10,
                rng=self.rng,
            ),
        }
        self.mixture_model.set_distribution("early", self.dists["early"])
        self.mixture_model.set_distribution("late", self.dists["late"])
        return super().setUp()

    def test_set_params(self) -> None:
        """Ensure that all params are set."""
        params_to_set = {k: self.rng.uniform() for k in self.mixture_model.get_params()}
        self.mixture_model.set_params(**params_to_set)
        self.assertEqual(params_to_set, self.mixture_model.get_params())


class LikelihoodsTestCase(fixtures.MixtureModelFixture, unittest.TestCase):
    """Test the different likelihood functions."""

    def setUp(self) -> None:
        """Create intermediate and helper objects for the tests."""
        self.setup_rng(seed=42)
        self.setup_mixture_model(
            model_cls=Unilateral,
            num_components=3,
            graph_size="small",
            load_data=True,
        )
        self.mixture_model.set_modality("max_llh", spec=1.0, sens=1.0)
        self.dists = {
            "early": fixtures.create_random_dist(
                "parametric",
                max_time=10,
                rng=self.rng,
            ),
            "late": fixtures.create_random_dist(
                "parametric",
                max_time=10,
                rng=self.rng,
            ),
        }
        self.mixture_model.set_distribution("early", self.dists["early"])
        self.mixture_model.set_distribution("late", self.dists["late"])
        params_to_set = {k: self.rng.uniform() for k in self.mixture_model.get_params()}
        params_to_set["early_p"] = 0.0
        params_to_set["late_p"] = 1.0
        self.mixture_model.set_params(**params_to_set)
        return super().setUp()

    def test_component_patient_likelihoods(self) -> None:
        """Test the component- and patient-wise likelihoods.

        This test also ensures that one can access the likelihoods for a single
        component or subgroup.
        """
        llhs = self.mixture_model.patient_component_likelihoods(log=False)
        self.assertTrue(np.all(llhs >= 0))
        self.assertTrue(np.all(llhs <= 1))
        self.assertEqual(llhs.shape[0], len(self.patient_data))

        # make sure accessing a single component works
        llhs = self.mixture_model.patient_component_likelihoods(
            component=1,
            log=True,
        )
        self.assertEqual(llhs.shape[0], len(self.patient_data))

        # make sure accessing a single subgroup
        llhs = self.mixture_model.patient_component_likelihoods(
            subgroup="C01",
            log=True,
        )
        is_c01 = self.patient_data[fixtures.SIMPLE_SUBSITE] == "C01"
        num_c01 = len(self.patient_data[is_c01])
        self.assertEqual(llhs.shape, (num_c01, self.num_components))


if __name__ == "__main__":
    unittest.main()
