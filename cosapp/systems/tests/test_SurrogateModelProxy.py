import pytest
import pickle
import numpy as np
from contextlib import nullcontext as does_not_raise

from cosapp.systems.systemSurrogate import SurrogateModelProxy
from cosapp.utils.surrogate_models import (
    SurrogateModel,
    KrigingSurrogate,
    FloatKrigingSurrogate,
    NearestNeighbor,
    ResponseSurface,
)


@pytest.mark.parametrize("cls", [
    KrigingSurrogate,
    FloatKrigingSurrogate,
    NearestNeighbor,
    ResponseSurface,
])
def test_SurrogateModelProxy_no_training_data(cls):
    surrogate = SurrogateModelProxy(cls())

    with pytest.raises(RuntimeError, match="has not been trained"):
        surrogate.predict([0., 1.])


class WrongApiModel:
    """Bogus model with training and prediction,
    but not derived from `SurrogateModel`."""
    def train(self, x, y):
        pass

    def predict(self, x):
        return x


class CustomModel(SurrogateModel):
    """Bogus implementation of `SurrogateModel` API."""
    def train(self, x, y):
        pass

    def predict(self, x):
        return x


@pytest.mark.parametrize("wrappee, expected", [
    (FloatKrigingSurrogate(), does_not_raise()),
    (CustomModel(), does_not_raise()),
    (WrongApiModel(), pytest.raises(TypeError)),
    ("string", pytest.raises(TypeError)),
])
def test_SurrogateModelProxy__init__(wrappee, expected):
    with expected:
        proxy = SurrogateModelProxy(wrappee)
        assert hasattr(proxy, 'trained')


@pytest.mark.parametrize("cls", [
    FloatKrigingSurrogate,
    NearestNeighbor,
])
def test_SurrogateModelProxy_pickle(tmp_path, cls):
    model = cls()
    with pytest.raises(AttributeError):
        model.trained
    # Check that underlying model is pickable
    with does_not_raise():
        with open(tmp_path / 'model.pickle', 'wb') as f:
            pickle.dump(model, f)
    
    proxy = SurrogateModelProxy(model)
    assert not proxy.trained

    # Serialization test on proxy
    with does_not_raise():
        with open(tmp_path / 'proxy.pickle', 'wb') as f:
            pickle.dump(proxy, f)

    with open(tmp_path / 'proxy.pickle', 'rb') as f:
        loaded = pickle.load(f)

    assert not loaded.trained

    with pytest.raises(RuntimeError, match="has not been trained"):
        loaded.predict([0., 1.])

    # Serializing test after training
    x = y = np.reshape(range(20), (10, -1))  # bogus data
    proxy.train(x, y)
    assert proxy.trained

    with open(tmp_path / 'proxy.pickle', 'wb') as f:
        pickle.dump(proxy, f)

    with open(tmp_path / 'proxy.pickle', 'rb') as f:
        loaded = pickle.load(f)
    
    assert loaded.trained

    with does_not_raise():
        loaded.predict([0., 1.])


@pytest.mark.parametrize("cls", [
    KrigingSurrogate,
    FloatKrigingSurrogate,
    NearestNeighbor,
    ResponseSurface,
])
def test_SurrogateModelProxy_get_type(cls):
    model = cls()
    proxy = SurrogateModelProxy(model)
    assert proxy.get_type() is cls
