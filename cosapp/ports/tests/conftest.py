import pytest

from cosapp.ports import Port

# <codecell>

@pytest.fixture(scope="function")
def DummyPort():
    """Factory creating a dummy port with custom attributes"""
    def Factory(name, direction, init_values=None, **options):
        # options listed below should be (args, kwargs) or a list thereof
        method_dict = {
            # correspondance method / option
            # for example: `add_variable` <-> `variables`
            "add_" + option[:-1] : options.pop(option, None)
            for option in ["variables"]
        }
        base = options.pop("base", Port)
        owner = options.pop("owner", None)
        class PrototypePort(base):
            def setup(self, **options):
                super().setup(**options)
                for method, values in method_dict.items():
                    if values is None:
                        continue
                    if not isinstance(values, list):
                        values = [values]
                    for args, kwargs in values:  # expects a list of (tuple, dict)
                        getattr(self, method)(*args, **kwargs)
                if owner is not None:
                    self.owner = owner
        return PrototypePort(name, direction, init_values, **options)
    return Factory
