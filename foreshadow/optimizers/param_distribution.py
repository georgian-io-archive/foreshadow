"""Classes to be configured by user for customizing parameter tuning."""
import foreshadow as fs
import foreshadow.serializers as ser


class ParamSpec(fs.Foreshadow, ser.ConcreteSerializerMixin):
    def __init__(self, parameter_distribution):
        self.parameter_distribution = parameter_distribution

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + fs.Foreshadow._get_param_names()

    def set_params(self, **params):



if __name__ == '__main__':
    ParamSpec().to_json("test")
