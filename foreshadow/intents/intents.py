"""
Intent base and registry defentions
"""

_registry = []


def _register_intent(cls_target):
    global _registry
    _registry.append(cls_target)


def get_registry():
    global _registry
    return _registry


class IntentRegistry(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        if not name == 'BaseIntent':
            cls._check_required_attributes()
            _check_required_instance_attributes()
            _register_intent(cls)
        return cls

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj._verify_pipeline_setup()
        return obj


class BaseIntent(object, metaclass=IntentRegistry):
    intent = None
    dtype = None
    parent = None
    children = None
    single_pipeline = None
    multi_pipeline = None

    def __new__(cls, *args, **kwargs):
        if cls is BaseIntent:
            raise TypeError("BaseIntent may not be instantiated")
        return object.__new__(cls, *args, **kwargs)

    @classmethod
    def is_fit(cls, series):
        """Determines whether intent is the appropriate fit

        Args:
            series (pd.Series): data to determine intent fit

        Returns:
            True/False
        """
        raise NotImplementedError('Function is not immplemented')

    @classmethod
    def _check_required_class_attributes(cls):
        """Validate class variables are setup properly"""
        not_implemented = lambda x, y: 'Subclass must define {} attribute.\n{}'\
            .format(x, y)
        if cls.intent is None:
            raise NotImplementedError(
                not_implemented(
                    'cls.intent',
                    'This attribute should define the name of the intent.'
                )
            )
        elif cls.dtype is None:
            raise NotImplementedError(
                not_implemented(
                    'cls.dtype',
                    'This attribute should define the dtype of the intent.'
                )
            )
        elif cls.parent is None:
            raise NotImplementedError(
                not_implemented(
                    'cls.parent',
                    'This attribute should define the parent of the intent.'
                )
            )
        elif cls.children is None:
            raise NotImplementedError(
                not_implemented(
                    'cls.children',
                    'This attribute should define the children of the intent.'
                )
            )
        elif cls.single_pipeline_spec is None:
            raise NotImplementedError(
                not_implemented(
                    'cls.single_pipeline_spec',
                    ('This attribute should define the sklearn single '
                     'pipeline spec of the intent.')
                )
            )
        elif cls.multi_pipeline_spec is None:
            raise NotImplementedError(
                not_implemented(
                    'cls.multi_pipeline_spec',
                    ('This attribute should define the sklearn multi pipeline '
                     'spec of the intent.')
                )
            )

    def _check_required_instance_attributes(self):
        """Validate instance variables are setup properly"""
        not_implemented = lambda x, y: 'Subclass must define {} attribute.\n{}'\
            .format(x, y)
        if self.intent is None:
            raise NotImplementedError(
                not_implemented(
                    'self.single_pipeline_spec',
                    'This attribute should define the default single pipeline.'
                )
            )
        elif self.dtype is None:
            raise NotImplementedError(
                not_implemented(
                    'self.multi_pipeline_spec',
                    'This attribute should define the default multi pipeline.'
                )
            )

    def _verify_pipeline_setup(self):
        not_implemented = lambda x, y: 'Subclass initialize {} to {}'\
            .format(x, y)
        if not hasattr(self, single_pipeline):
            raise NotImplementedError(
                not_implemented(
                    'self.single_pipeline_spec',
                    'self.single_pipeline'
                )
            )
        if not hasattr(self, single_pipeline):
            raise NotImplementedError(
                not_implemented(
                    'self.multi_pipeline_spec',
                    'self.multi_pipeline'
                )
            )
