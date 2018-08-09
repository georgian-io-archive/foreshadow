"""
Parameter mapping utils
"""

import itertools

from copy import deepcopy

from ..preprocessor import Preprocessor

#TODO: Write default parameters here

config_dict = {

    'StandardScaler.with_std': [True, False]

}


def param_mapping(pipeline, X_df, y_df):

    preprocessors = [k
                     for k, v in pipeline.get_params().items()
                     if isinstance(v, Preprocessor)]

    configs = {p: _parse_json_params(pipeline.get_params()[p].from_json) for p in
               preprocessors}

    tasks = [{k: tsk for k, tsk in i}
             for i in itertools.product(*[[(p, t) for t in cfgs]
                                         for p, cfgs in configs.items()])]

    params = []
    for task in tasks:

        pipeline.set_params(**{"{}__from_json".format(k): v for k, v in task.items()})
        pipeline.fit(X_df, y_df)
        param = pipeline.get_params()

        explicit_params = {"{}__from_json".format(k): [param[k].serialize()]
                           for k, v in task.items()}

        params.append({**explicit_params, **_extract_config_params(param)})

    return params


def _parse_json_params(from_json):

    if from_json is None:
        return [None]

    combinations = from_json.pop("combinations", [])

    out = []
    for combo in combinations:

        t = [[(k, value) for value in eval(v)] for k, v in combo.items()]
        c = itertools.product(*t)
        d = [{k: v for k, v in i} for i in c]

        try:
            out += [_override_dict(i, from_json) for i in d]
        except Exception as e:
            raise ValueError("Malformed JSON. Check keys and try again.")

    if len(out) < 1:
        out.append(None)

    return out


def _override_dict(override, original):

    temp = deepcopy(original)
    for k, v in override.items():
        _set_path(k, v, temp)
    return temp


def _set_path(key, value, original):

    path = key.split('.')
    temp = original

    for p in path[:-1]:
        if isinstance(temp, list):
            temp = temp[int(p)]
        elif isinstance(temp, dict):
            temp = temp[p]
        else:
            raise ValueError("Malformed JSON Key")

    if isinstance(temp, list):
        temp[int(path[-1])] = value
    elif isinstance(temp, dict):
        temp[path[-1]] = value
    else:
        raise ValueError("Malformed JSON Key")


def _extract_config_params(param):

    out = {}

    for k, v in param.items():
        trace = k.split('__')
        names = []
        for i, t in enumerate(trace[0:-1]):
            key = "__".join(trace[0:i+1])
            name = type(param[key]).__name__
            if name not in ['ParallelProcessor', 'Pipeline', 'Preprocessor',
                            'FeatureUnion']:
                names.append(name)
        names.append(trace[-1])
        p = ".".join(names)
        if p in config_dict:
            out[k] = list(config_dict[p])

    return out
