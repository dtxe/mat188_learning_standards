from types import SimpleNamespace


class Configs(SimpleNamespace):
    '''
    Extend SimpleNamespace to return None instead of raising AttributeError 
    and recursively create Configs objects from nested dictionaries
    '''

    def __init__(self, dictionary: dict, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, Configs(value))
            else:
                self.__setattr__(key, value)

    def __getattribute__(self, value):
        try:
            return super().__getattribute__(value)
        except AttributeError:
            return None
