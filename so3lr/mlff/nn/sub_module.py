from abc import abstractmethod

import flax.linen as nn


class BaseSubModule(nn.Module):

    @abstractmethod
    def __dict_repr__(self):
        pass

    def reset_prop_keys(self, prop_keys):
        self.prop_keys.update(prop_keys)
