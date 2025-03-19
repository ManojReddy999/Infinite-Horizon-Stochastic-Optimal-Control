import numpy as np


class ValueFunction:
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        self.T = T
        self.ex_space = ex_space
        self.ey_space = ey_space
        self.etheta_space = etheta_space
        self.values = np.zeros((T, len(ex_space), len(ey_space), len(etheta_space)))

    def copy_from(self, other):
        """
        Update the underlying value function storage with another value function
        """
        # TODO: your implementation
        if isinstance(other, ValueFunction):
            self.values = np.copy(other.values)
        else:
            raise ValueError("Input must be a ValueFunction instance")

    def update(self, t, ex, ey, etheta, target_value):
        """
        Update the value function at given states
        Args:
            t: time step
            ex: x position error
            ey: y position error
            etheta: theta error
            target_value: target value
        """
        # TODO: your implementation
        ex_index = self.ex_space.index(ex)
        ey_index = self.ey_space.index(ey)
        etheta_index = self.etheta_space.index(etheta)
        self.values[t, ex_index, ey_index, etheta_index] = target_value

    def __call__(self, t, ex, ey, etheta):
        """
        Get the value function results at given states
        Args:
            t: time step
            ex: x position error
            ey: y position error
            etheta: theta error
        Returns:
            value function results
        """
        # TODO: your implementation
        ex_index = self.ex_space.index(ex)
        ey_index = self.ey_space.index(ey)
        etheta_index = self.etheta_space.index(etheta)
        return self.values[t, ex_index, ey_index, etheta_index]

    def copy(self):
        """
        Create a copy of the value function
        Returns:
            a copy of the value function
        """
        # TODO: your implementation
        new_copy = ValueFunction(self.T, self.ex_space, self.ey_space, self.etheta_space)
        new_copy.copy_from(self)
        return new_copy


class GridValueFunction(ValueFunction):
    """
    Grid-based value function
    """
    # TODO: your implementation
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        super().__init__(T, ex_space, ey_space, etheta_space)
        self.values = np.full((T, len(ex_space), len(ey_space), len(etheta_space)), np.inf)

    def copy_from(self, other):
        self.values = np.copy(other.values)

    def update(self, t, ex, ey, etheta, target_value):
        ix = np.digitize([ex], self.ex_space, right=True)[0] - 1
        iy = np.digitize([ey], self.ey_space, right=True)[0] - 1
        it = np.digitize([etheta], self.etheta_space, right=True)[0] - 1
        self.values[t, ix, iy, it] = target_value

    def __call__(self, t, ex, ey, etheta):
        ix = np.digitize([ex], self.ex_space, right=True)[0] - 1
        iy = np.digitize([ey], self.ey_space, right=True)[0] - 1
        it = np.digitize([etheta], self.etheta_space, right=True)[0] - 1
        return self.values[t, ix, iy, it]

    def copy(self):
        new_copy = GridValueFunction(self.T, self.ex_space, self.ey_space, self.etheta_space)
        new_copy.copy_from(self)
        return new_copy


class FeatureValueFunction(ValueFunction):
    """
    Feature-based value function
    """
    # TODO: your implementation
    def __init__(self, T: int, ex_space, ey_space, etheta_space, alpha=1.0, beta_t=1.0, beta_e=1.0):
        super().__init__(T, ex_space, ey_space, etheta_space)
        self.alpha = alpha
        self.beta_t = beta_t
        self.beta_e = beta_e
        self.theta = np.zeros((T, len(ex_space) * len(ey_space) * len(etheta_space)))

    def rbf(self, t, ex, ey, etheta, ti, exi, eyi, ethetai):
        return self.alpha * np.exp(-self.beta_t * (t - ti)**2 - self.beta_e * ((ex - exi)**2 + (ey - eyi)**2 + (etheta - ethetai)**2))

    def construct_features(self, t, ex, ey, etheta):
        features = []
        for ti in range(self.T):
            for exi in self.ex_space:
                for eyi in self.ey_space:
                    for ethetai in self.etheta_space:
                        features.append(self.rbf(t, ex, ey, etheta, ti, exi, eyi, ethetai))
        return np.array(features)

    def copy_from(self, other):
        if not isinstance(other, FeatureValueFunction):
            raise TypeError("Can only copy from another FeatureValueFunction instance")
        np.copyto(self.theta, other.theta)

    def update(self, t, ex, ey, etheta, target_value):
        features = self.construct_features(t, ex, ey, etheta)
        self.theta += features * (target_value - self.__call__(t, ex, ey, etheta))

    def __call__(self, t, ex, ey, etheta):
        features = self.construct_features(t, ex, ey, etheta)
        return np.dot(self.theta, features)

    def copy(self):
        new_instance = FeatureValueFunction(self.T, self.ex_space, self.ey_space, self.etheta_space, self.alpha, self.beta_t, self.beta_e)
        new_instance.copy_from(self)
        return new_instance


