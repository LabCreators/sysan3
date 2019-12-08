from numpy.polynomial import Polynomial as pm


class BasicBasisGenerator:
    def __init__(self, degree):
        self.degree = degree

    def basis_smoothed_chebyshev(self, degree):
        if not degree:
            return pm([1])
        elif degree == 1:
            return pm([-1, 2])
        else:
            return pm([-2, 4]) * self.basis_smoothed_chebyshev(degree - 1) - self.basis_smoothed_chebyshev(degree - 2)

    def basis_smoothed_legendre(self, degree):
        n = degree - 1
        if not degree or degree == 0:
            return pm([1])
        elif degree == 1:
            return pm([0, 1])
        else:
            return pm([(2 * n + 1)/(n + 1)]) * pm([0,1]) * self.basis_smoothed_legendre(n) - (n/(n+1)) * self.basis_smoothed_legendre(n - 1)

    def basis_laguerre(self, degree):
        k = degree - 1
        if not degree or degree == 0:
            return pm([1])
        elif degree == 1:
            return pm([1, -1])
        else:
            return (1/(k+1)) * (pm([2 * k + 1, -1]) * self.basis_laguerre(k) - k*self.basis_laguerre(k-1))

    def basis_hermite(self, degree):
        if not degree:
            return pm([1])
        elif degree == 1:
            return pm([0, 2])
        else:
            return pm([0, 2]) * self.basis_hermite(degree - 1) - 2 * (degree - 1) * self.basis_hermite(degree - 2)

    def basis_smoothed_chebyshev_2(self, degree):
        if not degree:
            return [pm([1])]
        elif degree == 1:
            return pm([-2, 4])
        else:
            return pm([-2, 4]) * self.basis_smoothed_chebyshev_2(degree - 1) - self.basis_smoothed_chebyshev_2(degree - 2)

    def basis_smoothed_chebyshev_2_shrinked(self, degree):
        if degree>1:
            return self.basis_smoothed_chebyshev_2(degree) / (degree - 1)
        else:
            return self.basis_smoothed_chebyshev_2(degree)


class BasisGenerator(BasicBasisGenerator):
    def __init__(self, degree, type):
        super().__init__(degree)
        self.type = type

    def _init_generator(self):
        generators_mapping = {'laguerre': self.basis_laguerre,
                              'hermite': self.basis_hermite,
                              'smoothed_legandr': self.basis_smoothed_legendre,
                              'smoothed_chebyshev': self.basis_smoothed_chebyshev,
                              'combined_cheb': self.basis_smoothed_chebyshev_2_shrinked,
                              'sh_cheb_2': self.basis_smoothed_chebyshev_2}

        return generators_mapping

    def generate(self):
        generator = self._init_generator().get(self.type)
        for deg in range(self.degree):
            yield generator(deg)