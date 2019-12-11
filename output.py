import numpy as np
import matplotlib.pyplot as plt
from os import name as os_name
from task_solution import Solve
from basis_generators import BasisGenerator
from depict_poly import _Polynom
from numpy.polynomial import Polynomial as pnm
import itertools


class PolynomialBuilder(object):
    def __init__(self, solver, solution):
        assert isinstance(solver, Solve)
        self._solution = solution
        self._solver = solver
        max_degree = max(solver.p)
        if solver.poly_type == 'smoothed_chebyshev':
            self.symbol = 'T'
            self.basis = BasisGenerator(max_degree,'smoothed_chebyshev').basis_smoothed_chebyshev(max_degree)
        elif solver.poly_type == 'smoothed_legandr':
            self.symbol = 'P'
            self.basis = BasisGenerator(max_degree,'smoothed_legandr').basis_smoothed_legendre(max_degree)
        elif solver.poly_type == 'laguerre':
            self.symbol = 'L'
            self.basis = BasisGenerator(max_degree,'laguerre').basis_laguerre(max_degree)
        elif solver.poly_type == 'hermite':
            self.symbol = 'H'
            self.basis = BasisGenerator(max_degree,'hermite').basis_hermite(max_degree)
        elif solver.poly_type == 'combined_cheb':
            self.symbol = 'CC'
            self.basis = BasisGenerator(max_degree, 'combined_cheb').basis_hermite(max_degree)
        elif solver.poly_type == 'sh_cheb_2':
            self.symbol = 'SC'
            self.basis = BasisGenerator(max_degree, 'sh_cheb_2').basis_hermite(max_degree)
        self.a = self._solution[9]
        self.c = self._solution[13]
        self.lamb = self._solution[5]
        self.dt = self._solution[0]
        self.dt_norm = self._solution[1]
        self.y = self._solution[-12]
        self.y_norm = self._solution[-11]
        self.deg = self._solution[-2]
        self.errors = self._solution[-6]
        self.errors.columns = ['Y{}'.format(i+1) for i in range(self.deg[-1])]
        self.errors_norm = self._solution[-5]
        self.errors_norm.columns = ['Y{}'.format(i + 1) for i in range(self.deg[-1])]
        self.y.columns = ['Y{}'.format(i + 1) for i in range(self.deg[-1])]
        self.y_norm.columns = ['Y{}'.format(i + 1) for i in range(self.deg[-1])]
        self.ft = self._solution[-8]
        self.ft_norm = self._solution[-7]
        self.p = self._solution[-1]
        use_cols = [['X{}{}'.format(i + 1, j + 1) for j in
                     range(len([el for el in self.dt.columns if el.find('X{}'.format(i + 1)) != -1]))]
                    for i in range(len(self.deg))][:-1]
        self.minX = [self.dt[el].min(axis=0).min() for el in use_cols]
        self.maxX = [self.dt[el].max(axis=0).max() for el in use_cols]
        self.minY = self.y.min(axis=0).min()
        self.maxY = self.y.max(axis=0).max()

    def _form_lamb_lists(self):

        self.psi = [[[self.lamb.loc[i, 'lambda_{}'.format(j + 1)][0].tolist()] for j in range(self.deg[-1])]
                for i in range(self.deg[-1])]

    def _transform_to_standard(self, coefs):
        """
        Transforms special polynomial to standard
        :param coeffs: coefficients of special polynomial
        :return: coefficients of standard polynomial
        """
        std_coeffs = np.zeros(coefs.shape)
        for index in range(coefs.shape[0]):
            try:
                cp = self.basis.coef[index]
                cp.resize(coefs.shape)
                std_coeffs += coefs[index] * cp
            except:
                return std_coeffs
        return std_coeffs

    def _print_psi_i_jk(self, i, j, k):
        """
        Returns string of Psi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :param k: an index for vector component
        :return: result string
        """
        try:
            return (' * ').join(['{0:.6f}*{symbol}{deg}(x{1}{2})'.format(self.lamb.iloc[i][j][0][k][n], j + 1, k + 1,
                                                                     symbol='T', deg=n)
                             for n in range(len(self.lamb.iloc[i][j][0][k]))])
        except:
            return (' * ').join(['{0:.6f}*{symbol}{deg}(x{1}{2})'.format(self.lamb.iloc[i][j][0][n], j + 1, k + 1,
                                                                  symbol='T', deg=n)
                          for n in range(len(self.lamb.iloc[i][j][0]))])

    def _print_phi_i_j(self, i, j):
        """
        Returns string of Phi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :return: result string
        """
        try:
            return (' * ').join(list(
            itertools.chain(*[['{0:.6f}*{symbol}{deg}(x{1}{2})'.format(self.a.loc[i, j][sum(self.p[:j]) + k] * self.lamb.iloc[i][j][0][k][n],
                                                                       j + 1, k + 1, symbol=self.symbol, deg=n)
                               for n in range(len(self.lamb.iloc[i][j][0][k]))] for k in range(len(self.lamb.iloc[i][j]))])))
        except:
            return (' * ').join(list(
                itertools.chain(*[['{0:.6f}*{symbol}{deg}(x{1}{2})'.format(
                    self.a.loc[i, j][sum(self.p[:j]) + k] * self.lamb.iloc[i][j][0][n],
                    j + 1, k + 1, symbol=self.symbol, deg=n)
                                   for n in range(len(self.lamb.iloc[i][j][0]))] for k in
                                  range(len(self.lamb.iloc[i][j]))])))

    def _print_F_i(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        try:
            return (' * ').join(list(itertools.chain(*list(
            itertools.chain(*[[['{0:.6f}*{symbol}{deg}(x{1}{2})'.format(self.c.loc[0, j] * self.a.loc[i, j][sum(self.p[:j]) + k] *
                                                                        self.lamb.iloc[i][j][0][k][n],
                                                                        j + 1, k + 1, symbol=self.symbol, deg=n)
                                for n in range(len(self.psi[i][j][k]))]
                               for k in range(len(self.psi[i][j]))]
                              for j in range(self.deg[-1])])))))
        except:
            return (' * ').join(list(itertools.chain(*list(
                itertools.chain(*[
                    [['{0:.6f}*{symbol}{deg}(x{1}{2})'.format(self.c.loc[0, j] * self.a.loc[i, j][sum(self.p[:j]) + k] *
                                                              self.lamb.iloc[i][j][0][n],
                                                              j + 1, k + 1, symbol=self.symbol, deg=n)
                      for n in range(len(self.psi[i][j][k]))]
                     for k in range(len(self.psi[i][j]))]
                    for j in range(self.deg[-1])])))))

    def _print_F_i_transformed_recovered(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        power_sum = 0
        for j in range(self.deg[-1]):
            for k in range(len(self.lamb.iloc[i][j])):
                shift = sum(self.p[:j]) + k
                diff = self.maxX[j] - self.minX[j]
                mult_poly = pnm([- self.minX[j][k] / diff, 1 / diff])
                power_sum += self.c[i][j] * self.a[i][shift] * self.lamb.iloc[i][j][0][k][0]
                for n in range(1, len(self.lamb.iloc[i][j][0][k])):
                    res_polynomial = self.basis[n](mult_poly) + 1
                    coeffs = res_polynomial.coef
                    summands = ['{0}(x{1}{2})^{deg}'.format(coeffs[index], j + 1, k + 1, deg=index)
                                for index in range(1, len(coeffs))]
                    summands.insert(0, str(coeffs[0]))
                    strings.append(
                        '({repr})^({0:.6f})'.format(self.c[i][j] * self.a[i][shift] * self.lamb.iloc[i][j][0][k][n],
                                                    j + 1, k + 1, repr=' + '.join(summands)))
            strings.insert(0, str((self.maxY[i] - self.minY[i]) * (1 + self.basis[0].coef[0]) ** (power_sum)))
            return ' * '.join(strings) + ' + ' + str((2 * self.minY[i] - self.maxY[i]))

    def _print_F_i_transformed(self, i):
        """
        Returns string of F function in regular polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        power_sum = 0
        for j in range(len(self.p)):
            for k in range(len(self.lamb.iloc[i][j])):
                shift = sum(self.deg[:j]) + k
                try:
                    power_sum += self.c[i][j] * self.a[i][shift] * self.lamb.iloc[i][j][0][k][0]
                except:
                    power_sum += self.c[i][j] * self.a[i][shift] * self.lamb.iloc[i][j][0][k]
                for n in range(1, len(self.lamb.iloc[i][j][0][k])):
                    summands = ['{0}(x{1}{2})^{deg}'.format(self.basis[n].coef[index], j + 1, k + 1, deg=index)
                                for index in range(1, len(self.basis[n].coef)) if self.basis[n].coef[index] != 0]
                    if self.basis[n].coef[0] != -1:
                        summands.insert(0, str(1 + self.basis[n].coef[0]))
                    strings.append('({repr})^({0:.6f})'.format(self.c[i][j] * self.a[i][shift] * self.lamb.iloc[i][j][0][k][n],
                                                               j + 1, k + 1, repr=' + '.join(summands)))
        strings.insert(0, str((1 + self.basis[0].coef[0]) ** (power_sum)))
        return ' * '.join(strings)

    def get_results(self):
        """
        Generates results based on given solution
        :return: Results string
        """
        if self.symbol == 'CC':
            return ''
        self._form_lamb_lists()
        psi_strings = ['Psi^{0}_[{1},{2}]={result} - 1\n'.format(i + 1, j + 1, k + 1,
                                                                 result=self._print_psi_i_jk(i, j, k))
                       for i in range(self.y.shape[1])
                       for j in range(self.deg[-1])
                       for k in range(self.deg[j])]
        phi_strings = ['Phi^{0}_[{1}]={result} - 1\n'.format(i + 1, j + 1, result=self._print_phi_i_j(i, j))
                       for i in range(self.y.shape[1])
                       for j in range(self.deg[-1])]
        f_strings = ['F^{0} in special basis:\n{result} - 1\n'.format(i + 1, result=self._print_F_i(i))
                     for i in range(self.y.shape[1])]
        f_strings_transformed = [
            'F^{0} in standard basis:\n{result} - 1\n'.format(i + 1, result=self._print_F_i_transformed(i))
            for i in range(self.y.shape[1])]
        f_strings_transformed_denormed = ['F^{0} in standard basis '
                                          'denormed:\n{result}\n'.format(i + 1, result=
        self._print_F_i_transformed_recovered(i))
                                          for i in range(self.y.shape[1])]
        return '\n'.join(psi_strings + phi_strings + f_strings + f_strings_transformed + f_strings_transformed_denormed)

    def plot_graphs(self):
        fig, axes = plt.subplots(4, self.y.shape[1], figsize=(20, 20))

        for i in range(self.y.shape[1]):
            axes[0][i].plot(self.y['Y{}'.format(i + 1)])
            axes[0][i].plot(self.ft.loc[:, i])
            axes[0][i].legend(['True', 'Predict'])
            axes[0][i].set_title('Not normalized version: Degrees: {}, Poly type: {}, Lambdas: {}'.format(self.p,
                                                                                                          self._solver.poly_type,
                                                                                                          self._solver.splitted_lambdas))

        for i in range(self.y.shape[1]):
            axes[1][i].plot(self.errors.apply(abs).loc[:, 'Y{}'.format(i + 1)])
            axes[1][i].set_title('Not normalized version: Degrees: {}, Poly type: {}, Lambdas: {}'.format(self.p,
                                                                                                          self._solver.poly_type,
                                                                                                          self._solver.splitted_lambdas))


        for i in range(self.y.shape[1]):
            axes[2][i].plot(self.y_norm['Y{}'.format(i + 1)])
            axes[2][i].plot(self.ft_norm.loc[:, i])
            axes[2][i].legend(['True', 'Predict'])
            axes[2][i].set_title('Normalized version: Degrees: {}, Poly type: {}'.format(self.p, self._solver.poly_type))

        for i in range(self.y.shape[1]):
            axes[3][i].plot(self.errors_norm.apply(abs).loc[:, 'Y{}'.format(i + 1)])
            axes[3][i].set_title('Normalized version: Degrees: {}, Poly type: {}'.format(self.p, self._solver.poly_type))

        plt.savefig('graphics/graph_{}_{}_{}_{}.png'.format(self.p, self._solver.poly_type,
                                                         self._solver.weights, self._solver.splitted_lambdas))
        manager = plt.get_current_fig_manager()
        manager.set_window_title('Graph')
        if os_name == 'posix':
            fig.show()
        else:
            plt.show()
        plt.waitforbuttonpress(0)
        plt.close(fig)

    def compare_vals(self, name, real, predicted, reconstructed=None):
        fig = plt.figure()
        axes = plt.axes()
        r = np.arange(len(real))
        axes.set_title(name)
        axes.set_xlim(0, len(real))
        axes.grid()
        axes.plot(r, predicted, label='predicted')
        if reconstructed != None:
            axes.plot(r, reconstructed, label='reconstructed')
        axes.plot(r, real, label='real')
        axes.legend(loc='upper right', fontsize=16)
        if os_name == 'posix':
            fig.show()
        else:
            plt.show()

    def plot_graphs_with_prediction(self, steps):
        XF, YF = self._solution._build_predicted(steps)
        for i in range(self.dt_norm.shape[0]):
            for j in range(self.dt_norm.loc[i, :].shape[1]):
                self.compare_vals('X{}{}'.format(i + 1, j + 1), self.dt_norm.loc[i, j], XF[i][j])
        for i in range(self._solution.dim[3]):
            self.compare_vals('Y{}'.format(i + 1), self.y_norm[:, i], YF[:, i],
                              self.ft_norm[:, i])
