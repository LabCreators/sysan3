from numpy.polynomial import Polynomial as pnm
from output import PolynomialBuilder


class PolynomialBuilderExpTh(PolynomialBuilder):
    def __init__(self, solver, solution):
        super().__init__(solver, solution)

    def _print_psi_i_jk(self, i, j, k, mode=0):
        """
        Returns string of Psi function
        mode = 0 -  in special polynomial form
        mode = 1 -  in regular polynomial form
        mode = 2 -  in regular polynomial form with restored X
        :param i: an index for Y
        :param j: an index to choose vector from X
        :param k: an index for vector component
        :return: result string
        """
        strings = list()
        for n in range(len(self.lamb[i][j][k])):
            inner = 'stub'
            if mode == 0:
                inner = '{symbol}{deg}(x{0}{1})'.format(j + 1, k + 1, symbol=self.symbol, deg=n)
            elif mode == 1:
                inner = str(self.basis[n].coef[0])
                if n > 0:
                    inner += ' + ' + ' + '.join('({coef})(x{0}{1})^{deg}'.format(j + 1, k + 1, coef=coef, deg=index)
                                                for index, coef in enumerate(self.basis[n].coef) if index > 0)
            elif mode == 2:
                diff = self.maxX[j][k] - self.minX[j][k]
                mult_poly = pnm([- self.minX[j][k] / diff, 1 / diff])
                cur_poly = self.basis[n](mult_poly)
                inner = str(cur_poly.coef[0])
                if n > 0:
                    inner += ' + ' + ' + '.join('({coef})(x{0}{1})^{deg}'.format(j + 1, k + 1, coef=coef, deg=index)
                                                for index, coef in enumerate(cur_poly.coef) if index > 0)
            strings.append('exp({0:.6f}*tanh({inner}))'.format(self.lamb[i][j][k][n], inner=inner))
        return ' * '.join(strings) + ' - 1'

    def _print_phi_i_j(self, i, j, mode=0):
        """
        Returns string of Phi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :return: result string
        """
        strings = list()
        for k in range(len(self.lamb[i][j])):
            strings.append('exp({0:.6f}*tanh({inner}))'.format(self.a[i][sum(self.deg[:j]) + k],
                                                               inner=self._print_psi_i_jk(i, j, k, mode)))
        return ' * '.join(strings) + ' - 1'

    def _print_F_i(self, i, mode=0):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        for j in range(3):
            strings.append('exp({0:.6f}*tanh({inner}))'.format(self.c[i][j], inner=self._print_phi_i_j(i, j, mode)))
        if mode == 2:
            strings.insert(0, str(self.maxY[i] - self.minY[i]))
            return ' * '.join(strings) + ' + (' + str((2 * self.minY[i] - self.maxY[i])) + ')'
        else:
            return ' * '.join(strings) + ' - 1'

    def get_results(self):
        """
        Generates results based on given solution
        :return: Results string
        """
        if self.symbol == 'CC':
            return ''

        self._form_lamb_lists()
        psi_strings = ['Psi^{0}_[{1},{2}]={result}\n'.format(i + 1, j + 1, k + 1,
                                                             result=self._print_psi_i_jk(i, j, k))
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)
                       for k in range(self._solution.dim[j])]
        phi_strings = ['Phi^{0}_[{1}]={result}\n'.format(i + 1, j + 1, result=self._print_phi_i_j(i, j))
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)]
        f_strings = ['F^{0} in special basis:\n{result}\n'.format(i + 1, result=self._print_F_i(i))
                     for i in range(self._solution.Y.shape[1])]
        f_strings_transformed = ['F^{0} in standard basis:\n{result}\n'.format(i + 1, result=self._print_F_i(i, mode=1))
                                 for i in range(self._solution.Y.shape[1])]
        f_strings_transformed_denormed = ['F^{0} in standard basis '
                                          'denormed:\n{result}\n'.format(i + 1, result=self._print_F_i(i, mode=2))
                                          for i in range(self._solution.Y.shape[1])]
        return '\n'.join(psi_strings + phi_strings + f_strings + f_strings_transformed + f_strings_transformed_denormed)