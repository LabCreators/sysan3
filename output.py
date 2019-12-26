import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from numpy.polynomial import Polynomial as pnm
from os import name as os_name
import parser

from solve import Solve
import basis_gen as b_gen
import itertools

class PolynomialBuilder(object):
    def __init__(self, solution):
        assert isinstance(solution, Solve)
        self._solution = solution
        max_degree = max(solution.deg) - 1
        if solution.poly_type == 'combined_cheb':
            self.symbol = 'CC'
        elif solution.poly_type == 'laguerre':
            self.symbol = 'L'
            self.basis = b_gen.basis_laguerre(max_degree)
        elif solution.poly_type == 'sh_cheb_2':
            self.symbol = 'U'
            self.basis = b_gen.basis_sh_chebyshev_2_shrinked(max_degree)
        assert self.symbol
        self.a = solution.a.T.tolist()
        self.c = solution.c.T.tolist()
        self.minX = [X.min(axis=0).getA1() for X in solution.X_]
        self.maxX = [X.max(axis=0).getA1() for X in solution.X_]
        self.minY = solution.Y_.min(axis=0).getA1()
        self.maxY = solution.Y_.max(axis=0).getA1()
        self.x_bort_net = [1,2,3,4,5,6,7,8,9,10]
        self.y_bort_net = [23,28,23,28,23,28,23,28,23,28]
        self.bort_net = self.parse(1)
        self.x_fuel = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.y_fuel = [50, 60, 50, 60, 50, 60, 50, 60, 50, 60]
        self.x_battery = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.y_battery = [23, 28, 23, 28, 23, 28, 23, 28, 23, 28]
        self.current_time = 0

    def parse(self, iteration):
        f = open("bort_net", 'r')
        # 627 iterations count, 5 number of charts, 60 max number of
        # points, 2 point`s dimension
        arrs = np.zeros((5, 2, 60), dtype='float')
        content = f.read()
        arrays = content.split("\t\n")
        for index_array, arr in enumerate(arrays):
            raw_points = arr.split(" )	( ")
            for index_point, point in enumerate(raw_points):
                xy = point.split(", ")
                if index_array == 4:
                    if index_point == 0:
                        arrs[index_array][0][index_point + 50] = 490
                        arrs[index_array][1][index_point + 50] = float(xy[1].replace(",", "."))
                    else:
                        arrs[index_array][0][index_point + 50] = float(xy[0])
                        arrs[index_array][1][index_point + 50] = float(xy[1].replace(",", ".").replace(" )", ""))
                else:
                    if index_point == 0:
                        arrs[index_array][0][index_point] = 0
                        arrs[index_array][1][index_point] = float(xy[1].replace(",", "."))
                    else:
                        arrs[index_array][0][index_point] = float(xy[0])
                        arrs[index_array][1][index_point] = float(xy[1].replace(",", ".").replace(" )", ""))
        return arrs

    def _form_lamb_lists(self):
        """
        Generates specific basis coefficients for Psi functions
        """
        self.lamb = list()
        for i in range(self._solution.Y.shape[1]):  # `i` is an index for Y
            lamb_i = list()
            shift = 0
            for j in range(3):  # `j` is an index to choose vector from X
                lamb_i_j = list()
                for k in range(self._solution.dim[j]):  # `k` is an index for vector component
                    lamb_i_jk = self._solution.Lamb[shift:shift + self._solution.deg[j], i].getA1()
                    shift += self._solution.deg[j]
                    lamb_i_j.append(lamb_i_jk)
                lamb_i.append(lamb_i_j)
            self.lamb.append(lamb_i)

    def _print_psi_i_jk(self, i, j, k):
        """
        Returns string of Psi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :param k: an index for vector component
        :return: result string
        """
        strings = list()
        for n in range(len(self.lamb[i][j][k])):
            strings.append('(1 + {symbol}{deg}(x{1}{2}))^({0:.6f})'.format(self.lamb[i][j][k][n], j + 1, k + 1,
                                                                           symbol=self.symbol, deg=n))
        return ' * '.join(strings)

    def _print_phi_i_j(self, i, j):
        """
        Returns string of Phi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :return: result string
        """
        strings = list()
        for k in range(len(self.lamb[i][j])):
            shift = sum(self._solution.dim[:j]) + k
            for n in range(len(self.lamb[i][j][k])):
                strings.append('(1 + {symbol}{deg}(x{1}{2}))^({0:.6f})'.format(self.a[i][shift] * self.lamb[i][j][k][n],
                                                                               j + 1, k + 1, symbol=self.symbol, deg=n))
        return ' * '.join(strings)

    def _print_F_i(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        for j in range(3):
            for k in range(len(self.lamb[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                for n in range(len(self.lamb[i][j][k])):
                    strings.append('(1 + {symbol}{deg}(x{1}{2}))^({0:.6f})'.format(self.c[i][j] * self.a[i][shift] *
                                                                                   self.lamb[i][j][k][n],
                                                                                   j + 1, k + 1, symbol=self.symbol,
                                                                                   deg=n))
        return ' * '.join(strings)

    def _print_F_i_transformed(self, i):
        """
        Returns string of F function in regular polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        power_sum = 0
        for j in range(3):
            for k in range(len(self.lamb[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                power_sum += self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][0]
                for n in range(1, len(self.lamb[i][j][k])):
                    summands = ['{0}(x{1}{2})^{deg}'.format(self.basis[n].coef[index], j + 1, k + 1, deg=index)
                                for index in range(1, len(self.basis[n].coef)) if self.basis[n].coef[index] != 0]
                    if self.basis[n].coef[0] != -1:
                        summands.insert(0, str(1 + self.basis[n].coef[0]))
                    strings.append('({repr})^({0:.6f})'.format(self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][n],
                                                               j + 1, k + 1, repr=' + '.join(summands)))
        strings.insert(0, str((1 + self.basis[0].coef[0]) ** (power_sum)))
        return ' * '.join(strings)

    def _print_F_i_transformed_recovered(self, i):
        """
        Returns string of recovered F function in regular polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        power_sum = 0
        for j in range(3):
            for k in range(len(self.lamb[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                diff = self.maxX[j][k] - self.minX[j][k]
                mult_poly = pnm([- self.minX[j][k] / diff, 1 / diff])
                power_sum += self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][0]
                for n in range(1, len(self.lamb[i][j][k])):
                    res_polynomial = self.basis[n](mult_poly) + 1
                    coeffs = res_polynomial.coef
                    summands = ['{0}(x{1}{2})^{deg}'.format(coeffs[index], j + 1, k + 1, deg=index)
                                for index in range(1, len(coeffs))]
                    summands.insert(0, str(coeffs[0]))
                    strings.append('({repr})^({0:.6f})'.format(self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][n],
                                                               j + 1, k + 1, repr=' + '.join(summands)))
        strings.insert(0, str((self.maxY[i] - self.minY[i]) * (1 + self.basis[0].coef[0]) ** (power_sum)))
        return ' * '.join(strings) + ' + ' + str((2 * self.minY[i] - self.maxY[i]))

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
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)
                       for k in range(self._solution.dim[j])]
        phi_strings = ['Phi^{0}_[{1}]={result} - 1\n'.format(i + 1, j + 1, result=self._print_phi_i_j(i, j))
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)]
        f_strings = ['F^{0} in special basis:\n{result} - 1\n'.format(i + 1, result=self._print_F_i(i))
                     for i in range(self._solution.Y.shape[1])]
        f_strings_transformed = [
            'F^{0} in standard basis:\n{result} - 1\n'.format(i + 1, result=self._print_F_i_transformed(i))
            for i in range(self._solution.Y.shape[1])]
        f_strings_transformed_denormed = ['F^{0} in standard basis '
                                          'denormed:\n{result}\n'.format(i + 1, result=
        self._print_F_i_transformed_recovered(i))
                                          for i in range(self._solution.Y.shape[1])]
        return '\n'.join(psi_strings + phi_strings + f_strings + f_strings_transformed + f_strings_transformed_denormed)

    def plot_in_realtime(self):
        style.use('fivethirtyeight')

        #bort_net
        fig1 = plt.figure()
        ax1_0 = fig1.add_subplot(1, 1, 1)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1, 1, 1)
        def animate_bort_net(i):
            ax1_0.clear()
            ax1_0.plot(self.bort_net[0][0][:i], self.bort_net[0][1][:i])
            ax1_0.plot(self.bort_net[1][0][:i], self.bort_net[1][1][:i])
            ax1_0.plot(self.bort_net[2][0][:i], self.bort_net[2][1][:i])
            ax1_0.plot(self.bort_net[3][0][:i], self.bort_net[3][1][:i])
            ax1_0.plot(self.bort_net[4][0][:i], self.bort_net[4][1][:i])
            # if i >= len(self.x_bort_net):
            #     manager = plt.get_current_fig_manager()
            #     manager.destroy()
        def animate_fuel(i):
            ax2.clear()
            ax2.plot(self.fuel[0][0][:i], self.fuel[0][1][:i])
            ax2.plot(self.fuel[1][0][:i], self.fuel[1][1][:i])
            ax2.plot(self.fuel[2][0][:i], self.fuel[2][1][:i])
            ax2.plot(self.fuel[3][0][:i], self.fuel[3][1][:i])
            ax2.plot(self.fuel[4][0][:i], self.fuel[4][1][:i])
            # if i >= len(self.x_fuel):
            #     manager = plt.get_current_fig_manager()
            #     manager.destroy()
        def animate_battery(i):
            ax3.clear()
            ax3.plot(self.x_battery[:i], self.y_battery[:i])
            # if i >= len(self.x_battery):
            #     manager = plt.get_current_fig_manager()
            #     manager.destroy()
        ani_bort_net = animation.FuncAnimation(fig1, animate_bort_net, interval=1000)
        ani_fuel = animation.FuncAnimation(fig2, animate_fuel, interval=1000)
        ani_battery = animation.FuncAnimation(fig3, animate_battery, interval=1000)
        plt.show()
        return "Автомобіль йобнувся"


    def plot_graphs(self):
        fig, axes = plt.subplots(4, self._solution.Y.shape[1], figsize=(10, 12))
        y_list = [self._solution.Y, self._solution.Y_]
        predict_list = [self._solution.F, self._solution.F_]
        if self._solution.Y.shape[1] > 1:
            for i, j in itertools.product([0, 2], range(self._solution.Y.shape[1])):
                axes[i][j].set_xticks(np.arange(0, self._solution.n + 1, 5))
                X = []
                Y = []
                Y_true = []
                for k, x in enumerate(np.arange(1, self._solution.n + 1)):
                    X.append(x)
                    Y.extend(y_list[i//2][:, j][k].tolist()[0])
                    Y_true.extend(predict_list[i//2][:, j][k].tolist()[0])
                    axes[i][j].plot(X, Y)
                    axes[i][j].plot(X, Y_true)
                    plt.pause(0.05)
                axes[i][j].legend(['True', 'Predict'])
                if i == 0:
                    axes[i][j].set_title('Y{} нормалізовані'.format(j + 1))
                else:
                    axes[i][j].set_title('Y{} ненормалізовані'.format(j + 1))

                axes[i+1][j].set_xticks(np.arange(0, self._solution.n + 1, 5))
                for k in range(len(X)):
                    axes[i+1][j].plot(X[i], abs(Y[k] - Y_true[k]))
                    plt.pause(0.05)
                #axes[i+1][j].plot(np.arange(1, self._solution.n + 1), abs(y_list[i//2][:, j] - predict_list[i//2][:, j]))
                axes[i+1][j].set_title('Похибки: {}'.format(j + 1))
        else:
            for i in [0, 2]:
                axes[i].set_xticks(np.arange(0, self._solution.n + 1, 5))
                axes[i].plot(np.arange(1, self._solution.n + 1), y_list[i // 2])
                axes[i].plot(np.arange(1, self._solution.n + 1), predict_list[i // 2])
                axes[i].legend(['True', 'Predict'])
                if i == 0:
                    axes[i].set_title('Y нормалізовані')
                else:
                    axes[i].set_title('Y ненормалізовані')

                axes[i + 1].set_xticks(np.arange(0, self._solution.n + 1, 5))
                axes[i + 1].plot(np.arange(1, self._solution.n + 1),
                                    abs(y_list[i // 2] - predict_list[i // 2]))
                axes[i + 1].set_title('Похибки')
        plt.savefig('graphics/graph_{}_{}_{}_{}_{}_{}_{}'.format(self._solution.deg, self._solution.poly_type, self._solution.dim,
                                                        self._solution.splitted_lambdas, self._solution.custom_func_struct,
                                                              self._solution.solving_method, self._solution.weights))
        manager = plt.get_current_fig_manager()
        manager.set_window_title('Graph')
        if os_name == 'posix':
            fig.show()
        else:
            plt.show()

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
        XF, YF = self._solution.build_predicted(steps)
        for i, x in enumerate(self._solution.X_):
            for j, xc in enumerate(x.T):
                self.compare_vals('X{}{}'.format(i + 1, j + 1), xc.getA1(), XF[i][j])
        for i in range(self._solution.dim[3]):
            self.compare_vals('Y{}'.format(i + 1), self._solution.Y_[:, i].getA1(), YF[:, i],
                              self._solution.F_[:, i].getA1())


class PolynomialBuilderExpTh(PolynomialBuilder):
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
            strings.append('exp({0:.6f}*tanh({inner}))'.format(self.a[i][sum(self._solution.dim[:j]) + k],
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


class PolynomialBuilderSigmoid(PolynomialBuilder):
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
            strings.append('exp({0:.6f}*1/(1+exp({inner})))'.format(self.lamb[i][j][k][n], inner=inner))
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
            strings.append('exp({0:.6f}*1/(1+exp({inner})))'.format(self.a[i][sum(self._solution.dim[:j]) + k],
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
            strings.append('exp({0:.6f}*1/(1+exp({inner})))'.format(self.c[i][j], inner=self._print_phi_i_j(i, j, mode)))
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
        f_strings_transformed = [
            'F^{0} in standard basis:\n{result}\n'.format(i + 1, result=self._print_F_i(i, mode=1))
            for i in range(self._solution.Y.shape[1])]
        f_strings_transformed_denormed = ['F^{0} in standard basis '
                                          'denormed:\n{result}\n'.format(i + 1, result=self._print_F_i(i, mode=2))
                                          for i in range(self._solution.Y.shape[1])]
        return '\n'.join(
            psi_strings + phi_strings + f_strings + f_strings_transformed + f_strings_transformed_denormed)


class PolynomialBuilderSin(PolynomialBuilder):
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
            strings.append('exp({0:.6f}*sin({inner}))'.format(self.lamb[i][j][k][n], inner=inner))
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
            strings.append('exp({0:.6f}*sin({inner}))'.format(self.a[i][sum(self._solution.dim[:j]) + k],
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
            strings.append('exp({0:.6f}*sin({inner}))'.format(self.c[i][j], inner=self._print_phi_i_j(i, j, mode)))
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
        f_strings_transformed = [
            'F^{0} in standard basis:\n{result}\n'.format(i + 1, result=self._print_F_i(i, mode=1))
            for i in range(self._solution.Y.shape[1])]
        f_strings_transformed_denormed = ['F^{0} in standard basis '
                                          'denormed:\n{result}\n'.format(i + 1, result=self._print_F_i(i, mode=2))
                                          for i in range(self._solution.Y.shape[1])]
        return '\n'.join(
            psi_strings + phi_strings + f_strings + f_strings_transformed + f_strings_transformed_denormed)


class PolynomialBuilderCos(PolynomialBuilder):
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
            strings.append('exp({0:.6f}*cos({inner}))'.format(self.lamb[i][j][k][n], inner=inner))
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
            strings.append('exp({0:.6f}*cos({inner}))'.format(self.a[i][sum(self._solution.dim[:j]) + k],
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
            strings.append('exp({0:.6f}*cos({inner}))'.format(self.c[i][j], inner=self._print_phi_i_j(i, j, mode)))
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
        f_strings_transformed = [
            'F^{0} in standard basis:\n{result}\n'.format(i + 1, result=self._print_F_i(i, mode=1))
            for i in range(self._solution.Y.shape[1])]
        f_strings_transformed_denormed = ['F^{0} in standard basis '
                                          'denormed:\n{result}\n'.format(i + 1, result=self._print_F_i(i, mode=2))
                                          for i in range(self._solution.Y.shape[1])]
        return '\n'.join(
            psi_strings + phi_strings + f_strings + f_strings_transformed + f_strings_transformed_denormed)


class PolynomialBuilderArctg(PolynomialBuilder):
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
            strings.append('exp({0:.6f}*arctang({inner}))'.format(self.lamb[i][j][k][n], inner=inner))
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
            strings.append('exp({0:.6f}*arctang({inner}))'.format(self.a[i][sum(self._solution.dim[:j]) + k],
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
            strings.append('exp({0:.6f}*arctang({inner}))'.format(self.c[i][j], inner=self._print_phi_i_j(i, j, mode)))
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
        f_strings_transformed = [
            'F^{0} in standard basis:\n{result}\n'.format(i + 1, result=self._print_F_i(i, mode=1))
            for i in range(self._solution.Y.shape[1])]
        f_strings_transformed_denormed = ['F^{0} in standard basis '
                                          'denormed:\n{result}\n'.format(i + 1, result=self._print_F_i(i, mode=2))
                                          for i in range(self._solution.Y.shape[1])]
        return '\n'.join(
            psi_strings + phi_strings + f_strings + f_strings_transformed + f_strings_transformed_denormed)
