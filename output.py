import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from numpy.polynomial import Polynomial as pnm
from os import name as os_name

from solve import Solve
import basis_gen as b_gen
import itertools
from PyQt5.QtWidgets import QTableWidgetItem

from tabulate import tabulate as tb

class PolynomialBuilder(object):
    def __init__(self, solution):
        # assert isinstance(solution, Solve)
        # self._solution = solution
        # max_degree = max(solution.deg) - 1
        # if solution.poly_type == 'combined_cheb':
        #     self.symbol = 'CC'
        # elif solution.poly_type == 'laguerre':
        #     self.symbol = 'L'
        #     self.basis = b_gen.basis_laguerre(max_degree)
        # elif solution.poly_type == 'sh_cheb_2':
        #     self.symbol = 'U'
        #     self.basis = b_gen.basis_sh_chebyshev_2_shrinked(max_degree)
        # assert self.symbol
        # self.a = solution.a.T.tolist()
        # self.c = solution.c.T.tolist()
        # self.minX = [X.min(axis=0).getA1() for X in solution.X_]
        # self.maxX = [X.max(axis=0).getA1() for X in solution.X_]
        # self.minY = solution.Y_.min(axis=0).getA1()
        # self.maxY = solution.Y_.max(axis=0).getA1()
        self.resultsField = solution.resultsField
        self.bort_net = self.parse("Data/2,2,2,T(x),norm,10,Reanim/Graphics0.txt")
        self.fuel = self.parse("Data/2,2,2,T(x),norm,10,Reanim/Graphics1.txt")
        self.battery = self.parse("Data/2,2,2,T(x),norm,10,Reanim/Graphics2.txt")
        self.table = self.parsetable("Data/2,2,2,T(x),norm,10,Reanim/Table.txt")
        self.current_time = 0
        self.ani_bort_net = None
        self.ani_fuel = None
        self.ani_battery = None

    def parsetable(self, filename):
        file = open(filename, 'r', encoding='utf-16')
        data = file.readlines()
        table = [line.split("\t") for line in data]
        return table

    def parse_row(self, row):
        a = np.array(list(map(lambda x: np.array([el.replace(' ', '').replace(',', '.') for el
                                                  in x.replace('(', '').replace(')', '').strip().split(', ')]),
                              row.split('\t'))))

        x = [float(el[0]) for el in a if el[0]]
        y = [float(el[1]) for el in a if len(el) > 1 and el[1]]

        return [x, y]

    def parse(self, filename):
        file = open(filename, 'r', encoding='utf-16')
        data = file.readlines()
        data = [el for el in data if el.find('iteration') == -1]

        res_data = list(map(lambda x: self.parse_row(x), data))
        all_dt = [res_data[i: i + 6] for i in range(0, len(res_data), 6)]

        # for iter, dt in enumerate(all_dt):
        #     for d in dt[0:4]:
        #         d[0] = [i + iter * 20 for i in d[0]]

        return [el[:-1] + [el[-1][0]] for el in all_dt]

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

    def pause(self):
        if self.ani_battery:
            self.ani_battery.event_source.stop()
            self.ani_bort_net.event_source.stop()
            self.ani_fuel.event_source.stop()

    def resume(self):
        if self.ani_battery:
            self.ani_battery.event_source.start()
            self.ani_bort_net.event_source.start()
            self.ani_fuel.event_source.start()

    def plot_in_realtime(self):
        #style.use('fivethirtyeight')

        #bort_net
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1)

        #fuel
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1, 1, 1)
        self.resultsField.setRowCount(len(self.table))
        self.resultsField.setColumnCount(8)
        self.resultsField.setHorizontalHeaderLabels(
            ['Час', 'Напрога бортової мережі', 'Пальне', 'Напруга акумулятора', 'Cтан', 'Ризик аварії', 'Причина нештатної ситуації'
            , 'Рівень небезпеки', 'Ресурс доп. ризику']
        )
        def animate_bort_net(i):
            ax1.clear()
            ax1.set_ylim([self.bort_net[i][5][2],self.bort_net[i][5][3]])
            ax1.plot(self.bort_net[i][0][0] , self.bort_net[i][0][1] )
            ax1.plot(self.bort_net[i][1][0] , self.bort_net[i][1][1] )
            ax1.plot(self.bort_net[i][2][0] , self.bort_net[i][2][1] )
            ax1.plot(self.bort_net[i][3][0] , self.bort_net[i][3][1] )
            ax1.plot(self.bort_net[i][4][0] , self.bort_net[i][4][1])
            time = QTableWidgetItem()
            time.setText(self.table[i][0])
            bort_net_val = QTableWidgetItem()
            bort_net_val.setText(self.table[i][1])
            fuel_val = QTableWidgetItem()
            fuel_val.setText(self.table[i][2])
            battery_val = QTableWidgetItem()
            battery_val.setText(self.table[i][3])
            state = QTableWidgetItem()
            state.setText(self.table[i][4])
            state = QTableWidgetItem()
            state.setText(self.table[i][4])
            state = QTableWidgetItem()
            state.setText(self.table[i][4])
            risk = QTableWidgetItem()
            risk.setText(self.table[i][5])
            cause = QTableWidgetItem()
            cause.setText(self.table[i][6])
            level = QTableWidgetItem()
            level.setText(self.table[i][7])
            resource = QTableWidgetItem()
            resource.setText(self.table[i][8])

            self.resultsField.setItem(i, 0, time)
            self.resultsField.setItem(i, 1, bort_net_val)
            self.resultsField.setItem(i, 2, fuel_val)
            self.resultsField.setItem(i, 3, battery_val)
            self.resultsField.setItem(i, 4, state)
            self.resultsField.setItem(i, 5, risk)
            self.resultsField.setItem(i, 6, cause)
            self.resultsField.setItem(i, 7, level)
            self.resultsField.setItem(i, 8, resource)
            self.resultsField.resizeColumnsToContents()

            # if i >= len(self.x_bort_net):
            #     manager = plt.get_current_fig_manager()
            #     manager.destroy()
        def animate_fuel(i):
            ax2.clear()
            ax2.set_ylim([self.fuel[i][5][2],self.fuel[i][5][3]])
            ax2.plot(self.fuel[i][0][0] , self.fuel[i][0][1] )
            ax2.plot(self.fuel[i][1][0] , self.fuel[i][1][1] )
            ax2.plot(self.fuel[i][2][0] , self.fuel[i][2][1] )
            ax2.plot(self.fuel[i][3][0] , self.fuel[i][3][1] )
            ax2.plot(self.fuel[i][4][0] , self.fuel[i][4][1] )
            # if i >= len(self.x_fuel):
            #     manager = plt.get_current_fig_manager()
            #     manager.destroy()
        def animate_battery(i):
            ax3.clear()
            ax3.set_ylim([self.battery[i][5][2],self.battery[i][5][3]])
            ax3.plot(self.battery[i][0][0], self.battery[i][0][1])
            ax3.plot(self.battery[i][1][0], self.battery[i][1][1])
            ax3.plot(self.battery[i][2][0], self.battery[i][2][1])
            ax3.plot(self.battery[i][3][0], self.battery[i][3][1])
            ax3.plot(self.battery[i][4][0], self.battery[i][4][1])
            # if i >= len(self.x_battery):
            #     manager = plt.get_current_fig_manager()
            #     manager.destroy()
        self.ani_bort_net = animation.FuncAnimation(fig1, animate_bort_net, interval=200)
        self.ani_fuel = animation.FuncAnimation(fig2, animate_fuel, interval=200)
        self.ani_battery = animation.FuncAnimation(fig3, animate_battery, interval=200)
        plt.show()
        return "The end"


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
