import pandas as pd
from basis_generators import BasisGenerator
from functools import reduce
from optimization_methods import *
import itertools
import numpy as np
from custom_exceptions import *
from scipy.sparse.linalg import cg
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from additional_functions import functions
from arima import forecast


class Solve(object):

    def __init__(self, d):
        self.n = d['samples']
        self.deg = d['dimensions']
        self.filename_input = d['input_file']
        self.filename_output = d['output_file']
        self.p = d['degrees']
        self.weights = d['weights']
        self.poly_type = d['poly_type']
        self.splitted_lambdas = d['lambda_multiblock']
        self.solving_method = d['method']
        self.eps = 1E-6
        self.norm_error = 0.0
        self.error = 0.0

    offset = 1e-10

    def _prepare_data(self):
        new_cols = list(
            itertools.chain(*[['X{}'.format(i + 1)] * self.deg[i] if i != len(self.deg) - 1 else ['Y'] * self.deg[i]
                              for i in range(len(self.deg))]))
        new_cols = np.unique([[el + str(i) for i in range(1, new_cols.count(el) + 1)] for el in new_cols])
        if len(new_cols.shape) > 1:
            new_cols = list(itertools.chain(*new_cols))
        dt = pd.read_csv(self.filename_input, sep='\t', header=None).astype(float)
        try:
            dt.columns = new_cols
        except:
            dt = dt.iloc[:, :-2]
            dt.columns = new_cols
        return dt

    def _minimize_equation(self, A, b):
        """
        Finds such vector x that |Ax-b|->min.
        :param A: Matrix A
        :param b: Vector b
        :return: Vector x
        """
        if self.solving_method == 'LSTM':
            return np.linalg.lstsq(A, b)[0]
        elif self.solving_method == 'conjucate':
            return cg(np.dot(A, A.T), b, tol=self.eps)[0] #conjugate_gradient_method(A, b, self.eps)
        elif self.solving_method == 'coordDesc':
            return Ridge(alpha=self.eps).fit(A, b).coef_ #coordinate_descent(A, b, self.eps) #Ridge().fit(A, b).coef_
        else:
            raise MethodNotFilledException

    def _norm_data(self, data):
        normalized_data = data.copy(deep=True)
        agg = normalized_data.agg([min, max])

        for col in normalized_data.columns:
            min_val = agg.loc['min', col]
            max_val = agg.loc['max', col]
            normalized_data[col] = normalized_data[col].apply(lambda x: (x - min_val) / (max_val - min_val))

        return normalized_data

    def _create_train_dataset(self, data):
        '''
        build matrix X and Y
        :return:
        '''
        X = data.loc[:, [el for el in data.columns if el.find('X') != -1]]
        Y = data.loc[:, [el for el in data.columns if el.find('Y') != -1]]

        return X, Y

    def _get_B(self, Y):
        if self.weights == 'average':
            Y_res = (Y.max(axis=1) + Y.min(axis=1)) / 2  # arguable, may be need not to normalize Y before this operation
            try:
                Y_ = np.tile(Y_res.values, (1, self.deg[-1])).reshape((self.n, self.deg[-1]))
                Y_log = np.log(Y_ + self.offset + 1)
                return Y_, Y_log
            except:
                return Y, np.log(Y + self.offset + 1)
        elif self.weights == 'width_interval':
            Y_res = (Y.max(axis=1) - Y.min(axis=1)).values
            try:
                Y_ = np.tile(Y_res, (1, self.deg[-1])).reshape((self.n, self.deg[-1]))
                Y_log = np.log(Y_ + self.offset + 1)
                return Y_, Y_log
            except:
                return Y, np.log(Y + self.offset + 1)
        elif self.weights == 'scaled':
            return Y, np.log(Y + self.offset + 1)

    def _evaluate_polynom(self, coefs, x):
        return sum([np.array(coef) * pow(x, i) for i, coef in enumerate(coefs)])

    def _get_A(self, data, polynoms_degrees):
        A = pd.DataFrame()
        for i, degree in enumerate(polynoms_degrees):
            if self.poly_type in functions.keys():
                func = functions.get(self.poly_type)
                A = pd.concat([A, data.apply(lambda x: func(degree, x))], axis=1)
            else:
                gen = BasisGenerator(degree, self.poly_type)
                coefs = list(map(lambda x: x, list(gen.generate())[-1]))
                A = pd.concat([A, data.apply(lambda x: self._evaluate_polynom(coefs, x))], axis=1)
        A_log = 1 + A + self.offset
        return A, A_log.apply(lambda x: [np.log(abs(el)) if isinstance(el, float) else 0 for el in x])

    def _get_lambdas(self, A, Y):
        lambdas = pd.DataFrame(columns=['lambda_{}'.format(i+1) for i in range(self.deg[-1])])
        A = pd.DataFrame(A)
        Y = pd.DataFrame(Y)
        for i, j in itertools.product(range(self.deg[-1]), range(self.deg[-1])):
            if self.splitted_lambdas:
                use_cols = [el for el in A.columns if el.find('X{}'.format(j + 1)) != -1]
                train_data = A.loc[:, use_cols]
                a = train_data.T * Y.loc[:, Y.columns[i]]
                lambdas.loc[i, lambdas.columns[j]] = [self._minimize_equation(a.T.values, Y.loc[:, Y.columns[i]])]
            else:
                a = A.T * Y.loc[:, Y.columns[i]].fillna(A.T.mean().mean())
                lambdas.loc[i, lambdas.columns[j]] = [self._minimize_equation(a.fillna(a.mean().mean()).T.apply(lambda x:
                                                        [el.coef[0] if not isinstance(el, float) else el for el in x]),
                                                                              Y.fillna(Y.mean()).loc[:, Y.columns[i]])]
        return lambdas

    def _get_psi(self, A, lambdas):
        if self.solving_method == 'LSTM':
            if self.splitted_lambdas:
                psi = [[A.loc[:, [el for el in A.columns if el.find('X{}'.format(i + 1)) != -1]
                        ] * lambdas.loc[j, 'lambda_{}'.format(i + 1)][0] for i in range(self.deg[-1])] for j in
                       range(self.deg[-1])]
            else:
                psi = [[A * lambdas.loc[j, 'lambda_{}'.format(i + 1)][0] for i in range(self.deg[-1])] for j in
                       range(self.deg[-1])]
        elif self.solving_method == 'conjucate':
            if self.splitted_lambdas:
                psi = [[(A.T.loc[[el for el in A.columns if el.find('X{}'.format(i + 1)) != -1],:
                        ] * lambdas.loc[j, 'lambda_{}'.format(i + 1)][0]).T for i in range(self.deg[-1])] for j in
                       range(self.deg[-1])]
            else:
                psi = [[(A.T * lambdas.loc[j, 'lambda_{}'.format(i + 1)][0]).T for i in range(self.deg[-1])] for j in
                       range(self.deg[-1])]
        else:
            if self.splitted_lambdas:
                psi = [[A.loc[:, [el for el in A.columns if el.find('X{}'.format(i + 1)) != -1]
                        ] * lambdas.loc[j, 'lambda_{}'.format(i + 1)][0] for i in range(self.deg[-1])] for j in
                       range(self.deg[-1])]
            else:
                psi = [[A * lambdas.loc[j, 'lambda_{}'.format(i + 1)][0] for i in range(self.deg[-1])] for j in
                       range(self.deg[-1])]
        psi_log = psi
        psi = [[el1.apply(lambda x: [np.exp(s) if isinstance(s, float) else np.exp(s.coef[0]) for s in x])
                for el1 in el] for el in psi_log]
        return psi, psi_log

    def _get_A1(self, psi, y):
        y = pd.DataFrame(y)
        return [[self._minimize_equation(psi[i][j][:].fillna(0), y.loc[:, y.columns[i]].apply(lambda x: abs(x) if x>0 else abs(-x)) + 1 + self.offset)
                 for j in range(self.deg[-1])] for i in range(self.deg[-1])]

    def _get_Fi(self, psi, a1):
        if self.solving_method == 'LSTM':
            if self.splitted_lambdas:
                fi = np.array([[psi[i][j] * a1[i][j] for j in range(self.deg[-1])] for i in range(self.deg[-1])])
            else:
                fi = [[psi[i][j] * a1[i][j] for j in range(self.deg[-1])] for i in range(self.deg[-1])]
        elif self.solving_method == 'conjucate':
            if self.splitted_lambdas:
                fi = np.array([[(psi[i][j].T * a1[i][j]).T for j in range(self.deg[-1])] for i in range(self.deg[-1])])
            else:
                fi = [[(psi[i][j].T * a1[i][j]).T for j in range(self.deg[-1])] for i in range(self.deg[-1])]
        else:
            if self.splitted_lambdas:
                fi = np.array([[psi[i][j] * a1[i][j] for j in range(self.deg[-1])] for i in range(self.deg[-1])])
            else:
                fi = [[psi[i][j] * a1[i][j] for j in range(self.deg[-1])] for i in range(self.deg[-1])]
        fi = [reduce(lambda x, y: pd.concat([x, y], axis=1), fi[i]) for i in range(self.deg[-1])]
        fi_log = fi
        fi = [el.apply(lambda x: np.exp(x) - 1 - self.offset) for el in fi_log]
        return fi, fi_log

    def _get_coefs(self, fi, y):
        y = pd.DataFrame(y)
        if self.solving_method == 'conjucate':
            return [self._minimize_equation(np.dot(fi[i].fillna(0).T, fi[i].fillna(0)),
                                        np.dot(fi[i].fillna(0).T, np.log(1 + y.iloc[:, i] + self.offset)))
                    for i in range(self.deg[-1])]
        else:
            return [self._minimize_equation(fi[i].fillna(0).replace([-np.inf, np.inf], [0, 0]),
                    y.iloc[:, i].apply(lambda x: np.log(x) if x > 0 else np.log(-x)).replace([-np.inf, np.inf], [0, 0]))
                    for i in range(self.deg[-1])]

    # TODO Fitness function for normalize version
    def _get_fitness_function(self, fi, y, coefs):
        y = pd.DataFrame(y)
        fitness = [np.dot(fi[i].replace([-np.inf, np.inf], [0, 0]), coefs[i]) for i in range(self.deg[-1])]
        fitness_log = fitness
        fitness = np.exp(fitness_log) - 1
        norm_error = [(y.iloc[:, i] - fitness[i]) for i in range(self.deg[-1])]
        return fitness, fitness_log, norm_error

    def _aggregate(self, values, coeffs):
        return np.exp(np.dot(np.log(1 + values + self.offset), coeffs)) - 1

    def _calculate_polynoms(self, x, degree, poly_type):
        res = []
        for deg in range(degree):
            coefs = list(BasisGenerator(deg, poly_type).generate())
            if coefs:
                coefs = coefs[-1].coef
            else:
                coefs = [1]
            res.append(self._evaluate_polynom(coefs, x))
        return res

    def _calculate_value(self, X, y, lambdas, coefs, A):
        use_cols = [['X{}{}'.format(i + 1, j + 1) for j in
                     range(len([el for el in X.columns if el.find('X{}'.format(i + 1)) != -1]))]
                    for i in range(len(self.deg))][:-1]

        minX = [X[el].min(axis=0).min() for el in use_cols]
        maxX = [X[el].max(axis=0).max() for el in use_cols]
        minY = y.min(axis=0).min()
        maxY = y.max(axis=0).max()

        X_normalized = (X - min(minX)) / (max(maxX) - min(minX))
        X_splitted = np.split(X_normalized, len(self.p))
        phi = [self._calculate_polynoms(vector, self.p[i], self.poly_type) for i, vector in enumerate(X_splitted)]
        psi = list()
        for i in range(len(self.p)):  # self.p:
            for j in range(self.deg[i]):
                psi.append(self._aggregate(phi[i][j], lambdas.iloc[i, i][0][:phi[i][j].shape[1]]))

        psi = pd.DataFrame(psi).fillna(np.random.uniform(pd.DataFrame(psi).dropna().min(axis=1).min(),
                                                         pd.DataFrame(psi).dropna().max(axis=1).max()))
        big_phi = list()
        for i in range(len(self.p)):
            use_cols = [el for el in X.columns if el.find('X{}'.format(i + 1)) != -1]
            big_phi.append([self._aggregate(psi.iloc[i, k], A[use_cols])
                            for k in range(self.deg[-1])])
        # big_phi = np.array(big_phi).T
        big_phi = pd.DataFrame(big_phi)
        result = np.array([self._aggregate(big_phi.loc[:, k].tolist()[0].T,
                                     coefs.iloc[k, :big_phi.loc[:, k].tolist()[0].shape[0]])
                           for k in range(self.deg[-1])])
        result = result * ((maxY) - (minY)) + (minY)
        return pd.DataFrame(result).fillna(np.random.uniform(pd.DataFrame(result).min(axis=1).min(),
                                                             pd.DataFrame(result).min(axis=1).min() +
                                                             np.random.normal(0, 1, size=1)[0]))

    def _build_predicted(self, X, Y, lambdas, coefs, A, steps, use_cols):

        XF = X.loc[:, use_cols].apply(lambda x: forecast(x, steps), axis=0)
        xf = [X.loc[-s, use_cols] for s in range(1, steps + 1)]

        yf = list(map(lambda x: self._calculate_value(x, Y, lambdas, coefs, A), xf))
        YF = Y.copy(deep=True)
        YF[-steps:] = np.array(yf)
        return XF, YF

    def _save_data(self, data, norm_data, A, norm_A, lambdas, lambdas_norm, psi, psi_norm,
                   A1, A1_norm, y_new, y_new_normalized, c, c_norm,
                   fit_res, fit_res_norm, errors, errors_norm, nor_errors, nor_errors_norm):
        with pd.ExcelWriter(self.filename_output) as writer:
            data.to_excel(writer, sheet_name='Вхідні дані')
            A.to_excel(writer, sheet_name='Матриця А')
            lambdas.to_excel(writer, sheet_name='Значення лямбд')
            for i in range(len(psi)):
                temp = reduce(lambda x, y: pd.concat([x, y], axis=1), psi[i])
                pd.DataFrame(temp).to_excel(writer, sheet_name='PSI{}'.format(i + 1))
            A1.to_excel(writer, sheet_name='матриця А1')
            y_new.to_excel(writer, sheet_name='Перебудовані Y')
            c.to_excel(writer, sheet_name='Коефіцієнти c')
            fit_res.to_excel(writer, sheet_name='Побудований прогноз')
            errors.to_excel(writer, sheet_name='Похибки')
            nor_errors.to_excel(writer, sheet_name='Норми похибок')

            norm_data.to_excel(writer, sheet_name='Вхідні дані (нормалізований варіант)')
            norm_A.to_excel(writer, sheet_name='Матриця А (нормалізований варіант)')
            lambdas_norm.to_excel(writer, sheet_name='Значення лямбд (нормалізований варіант)')
            for i in range(len(psi_norm)):
                temp = reduce(lambda x, y: pd.concat([x, y], axis=1), psi_norm[i])
                temp.to_excel(writer, sheet_name='PSI{} (нормалізований варіант)'.format(i + 1))
            A1_norm.to_excel(writer, sheet_name='матриця А1 (нормалізований варіант)')
            y_new_normalized.to_excel(writer, sheet_name='Перебудовані Y (нормалізований варіант)')
            c_norm.to_excel(writer, sheet_name='Коефіцієнти c (нормалізований варіант)')
            fit_res_norm.to_excel(writer, sheet_name='Побудований прогноз (нормалізований варіант)')
            errors_norm.to_excel(writer, sheet_name='Похибки (нормалізований варіант)')
            nor_errors_norm.to_excel(writer, sheet_name='Норми похибок (нормалізований варіант)')

    def print_data(self, data, norm_data, A, norm_A, lambdas, lambdas_norm, psi, psi_norm,
                   A1, A1_norm, y_new, y_new_normalized, c, c_norm,
                   fit_res, fit_res_norm, errors, errors_norm, nor_errors, nor_errors_norm, *args):
        text = []
        text.append('Вхідні дані')
        print('Вхідні дані')
        print(data.to_string())
        text.append(data.to_string())
        print('-------------------------')
        text.append('-------------------------')
        print('Матриця А')
        text.append('Матриця А')
        print(A.to_string())
        text.append(A.to_string())
        print('-------------------------')
        text.append('-------------------------')
        print('Значення лямбд')
        text.append('Значення лямбд')
        print(lambdas.to_string())
        text.append(lambdas.to_string())
        print('-------------------------')
        text.append('-------------------------')
        for i in range(len(psi)):
            temp = reduce(lambda x, y: pd.concat([x, y], axis=1), psi[i])
            print('PSI{}'.format(i + 1))
            text.append('PSI{}'.format(i + 1))
            print(temp.to_string())
            text.append(temp.to_string())
            print('---------------------')
            text.append('---------------------')
        print('матриця А1')
        text.append('матриця А1')
        print(A1.to_string())
        text.append(A1.to_string())
        print('-------------------------')
        text.append('-------------------------')
        print('Перебудовані Y')
        text.append('Перебудовані Y')
        print(y_new.to_string())
        text.append(y_new.to_string())
        print('-------------------------')
        text.append('-------------------------')
        print('Коефіцієнти c')
        text.append('Коефіцієнти c')
        print(c.to_string())
        text.append(c.to_string())
        print('-------------------------')
        text.append('-------------------------')
        print('Побудований прогноз')
        text.append(fit_res.to_string())
        print(fit_res.to_string())
        text.append('-------------------------')
        print('-------------------------')
        text.append('Похибки')
        print('Похибки')
        text.append(errors.to_string())
        print(errors.to_string())
        text.append('-------------------------')
        print('-------------------------')
        text.append('Норми похибок')
        print('Норми похибок')
        text.append(nor_errors.to_string())
        print(nor_errors.to_string())
        text.append('-------------------------')
        print('-------------------------')
        text.append('\n\n')
        print('\n\n')
        text.append('Нормалізований варіант')
        print('Нормалізований варіант')
        text.append('-------------------------')
        print('-------------------------')
        text.append('Вхідні дані')
        print('Вхідні дані')
        text.append(norm_data.to_string())
        print(norm_data.to_string())
        text.append('-------------------------')
        print('-------------------------')
        text.append('Матриця А')
        print('Матриця А')
        text.append(norm_A.to_string())
        print(norm_A.to_string())
        text.append('-------------------------')
        print('-------------------------')
        text.append('Значення лямбд')
        print('Значення лямбд')
        text.append(lambdas_norm.to_string())
        print(lambdas_norm.to_string())
        text.append('-------------------------')
        print('-------------------------')
        for i in range(len(psi_norm)):
            temp = reduce(lambda x, y: pd.concat([x, y], axis=1), psi_norm[i])
            print('PSI{}'.format(i + 1))
            text.append('PSI{}'.format(i + 1))
            print(temp.to_string())
            text.append(temp.to_string())
            print('---------------------')
            text.append('---------------------')
        print('матриця А1')
        text.append('матриця А1')
        print(A1_norm.to_string())
        text.append(A1_norm.to_string())
        print('-------------------------')
        text.append('-------------------------')
        print('Перебудовані Y')
        text.append('Перебудовані Y')
        print(y_new_normalized.to_string())
        text.append(y_new_normalized.to_string())
        print('-------------------------')
        text.append('-------------------------')
        print('Коефіцієнти c')
        text.append('Коефіцієнти c')
        print(c_norm.to_string())
        text.append(c_norm.to_string())
        print('-------------------------')
        text.append('-------------------------')
        text.append('Побудований прогноз')
        print('Побудований прогноз')
        text.append(fit_res_norm.to_string())
        print(fit_res_norm.to_string())
        text.append('-------------------------')
        print('-------------------------')
        text.append('Похибки')
        print('Похибки')
        text.append(errors_norm.to_string())
        print(errors_norm.to_string())
        text.append('-------------------------')
        print('-------------------------')
        text.append('Норми похибок')
        print('Норми похибок')
        text.append(nor_errors_norm.to_string())
        print(nor_errors_norm.to_string())

        return ('\n').join(text)

    def main(self, print_=False):
        prepared_data = self._prepare_data()
        normalized_data = self._norm_data(prepared_data)
        train_data_normalized, target_normalized = self._create_train_dataset(normalized_data)
        train_data, target = self._create_train_dataset(prepared_data)
        A, A_log = self._get_A(train_data, self.p)
        b, b_log = self._get_B(target)
        lambdas = self._get_lambdas(A_log, b_log)
        psi, psi_log = self._get_psi(A_log, lambdas)
        A1 = self._get_A1(psi_log, b)
        Fi, Fi_log = self._get_Fi(psi_log, A1)
        coefs = self._get_coefs(Fi, b)
        fitnes_result, fitnes_result_log, error = self._get_fitness_function(Fi, b, coefs)

        A_normalized, A_normalized_log = self._get_A(train_data_normalized, self.p)
        b_normalized, b_normalized_log = self._get_B(target_normalized)
        lambdas_normalized = self._get_lambdas(A_normalized, b_normalized)
        psi_normalized, psi_normalized_log = self._get_psi(A_normalized, lambdas_normalized)
        A1_normalized = self._get_A1(psi_normalized, b_normalized)
        Fi_normalized, Fi_normalized_log = self._get_Fi(psi_normalized, A1_normalized)
        coefs_normalized = self._get_coefs(Fi_normalized, b_normalized)
        fitnes_result_normalized, fitnes_result_normalized_log, error_normalized = self._get_fitness_function(Fi_normalized, b_normalized,
                                                                                coefs_normalized)
        if self.solving_method == 'coordDesc':
            fitnes_result_normalized = [(el - min(el)) / (max(el) - min(el)) for el in fitnes_result_normalized]
            error_normalized = [(el - min(el)) / (max(el) - min(el)) for el in error_normalized]

        self._save_data(prepared_data, normalized_data, pd.DataFrame(A), pd.DataFrame(A_normalized), lambdas, lambdas_normalized,
                        psi, psi_normalized, pd.DataFrame(A1),
                        pd.DataFrame(A1_normalized), pd.DataFrame(b), pd.DataFrame(b_normalized),
                        pd.DataFrame(coefs), pd.DataFrame(coefs_normalized), pd.DataFrame(fitnes_result).T,
                        pd.DataFrame(fitnes_result_normalized).T, pd.DataFrame(error).T,
                        pd.DataFrame(error_normalized).T,
                        pd.DataFrame(pd.DataFrame(error).T.apply(lambda x: np.linalg.norm(x))).T,
                        pd.DataFrame(pd.DataFrame(error_normalized).T.apply(lambda x: np.linalg.norm(x))).T)

        if print_:
            self.print_data(prepared_data, normalized_data, pd.DataFrame(A), pd.DataFrame(A_normalized), lambdas, lambdas_normalized,
                            psi, psi_normalized, pd.DataFrame(A1),
                            pd.DataFrame(A1_normalized), pd.DataFrame(b), pd.DataFrame(b_normalized),
                            pd.DataFrame(coefs), pd.DataFrame(coefs_normalized), pd.DataFrame(fitnes_result).T,
                            pd.DataFrame(fitnes_result_normalized).T, pd.DataFrame(error).T,
                            pd.DataFrame(error_normalized).T,
                            pd.DataFrame(pd.DataFrame(error).T.apply(lambda x: np.linalg.norm(x))).T,
                            pd.DataFrame(pd.DataFrame(error_normalized).T.apply(lambda x: np.linalg.norm(x))).T)

        return [prepared_data, normalized_data, pd.DataFrame(A), pd.DataFrame(A_normalized), lambdas, lambdas_normalized,
                psi, psi_normalized, pd.DataFrame(A1),
                pd.DataFrame(A1_normalized), pd.DataFrame(b), pd.DataFrame(b_normalized),
                pd.DataFrame(coefs), pd.DataFrame(coefs_normalized), pd.DataFrame(fitnes_result).T,
                pd.DataFrame(fitnes_result_normalized).T, pd.DataFrame(error).T,
                pd.DataFrame(error_normalized).T,
                pd.DataFrame(pd.DataFrame(error).T.apply(lambda x: np.linalg.norm(x))).T,
                pd.DataFrame(pd.DataFrame(error_normalized).T.apply(lambda x: np.linalg.norm(x))).T, self.deg, self.p]