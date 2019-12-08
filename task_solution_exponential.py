from task_solution import Solve
import numpy as np
import pandas as pd


class ExponentialSolve(Solve):

    def __init__(self, d):
        super().__init__(d)

    def _get_A_exp(self, data, polynoms_degrees):
        A, A_log_prev = self._get_A(data, polynoms_degrees)
        A_log = np.tanh(A)
        A_res = np.exp(A_log)
        return A_log, A_res

    def _get_psi_exp(self, A, lambdas):
        psi, psi_log = self._get_psi(A, lambdas)
        psi_res = [np.exp(el) - 1 for el in psi]
        psi_tanh = np.array([np.tanh(el) for el in psi_res])
        return psi_res, psi_tanh

    def _get_Fi_exp(self, psi_tanh, a1):
        Fi = self._get_Fi(psi_tanh, a1)
        Fi_res = [np.exp(el) - 1 - self.offset for el in Fi]
        Fi_tanh = [np.tanh(el) for el in Fi_res]

        return Fi_tanh, Fi_res

    def _get_fitness_function_exp(self, fi_tanh, y, coefs):
        F = self._get_fitness_function(fi_tanh, y, coefs)
        F_exp = np.exp(F) - 1 - self.offset
        norm_error = [(y.iloc[:, i] - F_exp[i]) for i in range(self.deg[-1])]
        return F_exp, norm_error

    def _aggregate(self, values, coeffs):
        return np.exp(np.dot(np.tanh(values), coeffs)) - 1

    def main(self, print_=False):
        prepared_data = self._prepare_data()
        normalized_data = self._norm_data(prepared_data)
        train_data_normalized, target_normalized = self._create_train_dataset(normalized_data)
        train_data, target = self._create_train_dataset(prepared_data)
        A_log, A = self._get_A_exp(train_data, self.deg)
        b, b_log = self._get_B(target)
        lambdas = self._get_lambdas(A_log, b_log)
        psi, psi_tanh = self._get_psi_exp(A_log, lambdas)
        A1 = self._get_A1(psi_tanh, b_log)
        Fi = self._get_Fi_exp(psi_tanh, A1)
        coefs = self._get_coefs(Fi, b_log)
        fitnes_result, error = self._get_fitness_function_exp(Fi, b, coefs) #b_log

        A_normalized_log, A_normalized = self._get_A_exp(train_data_normalized, self.deg)
        b_normalized, b_norm_log = self._get_B(target_normalized)
        lambdas_normalized = self._get_lambdas(A_normalized_log, b_norm_log)
        psi_normalized, psi_tanh_norm = self._get_psi_exp(A_normalized_log, lambdas_normalized)
        A1_normalized = self._get_A1(psi_tanh_norm, b_norm_log)
        Fi_normalized = self._get_Fi_exp(psi_tanh_norm, A1_normalized)
        coefs_normalized = self._get_coefs(Fi_normalized, b_norm_log)
        fitnes_result_normalized, error_normalized = self._get_fitness_function(Fi_normalized, b_normalized,
                                                                                coefs_normalized)

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

