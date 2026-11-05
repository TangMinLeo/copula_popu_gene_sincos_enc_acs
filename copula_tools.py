import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF


class CopulaScaler():
    """
    This class can encode and decode features using the probability integral transform
    and the pseudo-inverse of the provided data. It can also resample discrete encoded
    features uniformly between the steps of their Empirical Cumulative Distribution Function (ECDF).
    """
        
    def fit(self, data):
        """
        Compute the ECDFs and sorted pairs (x, ECDF(x)) used for scaling.

        Parameters
        ----------
        data : pd.DataFrame
            The data for which to compute ECDFs and pseudo-inverses.

        Returns
        -------
        self : object
            Fitted scaler object.
        """
        self.data = data
        self.columns = data.columns
        self.ecdfs = {}  # Empirical CDF as a step function.
        self.ecdf_map = {}  # {column : [(x, ecdf(x)), ...], ...} sorted
        self.u_dist = {}

        for column in self.columns:
            feature = data[column]
            # ECDF
            # ecdf = ECDF(list(feature))
            ecdf = ECDF(feature)
            self.ecdfs[column] = ecdf
            # Create the mapping for the pseudo inverse
            x_u = []  # [(x_1, ECDF(x_1)), ..., (x_d, ECDF(x_d))] with x_i < x_j for i < j
            uniques = np.sort(feature.unique())  # [x_1, ..., x_d]
            U = []
            for x in uniques:
                u = ecdf(x)
                U.append(u)
                x_u.append((x, u))
            self.ecdf_map[column] = x_u
            self.u_dist[column] = U

        return self

    # marginals are the dictionary consists of all the marginal distributions with each key represents an attribute distribution
    # here requires the ascending order arrangement of numerical attribute, if quantized into interval, make sure the ascending order of intervals
    def fit_from_marginals(self, marginals):
        for key in marginals.keys():
            ordered_dict = dict(sorted(marginals[key].items(), key=lambda x: x[0]))
            # print(ordered_dict)
            marginals[key] = ordered_dict

        print(marginals)
        self.marginals = marginals
        self.num_variables = len(marginals)

        self.ecdf_map = {}

        self.answer_mapping = {}
        self.reverse_mapping = {}
        self.columns = marginals.keys()
        self.u_dist = {}

        for key in marginals.keys():
            # each attribute, marginals[key] is a discrete distribution for attribute 'key'
            l = 0
            u = 0.0
            sub_mapping = {}
            rev_mapping = {}
            x_u = []
            u_s = []
            for subkey in marginals[key].keys():
                sub_mapping[subkey] = l
                rev_mapping[l] = subkey
                x = l
                u = u + marginals[key][subkey]
                x_u.append((x, u))
                u_s.append(u)

                l = l + 1
            self.answer_mapping[key] = sub_mapping
            self.reverse_mapping[key] = rev_mapping

            self.ecdf_map[key] = x_u
            self.u_dist[key] = u_s

        return self
            
    def transform(self, data):
        """Apply the probability integral transform on the marginal distributions.

        Parameters
        ----------
        data : pd.DataFrame
            The data used to scale,

        Returns
        -------
        data_tr : pd.DataFrame
            The transformed data 
        """
        data_tr = data.copy()
        for column in self.columns:
            data_tr[column] = data_tr[column].apply(self.ecdfs[column])
        return data_tr

    def transform_from_marginals(self, data):
        """Apply the probability integral transform on the marginal distributions.

        Parameters
        ----------
        data : pd.DataFrame
            The data used to scale,

        Returns
        -------
        data_tr : pd.DataFrame
            The transformed data
        """
        data_tr = data.copy()
        out_data = pd.DataFrame()
        for column in data_tr.columns:
            print(column)
            # x_u = np.array(self.ecdf_map[column])
            mapping = self.answer_mapping[column]
            print(mapping)
            print(self.ecdf_map[column])
            print(data_tr[column].value_counts())
            mapped = []
            for ii in range(len(data_tr[column])):
                k = data_tr[column].iloc[ii]
                # print(k)
                ordered_num = mapping[k]
                mapped.append(ordered_num)
            print(mapped)
            ecdf = ECDF(mapped)
            print(ecdf(mapped))

            out_data[column] = pd.Series(ecdf(mapped))
        return out_data

    def inverse_transform(self, data):
        """Apply the pseudo-inverse on the marginal distributions.

        Parameters
        ----------
        data : pd.DataFrame 
            The scaled data to be mapped back to the original space.

        Returns
        -------
        data_tr : pd.DataFrame
            The trasnformed data
        """
        data_tr = data.copy()
        for column in self.columns:
            x_u = np.array(self.ecdf_map[column])
            data_tr[column] = data_tr[column].apply(CopulaScaler.pseudo_inverse, args=(x_u,))
        return data_tr

    def interpolation(self, data, categorical):
        """Linear interpolations between consecutive steps of the ECDF.

        Parameters
        ----------
        data : pd.DataFrame
            The data for which to interpolate

        categorical : array-like
            The discrete features to interpolate

        Returns
        -------
        data_tr : pd.DataFrame
            The data with interpolated discrete features 
        """

        def sample_uniform(u, x_u):
            us = x_u[:, 1]
            # idx = list(us).index(u)
            # idx = np.searchsorted(us, u, side='left')
            # print(x_u)
            # print(us)
            # print(u)
            # print(idx)
            # print(len(us))
            idx = 0
            for id in range(len(us)):
                if u <= us[id]:
                    idx = id
                    break

            # print(u)
            # print(idx)

            if idx == 0:
                return np.random.uniform(low=0, high=u)
                # return np.random.uniform(low=0, high=us[idx])
            else:
                return np.random.uniform(low=us[idx-1], high=u)
                # return np.random.uniform(low=us[idx-1], high=us[idx])

        data_tr = data.copy()
        for column in categorical:
            # print('column: ', column)
            x_u = np.array(self.ecdf_map[column])
            # print(x_u)
            data_tr[column] = data_tr[column].apply(sample_uniform, args=(x_u,))
        return data_tr

    @staticmethod
    def pseudo_inverse(u, x_u):
        """Compute pseudo inverse.

        Parameters
        ----------
        u : float
            The value for which to compute the pseudo-inverse

        x_u : array-like of Tuples
            Array of pairs (x, ECDF(x)) 

        Returns
        -------
        x : float
            The pseudo-inverse value of u given the ECDF x_u

            x = inf{t; F(t)>=u}
        """
        us = x_u[:, 1]
        valid_i = np.ma.MaskedArray(us, us < u)
        x = x_u[np.ma.argmin(valid_i), 0]
        return x
