import numpy as np


def over_all_cost(env_params, X, y1, y2):
    """
    Calculate overall_cost in terms of the specified raw features and 2 groups of strategies.
    This is a robust version that handles nested list/array data types.
    :param env_params: a tuple of environment parameters.
    :param X: raw features, possibly with nested data structures.
    :param y1: prediction 1
    :param y2: prediction 2
    :return: average costs for y1, y2, local-only, and edge-only strategies.
    """

    def _to_float(value):
        """
        A robust utility to convert any potential sequence-like object to a float.
        It repeatedly extracts the first element until it gets a non-sequence type.
        """
        while isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                # Handle empty sequences to avoid errors, return a default value like 0.0
                return 0.0
            value = value[0]
        return float(value)

    # --- Step 1: Robustly parse all environment parameters ---
    F = _to_float(env_params[0])
    kappa = _to_float(env_params[1])
    P_t = _to_float(env_params[2])
    P_I = _to_float(env_params[3])
    P_d = _to_float(env_params[4])

    # --- Step 2: Initialize cost accumulators ---
    costs_1, costs_2 = [], []
    costs_local, costs_offload = [], []
    mu_num = len(y1[0])

    # --- Step 3: Iterate through each sample and user to calculate costs ---
    for i in range(X.shape[0]):
        cost_1_sample = 0
        cost_2_sample = 0
        cost_local_sample = 0
        cost_offload_sample = 0

        for j in range(mu_num):
            # --- Step 3a: Robustly extract and convert user-specific parameters from X ---
            s_n = _to_float(X[i][6 * j])
            w_n = s_n  # Assuming w_n is the same as s_n
            c_n = _to_float(X[i][6 * j + 1])
            f_n = _to_float(X[i][6 * j + 2])
            r_u = _to_float(X[i][6 * j + 3])
            r_d = r_u  # Assuming r_d is the same as r_u
            alpha = _to_float(X[i][6 * j + 4])

            # --- Step 3b: Calculate cost for prediction y1 ---
            # Offloading decision
            if y1[i][j] > 0.1:
                # Prevent division by zero if f_n is 0.0
                if F > 1e-9 and y1[i][j] > 1e-9 and r_u > 1e-9 and r_d > 1e-9:
                    offload_latency = (s_n / r_u) + (c_n / (F * y1[i][j])) + (w_n / r_d)
                    offload_energy = (P_t * s_n / r_u) + (P_I * c_n / (F * y1[i][j])) + (P_d * w_n / r_d)
                    cost_1_sample += alpha * offload_latency + (1 - alpha) * offload_energy
                # else: handle edge case, e.g., assign a very high cost
            # Local execution decision
            else:
                if f_n > 1e-9:
                    local_latency = c_n / f_n
                    local_energy = kappa * (f_n ** 2) * c_n
                    cost_1_sample += alpha * local_latency + (1 - alpha) * local_energy
                # else: handle edge case, e.g., assign a very high cost

            # --- Step 3c: Calculate cost for prediction y2 (Same logic as y1) ---
            if y2[i][j] > 0.1:
                if F > 1e-9 and y2[i][j] > 1e-9 and r_u > 1e-9 and r_d > 1e-9:
                    offload_latency = (s_n / r_u) + (c_n / (F * y2[i][j])) + (w_n / r_d)
                    offload_energy = (P_t * s_n / r_u) + (P_I * c_n / (F * y2[i][j])) + (P_d * w_n / r_d)
                    cost_2_sample += alpha * offload_latency + (1 - alpha) * offload_energy
            else:
                if f_n > 1e-9:
                    local_latency = c_n / f_n
                    local_energy = kappa * (f_n ** 2) * c_n
                    cost_2_sample += alpha * local_latency + (1 - alpha) * local_energy

            # --- Step 3d: Calculate baseline costs (Local-only and Edge-only) ---
            # Local-only cost
            if f_n > 1e-9:
                local_latency = c_n / f_n
                local_energy = kappa * (f_n ** 2) * c_n
                cost_local_sample += alpha * local_latency + (1 - alpha) * local_energy

            # Edge-only cost (assuming equal resource division)
            if F > 1e-9 and r_u > 1e-9 and r_d > 1e-9:
                F_per_user = F / mu_num
                offload_latency = (s_n / r_u) + (c_n / F_per_user) + (w_n / r_d)
                offload_energy = (P_t * s_n / r_u) + (P_I * c_n / F_per_user) + (P_d * w_n / r_d)
                cost_offload_sample += alpha * offload_latency + (1 - alpha) * offload_energy

        costs_1.append(cost_1_sample)
        costs_2.append(cost_2_sample)
        costs_local.append(cost_local_sample)
        costs_offload.append(cost_offload_sample)

    # --- Step 4: Calculate and return the average costs ---
    avg_cost_1 = np.mean(costs_1)
    avg_cost_2 = np.mean(costs_2)
    avg_cost_local = np.mean(costs_local)
    avg_cost_offload = np.mean(costs_offload)

    return avg_cost_1, avg_cost_2, avg_cost_local, avg_cost_offload