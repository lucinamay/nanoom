# ========================== jax implementations ==============================


def _task_vs_clusters_array_jax(
    datapoints_tasks_array: jnp.ndarray, clusters: jnp.ndarray
) -> jnp.ndarray:
    """
    Create a cross-tabulation 2D numpy array counting the # data points per task, per cluster

    Args:
        tasks (2d-Array): list of task indices per data point (e.g. one-hot encoding matrix for multilabels)
        clusters (list): list of cluster indices per data point (e.g. clusters per compound)
    Returns:
        jnp.ndarray: 2D array of shape (num_tasks+1, num_clusters)

    Comment:
        In the returned array
        - each column is a unique initial cluster
        - each row is a unique task (except the first row, which is the
          total # objects in the cluster)
        This is the format requrired by the balancing algorithm
        see: https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/660581be9138d231618d6047/original/readme-md.md

    """

    task_vs_clusters = jnp.zeros((len(datapoints_tasks_array) + 1, len(clusters)))
    # first row is total number of items per cluster
    task_vs_clusters.at[0, :].set(
        jnp.array([jnp.sum(jnp.array(clusters) == i) for i in clusters.unique()])
    )
    # get the number of items per cluster that have non-null task values
    for i, task_list in enumerate(datapoints_tasks_array):
        for task in task_list:
            task_vs_clusters = task_vs_clusters.at[1 + i, clusters[i]].add(1)

    return task_vs_clusters


def _balance_data_from_tasks_vs_clusters_array_jax(
    tasks_vs_clusters_array: jnp.ndarray,
    sizes: list[float] = [0.4, 0.4, 0.2],
    equal_weight_perc_compounds_as_tasks: bool = False,
    relative_gap: int = 0,
    time_limit_seconds: float = 60 * 60,
    max_N_threads: int = int((os.cpu_count() or 1) // 1.2) + 1,
    verbose: bool = False,
) -> list:
    """
    Linear programming function needed to balance the data while merging clusters
    taken from https://doi.org/10.26434/chemrxiv-2022-m8l33-v3

    Args:
        tasks_vs_clusters_array : 2D jnp.array
            - the cross-tabulation of the number of data points per cluster, per task.
            - columns represent unique clusters.
            - rows represent tasks, except the first row, which represents the number of records (or compounds).
            - Optionally, instead of the number of data points, the provided array may contain the *percentages*
                of data points _for the task across all clusters_ (i.e. each *row*, NOT column, may sum to 1).
            IMPORTANT: make sure the array has 2 dimensions, even if only balancing the number of data records,
                so there is only 1 row. This can be achieved by setting ndmin = 2 in the jnp.array function.
        sizes : list
            - list of the desired final sizes (will be normalised to fractions internally).
        equal_weight_perc_compounds_as_tasks : bool
            - if True, matching the % records will have the same weight as matching the % data of individual tasks.
            - if False, matching the % records will have a weight X times larger than the X tasks.
        relative_gap : float
            - the relative gap between the absolute optimal objective and the current one at which the solver
            stops and returns a solution. Can be very useful for cases where the exact solution requires
            far too long to be found to be of any practical use.
            - set to 0 to obtain the absolute optimal solution (if reached within the time_limit_seconds)
        time_limit_seconds : int
            - the time limit in seconds for the solver (by default set to 1 hour)
            - after this time, whatever solution is available is returned
        max_N_threads : int
            - the maximal number of threads to be used by the solver.
            - it is advisable to set this number as high as allowed by the available resources.

    Returns:
        List (of length equal to the number of columns of tasks_vs_clusters_array) of final cluster identifiers
            (integers, numbered from 1 to len(sizes)), mapping each unique initial cluster to its final cluster.

        - Example: if sizes == [20, 10, 70], the output will be a list like [3, 3, 1, 2, 1, 3...], where
        '1' represents the final cluster of relative size 20, '2' the one of relative size 10, and '3' the
        one of relative size 70.


    Note: from https://doi.org/10.26434/chemrxiv-2022-m8l33-v3, original code can be found at
    https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/660581be9138d231618d604c/original/balance-data-from-tasks-vs-clusters-array-pulp-py.py
    """
    # Calculate the fractions from sizes
    raise NotImplementedError

    sizes_ = jnp.array(sizes)
    fractional_sizes = sizes_ / jnp.sum(sizes_)
    S = sizes_.shape[0]

    # Normalise the data matrix
    tasks_vs_clusters_array = tasks_vs_clusters_array / tasks_vs_clusters_array.sum(
        axis=1, keepdims=True
    )

    # Find the number of tasks + compounds (M) and the number of initial clusters (N)
    M, N = tasks_vs_clusters_array.shape
    if S > N:
        errormessage = (
            "The requested number of new clusters to make ("
            + str(S)
            + ") cannot be larger than the initial number of clusters ("
            + str(N)
            + "). Please review."
        )
        raise ValueError(errormessage)

    if relative_gap < 0:
        errormessage = f"relative gap should be 0 or positive, is {relative_gap}"
        raise ValueError(errormessage)

    # Given matrix A (M x N) of fraction of data per cluster, assign each cluster to one of S final ML subsets,
    # so that the fraction of data per ML subset is closest to the corresponding fraction_size.
    # The weights on each ML subset (WML, S x 1) are calculated from fractional_sizes harmonic-mean-like.
    # The weights on each task (WT, M x 1) are calculated as requested by the user.
    # In the end: argmin SUM(ABS((A.X-T).WML).WT)
    # where X is the (N x S) binary solution matrix
    # where T is the (M x S) matrix of target fraction sizes (repeat of fractional_sizes)
    # constraint: assign one cluster to one and only one final ML subset
    # i.e. each row of X must sum to 1

    A = jnp.copy(tasks_vs_clusters_array)

    # Create WT = obj_weights
    if (M > 1) & (not equal_weight_perc_compounds_as_tasks):
        obj_weights = jnp.array([M - 1] + [1] * (M - 1))
    else:
        obj_weights = jnp.array([1] * M)

    obj_weights = obj_weights / jnp.sum(obj_weights)

    # Create WML
    sk_harmonic = (1 / fractional_sizes) / jnp.sum(1 / fractional_sizes)

    # Create the pulp model
    prob = pulp.LpProblem("Data_balancing", pulp.LpMinimize)

    # Create the pulp variables
    # x_names represent clusters, ML_subsets, and are binary variables
    x_names = ["x_" + str(i) for i in range(N * S)]
    x = [
        pulp.LpVariable(x_names[i], lowBound=0, upBound=1, cat="Integer")
        for i in range(N * S)
    ]
    # X_names represent tasks, ML_subsets, and are continuous positive variables
    X_names = ["X_" + str(i) for i in range(M * S)]
    X = [
        pulp.LpVariable(X_names[i], lowBound=0, cat="Continuous") for i in range(M * S)
    ]

    # Add the objective to the model

    obj = []
    coeff = []
    for m in range(S):
        for t in range(M):
            obj.append(X[m * M + t])
            coeff.append(sk_harmonic[m] * obj_weights[t])

    prob += pulp.LpAffineExpression([(obj[i], coeff[i]) for i in range(len(obj))])

    # Add the constraints to the model

    # Constraints forcing each cluster to be in one and only one ML_subset
    for c in range(N):
        prob += pulp.LpAffineExpression([(x[c + m * N], +1) for m in range(S)]) == 1

    # Constraints forcing each ML_subset to be non-empty
    for m in range(S):
        prob += (
            pulp.LpAffineExpression([(x[i], +1) for i in range(m * N, (m + 1) * N)])
            >= 1
        )

    # Constraints related to the ABS values handling, part 1 and 2
    for m in range(S):
        for t in range(M):
            cs = [c for c in range(N) if A[t, c] != 0]
            prob += (
                pulp.LpAffineExpression([(x[c + m * N], A[t, c]) for c in cs])
                - X[m * M + t]
                <= fractional_sizes[m]
            )
            prob += (
                pulp.LpAffineExpression([(x[c + m * N], A[t, c]) for c in cs])
                + X[m * M + t]
                >= fractional_sizes[m]
            )

    # Solve the model
    prob.solve(
        pulp.PULP_CBC_CMD(
            gapRel=relative_gap,
            timeLimit=time_limit_seconds,
            threads=max_N_threads,
            msg=verbose,
        )
    )

    # Extract the solution

    list_binary_solution = [pulp.value(x[i]) for i in range(N * S)]
    list_initial_cluster_indices = [
        (list(range(N)) * S)[i] for i, li in enumerate(list_binary_solution) if li == 1
    ]
    list_final_ML_subsets = [
        (list((1 + jnp.repeat(range(S), N)).astype("int64")))[i]
        for i, li in enumerate(list_binary_solution)
        if li == 1
    ]
    mapping = [
        x for _, x in sorted(zip(list_initial_cluster_indices, list_final_ML_subsets))
    ]

    return mapping


# ========================= legacy implementations ==============================


def _balance_data_from_tasks_vs_clusters_array_legacy(
    tasks_vs_clusters_array: np.ndarray,
    sizes: list[float] = [0.4, 0.4, 0.3],
    equal_weight_perc_compounds_as_tasks: bool = False,
    relative_gap: int = 0,
    time_limit_seconds: float = 60 * 60,
    max_N_threads: int = int((os.cpu_count() or 1) // 1.2) + 1,
    verbose: bool = False,
) -> list:
    """
    Linear programming function needed to balance the data while merging clusters
    taken from https://doi.org/10.26434/chemrxiv-2022-m8l33-v3

    Args:
        tasks_vs_clusters_array : 2D np.array
            - the cross-tabulation of the number of data points per cluster, per task.
            - columns represent unique clusters.
            - rows represent tasks, except the first row, which represents the number of records (or compounds).
            - Optionally, instead of the number of data points, the provided array may contain the *percentages*
                of data points _for the task across all clusters_ (i.e. each *row*, NOT column, may sum to 1).
            IMPORTANT: make sure the array has 2 dimensions, even if only balancing the number of data records,
                so there is only 1 row. This can be achieved by setting ndmin = 2 in the np.array function.
        sizes : list
            - list of the desired final sizes (will be normalised to fractions internally).
        equal_weight_perc_compounds_as_tasks : bool
            - if True, matching the % records will have the same weight as matching the % data of individual tasks.
            - if False, matching the % records will have a weight X times larger than the X tasks.
        relative_gap : float
            - the relative gap between the absolute optimal objective and the current one at which the solver
            stops and returns a solution. Can be very useful for cases where the exact solution requires
            far too long to be found to be of any practical use.
            - set to 0 to obtain the absolute optimal solution (if reached within the time_limit_seconds)
        time_limit_seconds : int
            - the time limit in seconds for the solver (by default set to 1 hour)
            - after this time, whatever solution is available is returned
        max_N_threads : int
            - the maximal number of threads to be used by the solver.
            - it is advisable to set this number as high as allowed by the available resources.

    Returns:
        List (of length equal to the number of columns of tasks_vs_clusters_array) of final cluster identifiers
            (integers, numbered from 1 to len(sizes)), mapping each unique initial cluster to its final cluster.

        - Example: if sizes == [20, 10, 70], the output will be a list like [3, 3, 1, 2, 1, 3...], where
        '1' represents the final cluster of relative size 20, '2' the one of relative size 10, and '3' the
        one of relative size 70.


    Note: from https://doi.org/10.26434/chemrxiv-2022-m8l33-v3, original code can be found at
    https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/660581be9138d231618d604c/original/balance-data-from-tasks-vs-clusters-array-pulp-py.py
    """
    # Calculate the fractions from sizes

    fractional_sizes = sizes / np.sum(sizes)
    S = len(sizes)

    # Normalise the data matrix
    tasks_vs_clusters_array = tasks_vs_clusters_array / tasks_vs_clusters_array.sum(
        axis=1, keepdims=True
    )

    # Find the number of tasks + compounds (M) and the number of initial clusters (N)
    M, N = tasks_vs_clusters_array.shape
    if S > N:
        errormessage = (
            "The requested number of new clusters to make ("
            + str(S)
            + ") cannot be larger than the initial number of clusters ("
            + str(N)
            + "). Please review."
        )
        raise ValueError(errormessage)

    if relative_gap < 0:
        errormessage = f"relative gap should be 0 or positive, is {relative_gap}"
        raise ValueError(errormessage)

    # Given matrix A (M x N) of fraction of data per cluster, assign each cluster to one of S final ML subsets,
    # so that the fraction of data per ML subset is closest to the corresponding fraction_size.
    # The weights on each ML subset (WML, S x 1) are calculated from fractional_sizes harmonic-mean-like.
    # The weights on each task (WT, M x 1) are calculated as requested by the user.
    # In the end: argmin SUM(ABS((A.X-T).WML).WT)
    # where X is the (N x S) binary solution matrix
    # where T is the (M x S) matrix of target fraction sizes (repeat of fractional_sizes)
    # constraint: assign one cluster to one and only one final ML subset
    # i.e. each row of X must sum to 1

    A = np.copy(tasks_vs_clusters_array)

    # Create WT = obj_weights
    if (M > 1) & (not equal_weight_perc_compounds_as_tasks):
        obj_weights = np.array([M - 1] + [1] * (M - 1))
    else:
        obj_weights = np.array([1] * M)

    obj_weights = obj_weights / np.sum(obj_weights)

    # Create WML
    sk_harmonic = (1 / fractional_sizes) / np.sum(1 / fractional_sizes)

    # Create the pulp model
    prob = pulp.LpProblem("Data_balancing", pulp.LpMinimize)

    # Create the pulp variables
    # x_names represent clusters, ML_subsets, and are binary variables
    x_names = ["x_" + str(i) for i in range(N * S)]
    x = [
        pulp.LpVariable(x_names[i], lowBound=0, upBound=1, cat="Integer")
        for i in range(N * S)
    ]
    # X_names represent tasks, ML_subsets, and are continuous positive variables
    X_names = ["X_" + str(i) for i in range(M * S)]
    X = [
        pulp.LpVariable(X_names[i], lowBound=0, cat="Continuous") for i in range(M * S)
    ]

    # Add the objective to the model

    obj = []
    coeff = []
    for m in range(S):
        for t in range(M):
            obj.append(X[m * M + t])
            coeff.append(sk_harmonic[m] * obj_weights[t])

    prob += pulp.LpAffineExpression([(obj[i], coeff[i]) for i in range(len(obj))])

    # Add the constraints to the model

    # Constraints forcing each cluster to be in one and only one ML_subset
    for c in range(N):
        prob += pulp.LpAffineExpression([(x[c + m * N], +1) for m in range(S)]) == 1

    # Constraints forcing each ML_subset to be non-empty
    for m in range(S):
        prob += (
            pulp.LpAffineExpression([(x[i], +1) for i in range(m * N, (m + 1) * N)])
            >= 1
        )

    # Constraints related to the ABS values handling, part 1 and 2
    for m in range(S):
        for t in range(M):
            cs = [c for c in range(N) if A[t, c] != 0]
            prob += (
                pulp.LpAffineExpression([(x[c + m * N], A[t, c]) for c in cs])
                - X[m * M + t]
                <= fractional_sizes[m]
            )
            prob += (
                pulp.LpAffineExpression([(x[c + m * N], A[t, c]) for c in cs])
                + X[m * M + t]
                >= fractional_sizes[m]
            )

    # Solve the model
    prob.solve(
        pulp.PULP_CBC_CMD(
            gapRel=relative_gap,
            timeLimit=time_limit_seconds,
            threads=max_N_threads,
            msg=verbose,
        )
    )

    # Extract the solution

    list_binary_solution = [pulp.value(x[i]) for i in range(N * S)]
    list_initial_cluster_indices = [
        (list(range(N)) * S)[i] for i, li in enumerate(list_binary_solution) if li == 1
    ]
    list_final_ML_subsets = [
        (list((1 + np.repeat(range(S), N)).astype("int64")))[i]
        for i, li in enumerate(list_binary_solution)
        if li == 1
    ]
    mapping = [
        x for _, x in sorted(zip(list_initial_cluster_indices, list_final_ML_subsets))
    ]

    return mapping
