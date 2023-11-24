import itertools

window_size = [3, 5, 7, 9, 11]
functions = ['CalculateSeasons', 'average', 'Variance', 'Covariance', 'Median', 'Entropy', 'Mode']

combinations = list(itertools.product([True, False], repeat=len(functions)))
i =0
for window in window_size:
    for combo in combinations:
        combination_dict = dict(zip(functions, combo))
        # print(f"Window Size: {window}, Functions: {combination_dict}")
        i += 1
print(i)