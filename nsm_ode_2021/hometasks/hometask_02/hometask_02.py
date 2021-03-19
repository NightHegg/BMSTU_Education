from math import sqrt
import numpy as np

def main():
    a = np.array([[1/4, 1/4 - sqrt(3) / 6], [1/4 + sqrt(3) / 6, 1/4]]) 
    b = np.array([1/2, 1/2])
    c = np.array([1/2 - sqrt(3) / 6, 1/2 + sqrt(3) / 6])

    condition_tree_4_2 = np.diag(np.diag(b) @ np.diag(c)) @ a @ c
    condition_tree_4_3 = b @ a @ np.diag(np.diag(c) @ np.diag(c))
    condition_tree_4_4 = b @ a @ a @ c 
    try:
        assert np.isclose(condition_tree_4_2, 1/8),  'Условие для дерева 4_2 не выполнено'
        assert np.isclose(condition_tree_4_3, 1/12), 'Условие для дерева 4_3 не выполнено'
        assert np.isclose(condition_tree_4_4, 1/24), 'Условие для дерева 4_4 не выполнено'
    except:
        raise
    print('Все вычисления верны!')

if __name__ == "__main__":
    main()