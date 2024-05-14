import jax
from jax import numpy as jnp
import util


tree = {
    'a': [jnp.array([1., 2.])],  # List of arrays
    'b': (jnp.array(3.), jnp.array(4.)),  # Tuple of arrays
    'c': {'d': jnp.array(5.)}  # Nested dictionary with an array
}


def test_tree_inner():
    print(">>> Testing tree_inner_product.")
    print("    expected result:", 55.0)
    print("    actual result:", util.tree_inner_product(tree, tree))


def test_tree_l1_norm():
    print(">>> Testing tree_l1_norm.")
    print("    expected result:", 15.0)
    print("    actual result:", util.tree_l1_norm(tree))


def test_tree_cosine_similarity():
    print(">>> Testing tree_cosine_similarity.")
    print("    expected result:", 1.0)
    print("    actual result:", util.tree_cosine_similarity(tree, tree))
    print("    expected result:", -1.0)
    print("    actual result:", util.tree_cosine_similarity(tree, util.negative_tree(tree)))


def test_is_finite_tree():
    print(">>> Testing tree_cosine_similarity.")
    print("    expected result: True")
    print("    actual result:", util.is_finite_tree(tree))
    inf_tree = [jnp.array([jnp.nan])]
    print("    expected result: False")
    print("    actual result:", util.is_finite_tree(inf_tree))


if __name__ == "__main__":
    # test_tree_inner()
    # test_tree_l1_norm()
    # test_tree_cosine_similarity()
    # test_is_finite_tree()
    import os
    cwd = os.getcwd()
    print(f"dir: {cwd}")
    path = os.path.join(os.getcwd(), "dir/file")
    print(f"path: {path}")