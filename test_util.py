import jax
from jax import numpy as jnp
import util


tree = {
    'a': [jnp.array(1.), jnp.array(2.)],  # List of arrays
    'b': (jnp.array(3.), jnp.array(4.)),  # Tuple of arrays
    'c': {'d': jnp.array(5.)}  # Nested dictionary with an array
}


def test_tree_inner():
    print(">>>tree_inner:", util.tree_inner_product(tree, tree))
    print(">>>tree_norm:", util.tree_norm(tree)**2)


if __name__ == "__main__":
    test_tree_inner()