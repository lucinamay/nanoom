import jax
import jax.numpy as jnp


def _auto_n_clusters(descriptors: jax.Array) -> int:
    return max(2, int(descriptors.shape[0] ** 0.5))


def dummy_hash_clustering(
    descriptors: jax.Array, n_clusters: int | None = None
) -> jax.Array:
    """Hash each feature vector to a cluster index via row-wise hashing."""
    n_clusters = n_clusters if n_clusters is not None else _auto_n_clusters(descriptors)
    hashes = jax.vmap(lambda row: jnp.dot(row, jnp.arange(1, row.shape[0] + 1)))(
        descriptors
    )
    hashes = jax.vmap(lambda row: jnp.dot(row, jnp.arange(1, row.shape[0] + 1)))(descriptors)
    return (hashes % n_clusters).astype(jnp.int32)
