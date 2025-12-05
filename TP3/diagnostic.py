import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

# Test to understand the issue with z not changing

# First, let's check if different keys produce different outputs from SRW_HM
print("=" * 60)
print("TEST 1: Does SRW_HM produce different outputs with different keys?")
print("=" * 60)

# We'll need to load the data from the notebook
# For now, let's create a simple test case

def test_key_usage():
    """Test if keys are being used correctly in scan"""

    def step_with_key(carry, key):
        # Simulate what happens in SAEM: call a random operation
        value = carry + jr.normal(key)
        return value, value

    key = jr.PRNGKey(0)
    keys = jr.split(key, 5)

    init = 0.0
    final, trace = jax.lax.scan(step_with_key, init, keys)

    print("Trace of values (should all be different):")
    print(trace)
    print()

    # Now test if calling the same function multiple times gives different results
    print("Calling with same keys again (should give same results):")
    final2, trace2 = jax.lax.scan(step_with_key, init, keys)
    print(trace2)
    print("Are traces identical?", jnp.allclose(trace, trace2))
    print()

test_key_usage()

print("=" * 60)
print("TEST 2: Checking key splitting behavior")
print("=" * 60)

key = jr.PRNGKey(42)
keys = jr.split(key, 3)
print(f"Original key: {key}")
print(f"Split keys shape: {keys.shape}")
for i, k in enumerate(keys):
    print(f"Key {i}: {k}")
    sample = jr.normal(k)
    print(f"  Sample: {sample}")
print()
