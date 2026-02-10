def test_mse_loss():
    import jax
    import jax.numpy as jnp

    from so3lr.mlff.training.loss import mse_loss
    a = jax.random.uniform(jax.random.PRNGKey(0), (10, 12, 3))
    b = jax.random.uniform(jax.random.PRNGKey(1), (10, 12, 3))

    mse_gt = jnp.mean((b - a)**2)

    assert jnp.isclose(mse_gt, mse_loss(y=a, y_true=b), atol=1e-5)


def test_mse_loss_gradients():
    import jax
    import jax.numpy as jnp

    from so3lr.mlff.training.loss import mse_loss
    a = jax.random.uniform(jax.random.PRNGKey(0), (10, 12, 3))
    b = jax.random.uniform(jax.random.PRNGKey(1), (10, 12, 3))

    mse_gt = lambda x, y: jnp.mean((y - x) ** 2)

    grad_mse_gt = jax.grad(mse_gt)(a, b)

    assert jnp.isclose(grad_mse_gt, jax.grad(lambda x, y: mse_loss(y=x, y_true=y))(a, b), atol=1e-5).all()


def test_masked_and_scaled_mse_loss():
    import jax
    import jax.numpy as jnp

    from so3lr.mlff.training.loss import scaled_safe_masked_mse_loss

    a = jax.random.uniform(jax.random.PRNGKey(0), (10, 12, 3))
    b = jax.random.uniform(jax.random.PRNGKey(1), (10, 12, 3))

    mse_fn = lambda y, y_true: jnp.mean((y_true - y) ** 2)
    mse_gt = mse_fn(a, b)

    msk = jnp.ones((a.shape[0]))[:, None, None].astype(bool)
    mse = scaled_safe_masked_mse_loss(y=a, y_true=b, scale=jnp.ones(1), msk=msk)

    assert jnp.isclose(mse_gt, mse)

    # check that masking works
    a_ = jnp.concatenate([a, jnp.zeros((2, 12, 3))], axis=0)
    b_ = jnp.concatenate([b, jnp.zeros((2, 12, 3))], axis=0)
    msk_ = jnp.concatenate([msk, jnp.zeros((2, 1, 1)).astype(bool)], axis=0)

    # check that one would now obtain a different loss without masking ...
    assert ~jnp.isclose(mse_fn(y=a_, y_true=b_),
                        mse)

    # which is actually smaller due to the zeros
    assert mse_fn(y=a_, y_true=b_) < mse

    # check that the scaled and masked loss gives the correct results
    assert jnp.isclose(mse,
                       scaled_safe_masked_mse_loss(y=a_, y_true=b_, scale=jnp.ones(1), msk=msk_))

    # check that exclusion of NaNs works
    a_ = jnp.concatenate([jax.random.uniform(jax.random.PRNGKey(0), (1, 12, 3)),
                          a,
                          jax.random.uniform(jax.random.PRNGKey(0), (1, 12, 3))], axis=0)

    b_ = jnp.concatenate([jnp.full([1, 12, 3], jnp.nan),
                          b,
                          jnp.full([1, 12, 3], jnp.nan)], axis=0)

    msk_ = jnp.concatenate([msk,
                            jnp.ones((2, 1, 1)).astype(bool)], axis=0)

    assert jnp.isnan(mse_fn(y=a_, y_true=b_))

    assert jnp.isclose(mse,
                       scaled_safe_masked_mse_loss(y=a_, y_true=b_, scale=jnp.ones(1), msk=msk_))


def test_gradients_masked_and_scaled_mse_loss():
    import jax
    import jax.numpy as jnp

    from so3lr.mlff.training.loss import scaled_safe_masked_mse_loss

    alpha = jnp.array(jnp.pi)

    a = jax.random.uniform(jax.random.PRNGKey(0), (10, 12, 3))
    b = jax.random.uniform(jax.random.PRNGKey(1), (10, 12, 3))

    mse_fn = lambda y, y_true: jnp.mean((y_true - y) ** 2)

    def loss_fn_gt(p, y, y_true):
        return mse_fn(y=p*y, y_true=y_true)

    loss_gt = loss_fn_gt(alpha, a, b)
    grad_loss_gt = jax.grad(loss_fn_gt)(alpha, y=a, y_true=b)

    def loss_fn(p, y, y_true):
        msk = jnp.ones((a.shape[0]))[:, None, None].astype(bool)
        return scaled_safe_masked_mse_loss(y=p*y, y_true=y_true, scale=jnp.ones(1), msk=msk)

    loss = loss_fn(alpha, y=a, y_true=b)
    grad_loss = jax.grad(loss_fn)(alpha, y=a, y_true=b)

    assert jnp.isclose(loss_gt, loss)
    assert jnp.isclose(grad_loss_gt, grad_loss).all()

    # check that masking works
    def loss_fn(p, y, y_true):
        msk = jnp.ones((a.shape[0]))[:, None, None].astype(bool)
        msk_ = jnp.concatenate([msk, jnp.zeros((2, 1, 1)).astype(bool)], axis=0)
        return scaled_safe_masked_mse_loss(y=p*y, y_true=y_true, scale=jnp.ones(1), msk=msk_)

    a_ = jnp.concatenate([a, jnp.zeros((2, 12, 3))], axis=0)
    b_ = jnp.concatenate([b, jnp.zeros((2, 12, 3))], axis=0)

    # check that one would now obtain a different loss without masking ...
    assert ~jnp.isclose(loss_fn_gt(alpha, y=a_, y_true=b_),
                        loss_gt)

    # check that one would now obtain a different gradient loss without masking ...
    assert ~jnp.isclose(jax.grad(loss_fn_gt)(alpha, y=a_, y_true=b_),
                        grad_loss_gt).all()

    # check that the scaled and masked loss gives the correct results
    assert jnp.isclose(loss_gt,
                       loss_fn(alpha, y=a_, y_true=b_))

    # check that the scaled and masked loss gives the correct gradients
    assert jnp.isclose(grad_loss_gt,
                       jax.grad(loss_fn)(alpha, y=a_, y_true=b_))

    # check that masking works
    def loss_fn(p, y, y_true):
        msk = jnp.ones((a.shape[0]))[:, None, None].astype(bool)
        msk_ = jnp.concatenate([msk,
                                jnp.ones((2, 1, 1)).astype(bool)], axis=0)
        return scaled_safe_masked_mse_loss(y=p * y, y_true=y_true, scale=jnp.ones(1), msk=msk_)

    # check that exclusion of NaNs works
    a_ = jnp.concatenate([jax.random.uniform(jax.random.PRNGKey(0), (1, 12, 3)),
                          a,
                          jax.random.uniform(jax.random.PRNGKey(0), (1, 12, 3))], axis=0)

    b_ = jnp.concatenate([jnp.full([1, 12, 3], jnp.nan),
                          b,
                          jnp.full([1, 12, 3], jnp.nan)], axis=0)

    assert jnp.isnan(loss_fn_gt(alpha, y=a_, y_true=b_))

    assert jnp.isnan(jax.grad(loss_fn_gt)(alpha, y=a_, y_true=b_))

    assert jnp.isclose(loss_gt,
                       loss_fn(alpha, y=a_, y_true=b_))

    assert jnp.isclose(grad_loss_gt,
                       jax.grad(loss_fn)(alpha, y=a_, y_true=b_))
