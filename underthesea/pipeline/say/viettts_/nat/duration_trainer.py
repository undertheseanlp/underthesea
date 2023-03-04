# flake8: noqa

from functools import partial
from typing import Deque

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm.auto import tqdm
from vietTTS.nat.config import DurationInput

from .config import FLAGS
from .data_loader import textgrid_data_loader
from .model import DurationModel
from .utils import load_latest_ckpt, print_flags, save_ckpt


def loss_fn(params, aux, rng, x: DurationInput, is_training=True):
    @hk.transform_with_state
    def net(x):
        return DurationModel(is_training=is_training)(x)

    durations, aux = net.apply(params, aux, rng, x)
    mask = jnp.arange(0, x.phonemes.shape[1])[None, :] < x.lengths[:, None]
    # NOT predict [WORD END] token
    mask = jnp.where(x.phonemes == FLAGS.word_end_index, False, mask)
    masked_loss = jnp.abs(durations - x.durations) * mask
    loss = jnp.sum(masked_loss) / jnp.sum(mask)
    return loss, aux


forward_fn = jax.jit(
    hk.transform_with_state(lambda x: DurationModel(is_training=False)(x)).apply
)


def predict_duration(params, aux, rng, x: DurationInput):
    d, _ = forward_fn(params, aux, rng, x)
    return d, x.durations


val_loss_fn = jax.jit(partial(loss_fn, is_training=False))

loss_vag = jax.value_and_grad(loss_fn, has_aux=True)

optimizer = optax.chain(
    optax.clip_by_global_norm(FLAGS.max_grad_norm),
    optax.adamw(FLAGS.duration_learning_rate, weight_decay=FLAGS.weight_decay),
)


@jax.jit
def update(params, aux, rng, optim_state, inputs: DurationInput):
    rng, new_rng = jax.random.split(rng)
    (loss, new_aux), grads = loss_vag(params, aux, rng, inputs)
    updates, new_optim_state = optimizer.update(grads, optim_state, params)
    new_params = optax.apply_updates(params, updates)
    return loss, (new_params, new_aux, new_rng, new_optim_state)


def initial_state(batch):
    rng = jax.random.PRNGKey(42)
    params, aux = hk.transform_with_state(lambda x: DurationModel(True)(x)).init(
        rng, batch
    )
    optim_state = optimizer.init(params)
    return params, aux, rng, optim_state


def plot_val_duration(step: int, batch, params, aux, rng):
    fn = FLAGS.ckpt_dir / f"duration_{step}.png"
    predicted_dur, gt_dur = predict_duration(params, aux, rng, batch)
    L = batch.lengths[0]
    x = np.arange(0, L) * 3
    plt.plot(predicted_dur[0, :L])
    plt.plot(gt_dur[0, :L])
    plt.legend(["predicted", "gt"])
    plt.title("Phoneme durations")
    plt.savefig(fn)
    plt.close()


def train():
    train_data_iter = textgrid_data_loader(
        FLAGS.data_dir, FLAGS.max_phoneme_seq_len, FLAGS.batch_size, mode="train"
    )
    val_data_iter = textgrid_data_loader(
        FLAGS.data_dir, FLAGS.max_phoneme_seq_len, FLAGS.batch_size, mode="val"
    )
    losses = Deque(maxlen=1000)
    val_losses = Deque(maxlen=100)
    latest_ckpt = load_latest_ckpt(FLAGS.ckpt_dir)
    if latest_ckpt is not None:
        last_step, params, aux, rng, optim_state = latest_ckpt
    else:
        last_step = -1
        print("Generate random initial states...")
        params, aux, rng, optim_state = initial_state(next(train_data_iter))

    tr = tqdm(
        range(last_step + 1, 1 + FLAGS.num_training_steps),
        total=1 + FLAGS.num_training_steps,
        initial=last_step + 1,
        ncols=80,
        desc="training",
    )
    for step in tr:
        batch = next(train_data_iter)
        loss, (params, aux, rng, optim_state) = update(
            params, aux, rng, optim_state, batch
        )
        losses.append(loss)

        if step % 10 == 0:
            val_loss, _ = val_loss_fn(params, aux, rng, next(val_data_iter))
            val_losses.append(val_loss)

        if step % 1000 == 0:
            loss = sum(losses).item() / len(losses)
            val_loss = sum(val_losses).item() / len(val_losses)
            plot_val_duration(step, next(val_data_iter), params, aux, rng)
            tr.write(
                f" {step:>6d}/{FLAGS.num_training_steps:>6d} | train loss {loss:.5f} | val loss {val_loss:.5f}"
            )
            save_ckpt(step, params, aux, rng, optim_state, ckpt_dir=FLAGS.ckpt_dir)


if __name__ == "__main__":
    print_flags(FLAGS.__dict__)
    if not FLAGS.ckpt_dir.exists():
        print("Create checkpoint dir at", FLAGS.ckpt_dir)
        FLAGS.ckpt_dir.mkdir(parents=True, exist_ok=True)
    train()
