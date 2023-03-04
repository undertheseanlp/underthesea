import os
import pickle
from functools import partial
from typing import Deque

import fire
import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import matplotlib.pyplot as plt
import optax
from tqdm.auto import tqdm

from .acoustic_trainer import initial_state, loss_vag, val_loss_fn
from .config import FLAGS
from .data_loader import load_textgrid_wav
from .dsp import MelFilter
from .utils import print_flags


def setup_colab_tpu():
    jax.tools.colab_tpu.setup_tpu()


def train(
    batch_size: int = 32,
    steps_per_update: int = 10,
    learning_rate: float = 1024e-6,
):
    """Train acoustic model on multiple cores (TPU)."""
    lr_schedule = optax.exponential_decay(learning_rate, 50_000, 0.5, staircase=True)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=FLAGS.weight_decay),
    )

    def update_step(prev_state, inputs):
        params, aux, rng, optim_state = prev_state
        rng, new_rng = jax.random.split(rng)
        (loss, new_aux), grads = loss_vag(params, aux, rng, inputs)
        grads = jax.lax.pmean(grads, axis_name="i")
        updates, new_optim_state = optimizer.update(grads, optim_state, params)
        new_params = optax.apply_updates(params, updates)
        next_state = (new_params, new_aux, new_rng, new_optim_state)
        return next_state, loss

    @partial(jax.pmap, axis_name="i")
    def update(params, aux, rng, optim_state, inputs):
        states, losses = jax.lax.scan(
            update_step, (params, aux, rng, optim_state), inputs
        )
        return states, jnp.mean(losses)

    print(jax.devices())
    num_devices = jax.device_count()
    train_data_iter = load_textgrid_wav(
        FLAGS.data_dir,
        FLAGS.max_phoneme_seq_len,
        batch_size * num_devices * steps_per_update,
        FLAGS.max_wave_len,
        "train",
    )
    val_data_iter = load_textgrid_wav(
        FLAGS.data_dir,
        FLAGS.max_phoneme_seq_len,
        batch_size,
        FLAGS.max_wave_len,
        "val",
    )
    melfilter = MelFilter(
        FLAGS.sample_rate,
        FLAGS.n_fft,
        FLAGS.mel_dim,
        FLAGS.fmin,
        FLAGS.fmax,
    )
    batch = next(train_data_iter)
    batch = jax.tree_map(lambda x: x[:1], batch)
    batch = batch._replace(mels=melfilter(batch.wavs.astype(jnp.float32) / (2 ** 15)))
    params, aux, rng, optim_state = initial_state(optimizer, batch)
    losses = Deque(maxlen=1000)
    val_losses = Deque(maxlen=100)

    last_step = -steps_per_update

    # loading latest checkpoint
    ckpt_fn = FLAGS.ckpt_dir / "acoustic_latest_ckpt.pickle"
    if ckpt_fn.exists():
        print("Resuming from latest checkpoint at", ckpt_fn)
        with open(ckpt_fn, "rb") as f:
            dic = pickle.load(f)
            last_step, params, aux, rng, optim_state = (
                dic["step"],
                dic["params"],
                dic["aux"],
                dic["rng"],
                dic["optim_state"],
            )

    tr = tqdm(
        range(
            last_step + steps_per_update, FLAGS.num_training_steps + 1, steps_per_update
        ),
        desc="training",
        total=FLAGS.num_training_steps // steps_per_update + 1,
        initial=last_step // steps_per_update + 1,
    )

    params, aux, rng, optim_state = jax.device_put_replicated(
        (params, aux, rng, optim_state), jax.devices()
    )

    def batch_reshape(batch):
        return jax.tree_map(
            lambda x: jnp.reshape(x, (num_devices, steps_per_update, -1) + x.shape[1:]),
            batch,
        )

    for step in tr:
        batch = next(train_data_iter)
        batch = batch_reshape(batch)
        (params, aux, rng, optim_state), loss = update(
            params, aux, rng, optim_state, batch
        )
        losses.append(loss)

        if step % 10 == 0:
            val_batch = next(val_data_iter)
            val_loss, val_aux, predicted_mel, gt_mel = val_loss_fn(
                *jax.tree_map(lambda x: x[0], (params, aux, rng)), val_batch
            )
            val_losses.append(val_loss)
            attn = jax.device_get(val_aux["acoustic_model"]["attn"])
            predicted_mel = jax.device_get(predicted_mel[0])
            gt_mel = jax.device_get(gt_mel[0])

        if step % 1000 == 0:
            loss = jnp.mean(sum(losses)).item() / len(losses)
            val_loss = sum(val_losses).item() / len(val_losses)
            tr.write(f"step {step}  train loss {loss:.3f}  val loss {val_loss:.3f}")

            # saving predicted mels
            plt.figure(figsize=(10, 10))
            plt.subplot(3, 1, 1)
            plt.imshow(predicted_mel.T, origin="lower", aspect="auto")
            plt.subplot(3, 1, 2)
            plt.imshow(gt_mel.T, origin="lower", aspect="auto")
            plt.subplot(3, 1, 3)
            plt.imshow(attn.T, origin="lower", aspect="auto")
            plt.tight_layout()
            plt.savefig(FLAGS.ckpt_dir / f"mel_{step}.png")
            plt.close()

            # saving checkpoint
            with open(ckpt_fn, "wb") as f:
                params_, aux_, rng_, optim_state_ = jax.tree_map(
                    lambda x: x[0], (params, aux, rng, optim_state)
                )
                pickle.dump(
                    {
                        "step": step,
                        "params": params_,
                        "aux": aux_,
                        "rng": rng_,
                        "optim_state": optim_state_,
                    },
                    f,
                )


if __name__ == "__main__":
    # we don't use these flags.
    del FLAGS.batch_size
    del FLAGS.learning_rate
    del FLAGS.duration_learning_rate
    del FLAGS.duration_lstm_dim
    del FLAGS.duration_embed_dropout_rate

    print_flags(FLAGS.__dict__)

    if "COLAB_TPU_ADDR" in os.environ:
        setup_colab_tpu()

    if not FLAGS.ckpt_dir.exists():
        print("Create checkpoint dir at", FLAGS.ckpt_dir)
        FLAGS.ckpt_dir.mkdir(parents=True, exist_ok=True)

    fire.Fire(train)
