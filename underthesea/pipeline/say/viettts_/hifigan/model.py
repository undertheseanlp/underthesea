import haiku as hk
import jax
import jax.numpy as jnp

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    p = int((kernel_size * dilation - dilation) / 2)
    return ((p, p),)


class ResBlock1(hk.Module):
    def __init__(
        self, h, channels, kernel_size=3, dilation=(1, 3, 5), name="resblock1"
    ):
        super().__init__(name=name)

        self.h = h
        self.convs1 = [
            hk.Conv1D(
                channels,
                kernel_size,
                1,
                rate=dilation[i],
                padding=get_padding(kernel_size, dilation[i]),
                name=f"convs1_{i}",
            )
            for i in range(3)
        ]

        self.convs2 = [
            hk.Conv1D(
                channels,
                kernel_size,
                1,
                rate=1,
                padding=get_padding(kernel_size, 1),
                name=f"convs2_{i}",
            )
            for i in range(3)
        ]

    def __call__(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = jax.nn.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = jax.nn.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(hk.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), name="ResBlock2"):
        super().__init__(name=name)
        self.h = h
        self.convs = [
            hk.Conv1D(
                channels,
                kernel_size,
                1,
                rate=dilation[i],
                padding=get_padding(kernel_size, dilation[i]),
            )
            for i in range(2)
        ]

    def __call__(self, x):
        for c in self.convs:
            xt = jax.nn.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class Generator(hk.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = hk.Conv1D(h.upsample_initial_channel, 7, 1, padding=((3, 3),))
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2
        self.ups = []
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                hk.Conv1DTranspose(
                    h.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_shape=k,
                    stride=u,
                    padding="SAME",
                    name=f"ups_{i}",
                )
            )

        self.resblocks = []

        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock(h, ch, k, d, name=f"res_block1_{len(self.resblocks)}")
                )
        self.conv_post = hk.Conv1D(1, 7, 1, padding=((3, 3),))

    def __call__(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = jax.nn.leaky_relu(x, LRELU_SLOPE)

            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = jax.nn.leaky_relu(x)  # default pytorch value
        x = self.conv_post(x)
        x = jnp.tanh(x)
        return x
