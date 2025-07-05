# oculr

**Ocu[LR‖RL]**: a flexible, biologically inspired environment for learning-based active vision.

_Where agents don't just see — they seek._

---

## 📦 Install

```bash
pip install git+https://github.com/pkuderov/oculr.git
```

## Use

Here's an example snippet similar to `benchmark_env` in [this file](oculr/env.py). It shows how to create and use an environment with a random policy agent.

```python
from functools import partial
import numpy as np

from oculr.dataset import Dataset
from oculr.env import ImageEnvironment
from oculr.image.buffer import PrefetchedImageBuffer
from oculr.timer import timer

seed = 8041990
ds = Dataset(seed, 'cifar', grayscale=False, lp_norm=None)
env = ImageEnvironment(
    ds, num_envs=64, obs_hw_shape=7, max_time_steps=20, seed=42,
    answer_reward=(1.0, -0.3), step_reward=-0.01,
    img_buffer_fn=partial(PrefetchedImageBuffer, prefetch_size=256),
)

state = None
o, info = env.reset()
n_steps, print_schedule = 30000, 5000
ret, ep_len_sum, n_eps, v_loss, pi_loss, acc = 0., 0., 0., 0., 0., 0.
rng = np.random.default_rng()

a = np.vstack(
    [
        rng.integers(3, size=env.bsz),
        rng.integers(env.n_classes, size=env.bsz),
        rng.integers(env.pos_range[1][0], size=env.bsz),
        rng.integers(env.pos_range[1][1], size=env.bsz),
    ]
).T
print(a.shape)

st = timer()
for i in range(1, n_steps + 1):
    a = np.vstack(
        [
            rng.integers(3, size=env.bsz),
            rng.integers(env.n_classes, size=env.bsz),
            rng.integers(env.pos_range[1][0], size=env.bsz),
            rng.integers(env.pos_range[1][1], size=env.bsz),
        ]
    ).T
    o_next, r, terminated, truncated, info = env.step(a)

    ret += r.mean()
    ep_len_sum += info['ep_len_sum']
    acc += info['n_correct']
    n_eps += info['n_done']

    o = o_next

    if i % print_schedule == 0:
        ret /= print_schedule
        ep_len_sum /= n_eps
        acc /= n_eps / 100.0
        print(f'{i}: {acc:.2f} {ret:.3f}  |  {ep_len_sum:.1f}')
        ret, ep_len_sum, n_eps, v_loss, pi_loss, acc = 0., 0., 0., 0., 0., 0.

end = timer()
print(f'{n_steps * env.bsz / (end - st) / 1e+3:.2f} kfps')
```

## 🧠 Overview

**`oculr`** is a simulation environment for **active visual perception**, designed to support both:
- **Reinforcement learning agents** (RL) trained on reward signals
- **Biologically plausible agents** using recurrent or local learning (LR)

The environment models a visual world where the agent perceives the scene through a small, moveable **retinal field** (patch), choosing where and how to look.

---

## 🔍 Features

- **Foveated input**: small local or rescaled global glimpses
- **Movement-based perception**: action = attention
- **Spatiotemporal learning**: recurrent-friendly setup
- **Tunable complexity**: from simple to rich vision tasks
- **Bio-plausible compatibility**: online, local, recurrent
- **Flexible control**: single/multi-scale, multiple "eyes", continuous/discrete control

---

## 🚧 Planned Tasks

- Image classification (glimpse-based)
- Next-glimpse prediction
- Object localization / saliency
- Video tracking and temporal memory tasks

---
