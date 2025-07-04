# oculr

**Ocu[LR‖RL]**: a flexible, biologically inspired environment for learning-based active vision.

_Where agents don't just see — they seek._

---

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

## 📦 Installation

```bash
pip install oculr
```
