import math
import torch

try:                                                  # torch >= 2.0
    from torch.optim.lr_scheduler import LRScheduler as _Base
except ImportError:                                   # older torch
    from torch.optim.lr_scheduler import _LRScheduler as _Base


class TriangularCosineLR(_Base):
    """Composed schedule: triangle wave x cosine anneal, with a phase shift.

        lr(t) = base_lr * tri((t + phase) / T) * cos(t / H)

        tri(u) = triangle on u in [0,1): 1/lr_ratio at the trough, 1 at the peak
        cos(t) = half-cosine from 1 down to cos_end_ratio over H steps,
                 clamped afterwards; 1.0 if H is None

    T (period_steps) and H (total_steps) are in OPTIMIZER STEPS, so the class is
    agnostic to what an "epoch" means. Build from epoch units with from_epochs().

    The phase shift is in steps. With T == steps_per_epoch and phase = 0 the
    triangle trough sits at every epoch boundary; phase = T/2 puts the peak there.

    Use it exactly like CosineAnnealingLR:

        opt = torch.optim.SGD(model.parameters(), lr=0.8, momentum=0.0,
                              weight_decay=5e-4)
        sched = TriangularCosineLR.from_epochs(
            opt, steps_per_epoch=len(train_loader), total_epochs=200,
            period_epochs=1.0, lr_ratio=1000.0)

        for epoch in range(200):
            for x, y in train_loader:
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
                sched.step()

    Shift the phase at any time (mid-run, or per epoch for randomised phase):

        sched.set_phase(sched.period_steps / 2)                # evaluate at peak
        sched.set_phase(rng.uniform(0, sched.period_steps))    # random per epoch
        sched.shift_phase(delta_steps)                         # relative shift

    Set lr_ratio=1.0 for a pure cosine anneal, or total_steps=None for a pure
    triangle. The optimiser's own weight decay is already scaled by the LR in both
    torch.optim.SGD and AdamW, so nothing else needs to be touched.
    """

    def __init__(self, optimizer, period_steps, total_steps=None, lr_ratio=1.0,
                 cos_end_ratio=1e-4, phase_steps=0.0, last_epoch=-1):
        if period_steps is None or float(period_steps) <= 1.0:
            raise ValueError("period_steps must be > 1")
        if total_steps is not None and float(total_steps) <= 0.0:
            raise ValueError("total_steps must be > 0, or None for no anneal")
        if float(lr_ratio) < 1.0:
            raise ValueError("lr_ratio must be >= 1 (max/min of the triangle)")
        if not 0.0 < float(cos_end_ratio) <= 1.0:
            raise ValueError("cos_end_ratio must be in (0, 1]")

        self.period_steps = float(period_steps)
        self.total_steps = None if total_steps is None else float(total_steps)
        self.lr_ratio = float(lr_ratio)
        self.cos_end_ratio = float(cos_end_ratio)
        self.phase_steps = float(phase_steps)
        # base __init__ calls step() once and therefore get_lr(), so every
        # attribute used there must already be set.
        super().__init__(optimizer, last_epoch)

    # ---------------- the schedule as pure functions ----------------
    @staticmethod
    def triangle(u, lr_ratio=1.0):
        """Triangle on u in [0,1): 1/lr_ratio at trough, 1.0 at peak."""
        u = u % 1.0
        tri = 2.0 * u if u < 0.5 else 2.0 - 2.0 * u
        lo = 1.0 / lr_ratio
        return lo + (1.0 - lo) * tri

    @staticmethod
    def cosine(t, total_steps, end_ratio=1e-4):
        """1.0 -> end_ratio over total_steps, clamped after; 1.0 if no anneal."""
        if total_steps is None:
            return 1.0
        p = min(max(float(t) / float(total_steps), 0.0), 1.0)
        return end_ratio + (1.0 - end_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))

    def ratio_at(self, step):
        """Multiplier applied to base_lr at an absolute step (phase included)."""
        tri = self.triangle((float(step) + self.phase_steps) / self.period_steps,
                            self.lr_ratio)
        return tri * self.cosine(step, self.total_steps, self.cos_end_ratio)

    def get_lr(self):
        r = self.ratio_at(self.last_epoch)
        return [base * r for base in self.base_lrs]

    # ---------------- phase control ----------------
    def set_phase(self, phase_steps, apply=True):
        """Set the absolute phase offset (steps), wrapped to one period.

        apply=True also rewrites the current param-group LRs, so the change takes
        effect immediately instead of at the next step().
        """
        self.phase_steps = float(phase_steps) % self.period_steps
        if apply:
            self._write_lrs()

    def shift_phase(self, delta_steps, apply=True):
        """Shift the phase by delta_steps relative to the current phase."""
        self.set_phase(self.phase_steps + float(delta_steps), apply=apply)

    def _write_lrs(self):
        vals = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, vals):
            group["lr"] = lr
        self._last_lr = vals          # keep LRScheduler's bookkeeping consistent

    # ---------------- checkpoint correctness ----------------
    def load_state_dict(self, state_dict):
        """Restore, then re-apply the LR for the restored step.

        PyTorch's base implementation restores last_epoch but leaves param-group LRs
        at whatever a freshly constructed scheduler set, so a resumed run would take
        its first steps at the wrong learning rate.
        """
        super().load_state_dict(state_dict)
        self._write_lrs()

    # ---------------- convenience ----------------
    @classmethod
    def from_epochs(cls, optimizer, steps_per_epoch, total_epochs=None,
                    period_epochs=1.0, lr_ratio=1.0, cos_end_ratio=1e-4,
                    phase_epochs=0.0, last_epoch=-1):
        """Build from epoch units; the schedule itself only ever sees steps."""
        spe = float(steps_per_epoch)
        return cls(
            optimizer,
            period_steps=float(period_epochs) * spe,
            total_steps=None if total_epochs is None else float(total_epochs) * spe,
            lr_ratio=lr_ratio,
            cos_end_ratio=cos_end_ratio,
            phase_steps=float(phase_epochs) * spe,
            last_epoch=last_epoch,
        )

    def extra_repr(self):
        return (f"period_steps={self.period_steps:g}, "
                f"total_steps={self.total_steps}, lr_ratio={self.lr_ratio:g}, "
                f"cos_end_ratio={self.cos_end_ratio:g}, "
                f"phase_steps={self.phase_steps:g}")