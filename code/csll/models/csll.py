"""CSLL -- Complex Spectral Lead-Lag Network (proposed method).

Cross-channel coupling is a per-frequency-band complex operator whose PHASE encodes an
inter-series lead-lag DELAY. We use an *additive* delay structure  D_ij = p_i - q_j  so that
the phase factorises per channel:

    Z_i(f) = sum_j A_ij exp(-i w_f (p_i - q_j)) X_j(f)
           = exp(-i w_f p_i) * ( A @ [ exp(i w_f q_j) X_j(f) ] )_i

i.e. a per-channel source phase (O(N)), a coupling matmul A (full or low-rank), and a
per-channel target phase (O(N)) -- no per-bin NxN operator is materialised.

Novelty knob -- DYNAMIC (input-adaptive) delays:  p_i, q_i are produced per input window by a
small controller,  p_i(X) = tau_max * tanh(p_i^base + ctrl(desc_i)).  A time-varying operator is
provably outside the linear-time-invariant filter class of all frequency-domain baselines
(FreTS/FITS/FilterTS/FNO), so it cannot be subsumed by "a general filter can absorb a delay".
Setting dynamic=False recovers the static special case (an ablation).

Ablation switches:
  dynamic=False      -> static delays (LTI operator)                              [A-static]
  use_phase=False    -> D:=0, real-valued mixing                                  [A-realphase]
  n_bands=1          -> single band                                              [A-1band]
  free_complex=True  -> unstructured per-band complex operator (no (A,p,q))       [A-freecplx]
  low_rank=r>0       -> A = U V^T (scales to large N)                            [A-lowrank]
  freeze_gate=True   -> spectral branch off (pure CI backbone)                    [A-backbone]
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn


def _series_decomp(x: torch.Tensor, kernel_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    pad = (kernel_size - 1) // 2
    xt = x.transpose(1, 2)
    left = xt[:, :, :1].repeat(1, 1, pad)
    right = xt[:, :, -1:].repeat(1, 1, kernel_size - 1 - pad)
    xp = torch.cat([left, xt, right], dim=2)
    trend = torch.nn.functional.avg_pool1d(xp, kernel_size=kernel_size, stride=1).transpose(1, 2)
    return trend, x - trend


def _make_bands(F: int, n_bands: int) -> List[Tuple[int, int]]:
    n_bands = max(1, min(n_bands, F))
    edges = [round(k * F / n_bands) for k in range(n_bands + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n_bands) if edges[i + 1] > edges[i]]


from ..revin import RevIN  # noqa: E402


class CSLL(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, n_bands=4, tau_max=None,
                 use_phase=True, dynamic=True, free_complex=False, low_rank=0,
                 kernel_size=25, use_revin=True, gate_init=0.0, freeze_gate=False,
                 desc_dim=16, ctrl_hidden=32, backbone="dlinear",
                 d_model=128, nhead=8, num_layers=3, dim_ff=256, dropout=0.1,
                 strict_bound=False, pad2x=False, direct=False, hybrid=False):
        super().__init__()
        self.L, self.H, self.N = seq_len, pred_len, n_vars
        self.use_phase = use_phase and not free_complex
        self.dynamic = bool(dynamic) and self.use_phase
        self.free_complex = free_complex
        self.low_rank = low_rank
        self.tau_max = float(tau_max) if tau_max is not None else seq_len / 2.0
        # strict_bound: bound p_i, q_j by tau_max/2 each so |D_ij| = |p_i - q_j| <= tau_max,
        # keeping the whole delay range inside the wrap-free identifiable region. The legacy
        # behaviour (False) bounds p, q by tau_max each, allowing |D| up to 2*tau_max.
        self.pq_bound = self.tau_max / 2.0 if strict_bound else self.tau_max
        self.pad2x = bool(pad2x)
        # direct: synthesise the phase-shifted spectrum AT THE FUTURE POSITIONS t in [L, L+H)
        # instead of reconstructing the look-back window and extrapolating with a shared
        # linear head. With delay D_ij = tau, the branch output at horizon h is the source
        # series read at (L+h-tau) -- i.e. the forecast-optimal delay IS the physical delay,
        # which is what makes the learned D interpretable. For h >= tau the read lands in the
        # zero padding (a lead of tau samples can only forecast tau steps ahead) and the gated
        # backbone covers the remainder. Requires pad2x and H <= L. Removes the L->H head from
        # the branch (and with it the head's ability to absorb a global delay).
        self.direct = bool(direct) and self.use_phase
        if self.direct:
            assert pad2x, "direct mode requires the zero-padded (linear-shift) basis"
        # hybrid: run BOTH a direct-phase branch (lead-lag; reads past into the future window)
        # and a real-mixing+head branch (same-time coupling; reconstructs the window and maps
        # L->H), each independently gated. In direct mode a zero-delay band reads the zero
        # padding and outputs ~nothing, so the phase branch alone cannot do same-time mixing;
        # the mixing branch supplies it. The two gates let the data pick the blend: on
        # same-time data the phase gate decays to 0 (mixing wins); on delayed data the phase
        # gate grows. A single model then spans traffic (mixing) and epidemics (delay).
        self.hybrid = bool(hybrid) and self.use_phase and self.direct

        self.backbone_type = backbone
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        if backbone == "itransformer":
            # strong channel-dependent backbone; CSLL's own RevIN handles normalisation
            from .itransformer import ITransformer
            self.itbackbone = ITransformer(seq_len, pred_len, n_vars, d_model=d_model,
                                           nhead=nhead, num_layers=num_layers,
                                           dim_ff=dim_ff, dropout=dropout, use_revin=False)
        else:
            self.lin_trend = nn.Linear(seq_len, pred_len)
            self.lin_seasonal = nn.Linear(seq_len, pred_len)
        if not self.direct:
            self.head = nn.Linear(seq_len, pred_len)   # direct mode forecasts by synthesis; no head

        # --- real DFT basis ---
        # pad2x: build the basis for a zero-padded transform of length M = 2L. The analysis
        # sums only over the L observed samples (the padding is zero) and the synthesis is
        # evaluated only at the first L samples, so both stay (F, L) matrices. Effect: the
        # phase ramp e^{-i w_f D} now implements a LINEAR (aperiodic) shift within the padded
        # window instead of a circular one -- delayed samples that fall off the window edge
        # land in the padding rather than wrapping around, matching physical delays.
        M = 2 * seq_len if pad2x else seq_len
        F = M // 2 + 1
        self.F = F
        t = torch.arange(seq_len).float()
        f = torch.arange(F).float()
        ang = 2.0 * math.pi * torch.outer(f, t) / M                # (F,L)
        alpha = torch.full((F,), 2.0 / M)
        alpha[0] = 1.0 / M
        if M % 2 == 0:
            alpha[F - 1] = 1.0 / M
        self.register_buffer("Cb", torch.cos(ang))
        self.register_buffer("Sb", torch.sin(ang))
        self.register_buffer("Acos", (alpha[None, :] * torch.cos(ang).transpose(0, 1)))
        self.register_buffer("Asin", (alpha[None, :] * torch.sin(ang).transpose(0, 1)))
        self.register_buffer("omega", 2.0 * math.pi * f / M)       # (F,)
        if self.direct:
            tf = torch.arange(seq_len, seq_len + pred_len).float()  # future positions in [L, L+H)
            angf = 2.0 * math.pi * torch.outer(f, tf) / M           # (F,H)
            # positions beyond the padded domain (h >= L) would wrap circularly back into the
            # window; a lead of tau can only forecast tau <= tau_max <= L steps ahead, so the
            # branch's direct forecast is defined as zero there (the gated backbone covers it).
            valid = (tf < M).float()[:, None]                       # (H,1)
            self.register_buffer("Acos_fut", valid * (alpha[None, :] * torch.cos(angf).transpose(0, 1)))
            self.register_buffer("Asin_fut", valid * (alpha[None, :] * torch.sin(angf).transpose(0, 1)))

        self.bands = _make_bands(F, n_bands)
        self.n_bands = len(self.bands)
        N = n_vars

        # --- coupling A_b ---
        if free_complex:
            self.Wr = nn.ParameterList([nn.Parameter(0.02 * torch.randn(N, N)) for _ in self.bands])
            self.Wi = nn.ParameterList([nn.Parameter(0.02 * torch.randn(N, N)) for _ in self.bands])
        elif low_rank and low_rank > 0:
            r = low_rank
            self.U = nn.ParameterList([nn.Parameter(0.02 * torch.randn(N, r)) for _ in self.bands])
            self.V = nn.ParameterList([nn.Parameter(0.02 * torch.randn(N, r)) for _ in self.bands])
        else:
            self.A = nn.ParameterList([nn.Parameter(0.02 * torch.randn(N, N)) for _ in self.bands])

        # --- additive delay positions p_b, q_b ---
        if self.use_phase:
            self.p_base = nn.ParameterList([nn.Parameter(torch.zeros(N)) for _ in self.bands])
            self.q_base = nn.ParameterList([nn.Parameter(torch.zeros(N)) for _ in self.bands])
            if self.dynamic:
                self.desc = nn.Linear(seq_len, desc_dim)         # per-channel window descriptor
                self.ctrl = nn.Sequential(nn.Linear(desc_dim, ctrl_hidden), nn.GELU(),
                                          nn.Linear(ctrl_hidden, 2 * self.n_bands))
                nn.init.zeros_(self.ctrl[-1].weight)             # start as static (offset 0)
                nn.init.zeros_(self.ctrl[-1].bias)

        if freeze_gate:
            self.register_buffer("alpha", torch.tensor(float(gate_init)))
        else:
            self.alpha = nn.Parameter(torch.tensor(float(gate_init)))

        # --- hybrid mixing branch (separate real coupling + L->H head + its own gate) ---
        if self.hybrid:
            if low_rank and low_rank > 0:
                self.Umix = nn.ParameterList([nn.Parameter(0.02 * torch.randn(N, low_rank)) for _ in self.bands])
                self.Vmix = nn.ParameterList([nn.Parameter(0.02 * torch.randn(N, low_rank)) for _ in self.bands])
            else:
                self.Amix = nn.ParameterList([nn.Parameter(0.02 * torch.randn(N, N)) for _ in self.bands])
            self.head_mix = nn.Linear(seq_len, pred_len)
            self.alpha_mix = nn.Parameter(torch.tensor(float(gate_init)))
        self.revin = RevIN(n_vars) if use_revin else None

    # ---- DFT (real arithmetic) ----
    def _rfft(self, x):
        return torch.einsum("btn,ft->bfn", x, self.Cb), -torch.einsum("btn,ft->bfn", x, self.Sb)

    def _irfft(self, Zr, Zi):
        return torch.einsum("bfn,tf->btn", Zr, self.Acos) - torch.einsum("bfn,tf->btn", Zi, self.Asin)

    def _irfft_future(self, Zr, Zi):
        """Synthesise at t in [L, L+H): the branch's direct forecast (see `direct`)."""
        return torch.einsum("bfn,tf->btn", Zr, self.Acos_fut) - torch.einsum("bfn,tf->btn", Zi, self.Asin_fut)

    def _delays(self, x):
        """Return per-band (p,q) each (B,N). Static -> broadcast base; dynamic -> base+ctrl(x)."""
        B = x.shape[0]
        if not self.use_phase:
            return None
        if self.dynamic:
            desc = self.desc(x.transpose(1, 2))              # (B,N,desc_dim)
            off = self.ctrl(desc)                            # (B,N,2*n_bands)
            p_off, q_off = off[..., :self.n_bands], off[..., self.n_bands:]
        ps, qs = [], []
        for b in range(self.n_bands):
            pb = self.p_base[b][None, :].expand(B, -1)
            qb = self.q_base[b][None, :].expand(B, -1)
            if self.dynamic:
                pb = pb + p_off[..., b]
                qb = qb + q_off[..., b]
            ps.append(self.pq_bound * torch.tanh(pb))
            qs.append(self.pq_bound * torch.tanh(qb))
        return ps, qs

    def _couple(self, b, Yr, Yi):
        if self.low_rank and self.low_rank > 0:
            Gr = torch.einsum("jr,bfj->bfr", self.V[b], Yr)
            Gi = torch.einsum("jr,bfj->bfr", self.V[b], Yi)
            return torch.einsum("ir,bfr->bfi", self.U[b], Gr), torch.einsum("ir,bfr->bfi", self.U[b], Gi)
        return torch.einsum("ij,bfj->bfi", self.A[b], Yr), torch.einsum("ij,bfj->bfi", self.A[b], Yi)

    def _mixing_corr(self, Xr, Xi):
        """Real-valued spectral cross-channel mixing (no phase) -> full-window iRFFT -> L->H
        head. The same-time coupling path of the hybrid model."""
        Zr, Zi = torch.zeros_like(Xr), torch.zeros_like(Xi)
        for b, (lo, hi) in enumerate(self.bands):
            Xr_b, Xi_b = Xr[:, lo:hi], Xi[:, lo:hi]
            if self.low_rank and self.low_rank > 0:
                Gr = torch.einsum("jr,bfj->bfr", self.Vmix[b], Xr_b)
                Gi = torch.einsum("jr,bfj->bfr", self.Vmix[b], Xi_b)
                Zr[:, lo:hi] = torch.einsum("ir,bfr->bfi", self.Umix[b], Gr)
                Zi[:, lo:hi] = torch.einsum("ir,bfr->bfi", self.Umix[b], Gi)
            else:
                Zr[:, lo:hi] = torch.einsum("ij,bfj->bfi", self.Amix[b], Xr_b)
                Zi[:, lo:hi] = torch.einsum("ij,bfj->bfi", self.Amix[b], Xi_b)
        z = self._irfft(Zr, Zi)
        return self.head_mix(z.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        if self.revin is not None:
            x = self.revin(x, "norm")
        if self.backbone_type == "itransformer":
            base = self.itbackbone(x)                       # (B,H,N), operates on normalised x
        else:
            trend, seasonal = _series_decomp(x, self.kernel_size)
            base = (self.lin_trend(trend.transpose(1, 2)) + self.lin_seasonal(seasonal.transpose(1, 2))).transpose(1, 2)

        Xr, Xi = self._rfft(x)
        Zr, Zi = torch.zeros_like(Xr), torch.zeros_like(Xi)
        delays = self._delays(x)
        for b, (lo, hi) in enumerate(self.bands):
            Xr_b, Xi_b = Xr[:, lo:hi], Xi[:, lo:hi]
            if self.free_complex:
                zr = torch.einsum("ij,bfj->bfi", self.Wr[b], Xr_b) - torch.einsum("ij,bfj->bfi", self.Wi[b], Xi_b)
                zi = torch.einsum("ij,bfj->bfi", self.Wr[b], Xi_b) + torch.einsum("ij,bfj->bfi", self.Wi[b], Xr_b)
            else:
                om = self.omega[lo:hi]                        # (Fb,)
                if delays is not None:
                    p_b, q_b = delays[0][b], delays[1][b]     # (B,N)
                    th = om[None, :, None] * q_b[:, None, :]  # (B,Fb,N) source phase
                    cq, sq = torch.cos(th), torch.sin(th)
                    Yr, Yi = cq * Xr_b - sq * Xi_b, cq * Xi_b + sq * Xr_b
                else:
                    Yr, Yi = Xr_b, Xi_b
                Sr, Si = self._couple(b, Yr, Yi)
                if delays is not None:
                    ps = om[None, :, None] * p_b[:, None, :]  # (B,Fb,N) target phase
                    cp, sp = torch.cos(ps), torch.sin(ps)
                    zr, zi = cp * Sr + sp * Si, cp * Si - sp * Sr
                else:
                    zr, zi = Sr, Si
            Zr[:, lo:hi], Zi[:, lo:hi] = zr, zi

        if self.direct:
            corr = self._irfft_future(Zr, Zi)                # (B,H,N): forecast by phase shift
        else:
            z = self._irfft(Zr, Zi)
            corr = self.head(z.transpose(1, 2)).transpose(1, 2)
        out = base + self.alpha * corr
        if self.hybrid:
            out = out + self.alpha_mix * self._mixing_corr(Xr, Xi)
        if self.revin is not None:
            out = self.revin(out, "denorm")
        return out

    def optim_groups(self, branch_wd: float):
        """Two param groups: backbone/RevIN with NO weight decay (fair vs. baselines), and the
        novel spectral branch (+correction head) with weight decay `branch_wd` to prevent it
        from over-fitting and harming datasets with weak cross-series signal."""
        backbone_ids = set()
        bb = [self.itbackbone] if self.backbone_type == "itransformer" else [self.lin_trend, self.lin_seasonal]
        for mod in bb + ([self.revin] if self.revin is not None else []):
            for p in mod.parameters():
                backbone_ids.add(id(p))
        if isinstance(self.alpha, nn.Parameter):
            backbone_ids.add(id(self.alpha))   # keep the gate free (no weight decay)
        if self.hybrid and isinstance(self.alpha_mix, nn.Parameter):
            backbone_ids.add(id(self.alpha_mix))
        backbone = [p for p in self.parameters() if id(p) in backbone_ids]
        branch = [p for p in self.parameters() if id(p) not in backbone_ids and p.requires_grad]
        return [{"params": backbone, "weight_decay": 0.0},
                {"params": branch, "weight_decay": branch_wd}]

    @torch.no_grad()
    def select_gate(self, val_loader, device, margin: float = 0.005):
        """Post-training safeguard: keep the spectral branch ONLY if it lowers the validation
        MSE by at least `margin` (relative); otherwise set the gate to 0 so the model falls back
        exactly to its backbone and cannot do worse than it. Guarantees CSLL(X) <= backbone on
        validation, which is what makes the added module safe (never harmful) in deployment."""
        if not isinstance(self.alpha, nn.Parameter):
            return
        self.eval()

        def _val_mse():
            tot, n = 0.0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = self(x)
                tot += float(((p - y) ** 2).mean()) * x.shape[0]
                n += x.shape[0]
            return tot / max(1, n)

        a = self.alpha.data.clone()
        l_full = _val_mse()
        self.alpha.data.zero_()
        l_base = _val_mse()
        if l_full < l_base * (1.0 - margin):
            self.alpha.data.copy_(a)          # branch clearly helps -> keep it
        # else: leave alpha = 0 (fall back to backbone)
        self._gate_kept = bool(float(self.alpha.data) != 0.0)

    @torch.no_grad()
    def learned_delays(self, band=0) -> torch.Tensor:
        """Static/base delay matrix D_ij = p_i - q_j (samples).

        band: int -> that band's matrix; None -> mean over all bands (the aggregate
        read-out; individual bands can park delay signal that a single-band read misses)."""
        if not self.use_phase or self.free_complex:
            return torch.zeros(self.N, self.N)
        bands = range(self.n_bands) if band is None else [band]
        mats = []
        for b in bands:
            p = self.pq_bound * torch.tanh(self.p_base[b])
            q = self.pq_bound * torch.tanh(self.q_base[b])
            mats.append(p[:, None] - q[None, :])
        return torch.stack(mats).mean(0)

    @torch.no_grad()
    def init_delays_from_xcorr(self, series, max_lag: int = None, highpass: int = None,
                               pairwise: bool = False) -> "np.ndarray":
        """Initialise the per-channel delay positions from lagged cross-correlation.

        Classic time-delay-estimation remedy: the MSE loss is oscillatory in D (period equal
        to the dominant signal period), so gradient descent from D=0 converges to the nearest
        local minimum, not the true delay. Correlation gives the coarse estimate; gradient
        training then refines it. Two passes: (1) lag each channel against an arbitrary anchor
        channel, (2) re-lag against the lag-aligned mean (removes anchor noise). Lags use the
        |xcorr| argmax, so sign-flipped couplings are handled. Estimates are mean-centred
        (the additive gauge (p,q)->(p+c,q+c) is unidentifiable) and written into p_base/q_base
        of every band via atanh. Returns the centred lag vector (numpy, in samples)."""
        import numpy as np
        if not self.use_phase:
            return None
        x = np.asarray(series, dtype=np.float64)
        T, N = x.shape
        assert N == self.N, f"series has {N} channels, model expects {self.N}"
        # High-pass (subtract a centred rolling mean) before estimating lags: strong shared
        # seasonal cycles are phase-aligned across channels and pin the xcorr peak at lag 0,
        # masking the propagation lags that live in the residual band. Window default L/4.
        hp = int(highpass) if highpass is not None else max(2, self.L // 4)
        if hp > 1:
            kern = np.ones(hp) / hp
            trend = np.apply_along_axis(lambda s: np.convolve(s, kern, mode="same"), 0, x)
            x = x - trend
        x = x - x.mean(0)
        x = x / (x.std(0) + 1e-8)
        L = int(max_lag) if max_lag is not None else int(round(self.tau_max))
        nfft = 1
        while nfft < 2 * T:
            nfft *= 2

        def lags_vs(ref):
            R = np.fft.rfft(ref, nfft)
            X = np.fft.rfft(x, nfft, axis=0)
            c = np.fft.irfft(X * np.conj(R)[:, None], nfft, axis=0)   # c[l,i] = sum_t x_i(t) ref(t-l)
            c = np.concatenate([c[-L:], c[:L + 1]], axis=0)           # lags -L..L
            return np.abs(c).argmax(0) - L                            # (N,)

        if pairwise:
            # Network mode: per-channel positions from PAIRWISE lags (sensor-network time
            # synchronisation). Global-reference alignment fails when lead-lag structure is
            # local (e.g. corridors in a road network): each channel's best lag against the
            # network mean is ~0. Instead: take each channel's top-k most-correlated partners,
            # measure the pairwise |xcorr| argmax lag l_ab, and solve the weighted least
            # squares  min_d  sum_ab w_ab (d_b - d_a - l_ab)^2  (graph-Laplacian system,
            # gauge fixed by mean-zero).
            k = min(8, N - 1)
            C = np.corrcoef(x.T)
            np.fill_diagonal(C, 0.0)
            Fx = np.fft.rfft(x, nfft, axis=0)
            rowsA, rowsB, lagsAB, wAB = [], [], [], []
            for a in range(N):
                for b in np.argsort(-np.abs(C[a]))[:k]:
                    if b <= a and a in set(np.argsort(-np.abs(C[b]))[:k]):
                        continue                       # avoid duplicating symmetric pairs
                    c = np.fft.irfft(Fx[:, b] * np.conj(Fx[:, a]), nfft)
                    c = np.concatenate([c[-L:], c[:L + 1]])
                    lag = int(np.abs(c).argmax()) - L
                    pk = float(np.abs(c).max() / T)
                    if pk > 0.15:
                        rowsA.append(a); rowsB.append(int(b))
                        lagsAB.append(float(lag)); wAB.append(pk)
            d = np.zeros(N)
            if rowsA:
                Lap = np.zeros((N, N)); rhs = np.zeros(N)
                for a, b, l, w in zip(rowsA, rowsB, lagsAB, wAB):
                    Lap[a, a] += w; Lap[b, b] += w
                    Lap[a, b] -= w; Lap[b, a] -= w
                    rhs[b] += w * l; rhs[a] -= w * l
                Lap += 1e-6 * np.eye(N)                # gauge/ridge; solution centred below
                d = np.linalg.solve(Lap, rhs)
        else:
            d0 = lags_vs(x[:, 0].copy())
            aligned = np.zeros_like(x)
            for i in range(N):                 # undo first-pass lags: aligned_i(t) = x_i(t + d0_i)
                di = int(d0[i])
                if di > 0:
                    aligned[:T - di, i] = x[di:, i]
                elif di < 0:
                    aligned[-di:, i] = x[:T + di, i]
                else:
                    aligned[:, i] = x[:, i]
            d = lags_vs(aligned.mean(1)).astype(np.float64)
        d = d - (d.min() + d.max()) / 2.0    # fix the additive gauge; midrange centring keeps
        #                                      the extreme offsets symmetric under the pq bound
        u = np.clip(d / self.pq_bound, -0.95, 0.95)                   # stay off tanh saturation
        raw = torch.tensor(np.arctanh(u), dtype=self.p_base[0].dtype)
        for b in range(self.n_bands):
            self.p_base[b].data.copy_(raw)
            self.q_base[b].data.copy_(raw)
        self._init_lags = d
        return d

    @torch.no_grad()
    def delays_for_input(self, x: torch.Tensor, band: int = 0) -> torch.Tensor:
        """Input-adaptive delay matrix averaged over a batch (for dynamic interpretability)."""
        if not self.use_phase or self.free_complex:
            return torch.zeros(self.N, self.N)
        if self.revin is not None:
            x = self.revin(x, "norm")
        d = self._delays(x)
        p, q = d[0][band].mean(0), d[1][band].mean(0)
        return p[:, None] - q[None, :]
