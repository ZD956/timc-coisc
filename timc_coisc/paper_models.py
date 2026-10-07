"""Thin model adapters; Proposed uses the original V12B class unchanged."""
from __future__ import annotations
import torch
from torch import nn
import paper_data  # noqa: F401
from model_structural_ablation import MultiHeaterStructuralLSTM, RAW_POWER_FULL_SCALE


def rollout(model, history, future):
    predictions = []
    state = history
    for k in range(future.shape[1]):
        value = model.forward_one_step(state, future[:, k])
        predictions.append(value)
        state = torch.cat([state[:, 1:], torch.cat([value[:, :6], future[:, k]], dim=1)[:, None]], dim=1)
    return torch.stack(predictions, dim=1)


class NormalizedModel(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.register_buffer('temperature_mean', torch.tensor(stats['temperature_mean'], dtype=torch.float32))
        self.register_buffer('temperature_std', torch.tensor(stats['temperature_std'], dtype=torch.float32))
        self.register_buffer('power_full_scale', torch.tensor(RAW_POWER_FULL_SCALE))

    def inputs(self, history, future):
        return torch.cat([(history[..., :6]-self.temperature_mean)/self.temperature_std,
                          history[..., 6:]/self.power_full_scale], dim=2), future/self.power_full_scale

    @staticmethod
    def anchor(history):
        last = history[:, -1, :6]
        return torch.cat([last, last.mean(dim=1, keepdim=True)], dim=1)

    def rollout_native_120s(self, history, future):
        return rollout(self, history, future)


class ARXRidge(NormalizedModel):
    def __init__(self, stats):
        super().__init__(stats)
        self.head = nn.Linear(20*12+6, 7)

    def features(self, history, future):
        h, f = self.inputs(history, future)
        return torch.cat([h.flatten(1), f], dim=1)

    def forward_one_step(self, history, future):
        return self.anchor(history) + self.head(self.features(history, future))


class NARXMLP(ARXRidge):
    def __init__(self, stats):
        super().__init__(stats)
        self.head = nn.Sequential(nn.Linear(246,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,7))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)


class VanillaLSTM(NormalizedModel):
    def __init__(self, stats):
        super().__init__(stats)
        self.encoder = nn.LSTM(12,128,batch_first=True)
        self.head = nn.Sequential(nn.Linear(134,128),nn.GELU(),nn.Linear(128,7))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward_one_step(self, history, future):
        h, f = self.inputs(history, future)
        _, (hidden, _) = self.encoder(h)
        return self.anchor(history) + self.head(torch.cat([hidden[-1],f],dim=1))


class UnstructuredV12B(MultiHeaterStructuralLSTM):
    def __init__(self, stats):
        super().__init__(stats)
        # Retain the exact multi-scale encoders, attention, and state encoder.
        # Remove all lag/heating/cooling heads and Tp convex/bounded anchor.
        for name in ('output_heads','residual_heat_heads','cooling_heads','tp_anchor_logits','tp_offset_head'):
            delattr(self, name)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(48+40+20+6,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1)) for _ in range(7)])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _forward_with_initial_state(self, history, future_step, initial_state):
        temp, uh, aux = self.normalized_components(history)
        contexts, _ = self.representations(temp, aux)
        common = torch.cat([initial_state,uh,future_step/self.power_full_scale],dim=1)
        residual = torch.cat([head(torch.cat([contexts[:,i],common],dim=1)) for i,head in enumerate(self.heads)],dim=1)
        return NormalizedModel.anchor(history) + residual


MODEL_IDS = ['P','B4','A1','A2','A3','A4','A5','B3','B2','B1']
NAMES = dict(P='Proposed-V12B',B1='ARX-Ridge',B2='NARX-MLP',B3='Vanilla-LSTM-1S',B4='Vanilla-LSTM-MS',A1='NoMS',A2='Unstructured',A3='NoPair',A4='NoOrder',A5='NoCooling')


def model_config(model_id):
    return dict(model_id=model_id,name=NAMES[model_id],
                multistep=model_id not in ('B1','B2','B3','A1'),
                paired=model_id != 'A3',
                order=model_id in ('P','A2','A5'),
                cooling=model_id != 'A5',
                initialization='independent random seed, no historical checkpoint',
                A1_note='strict one-step only: recursive order regularizer also disabled' if model_id=='A1' else None)


def build_model(model_id, stats):
    if model_id=='B1': return ARXRidge(stats)
    if model_id=='B2': return NARXMLP(stats)
    if model_id in ('B3','B4'): return VanillaLSTM(stats)
    if model_id=='A2': return UnstructuredV12B(stats)
    return MultiHeaterStructuralLSTM(stats)


def grid_order_loss(model, history, future):
    """Exact V12B 22-level loss, vectorized over candidates (no stochastic layers)."""
    from structural_training_common import UH_LEVELS_W
    b = len(history)
    h = history[:,None].expand(-1,22,-1,-1).reshape(b*22,20,12)
    u = future[:,None].expand(-1,22,-1,-1).clone()
    u[:,:,:,1] = torch.tensor(UH_LEVELS_W,device=future.device,dtype=future.dtype)[None,:,None]*5
    y = rollout(model,h,u.reshape(b*22,12,6)).reshape(b,22,12,7)
    return torch.relu(-(y[:,1:]-y[:,:-1])).mean()
