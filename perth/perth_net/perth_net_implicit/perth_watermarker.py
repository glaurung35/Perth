import torch
import numpy as np
from librosa import resample

from .model.perth_net import PerthNet
from .. import PREPACKAGED_MODELS_DIR
from perth.watermarker import WatermarkerBase


def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.copy())
    return x.to(dtype=torch.float, device=device)


class PerthImplicitWatermarker(WatermarkerBase):
    def __init__(self, run_name:str="implicit", models_dir=PREPACKAGED_MODELS_DIR,
                 device="cpu", perth_net=None):
        assert (run_name is None) or (perth_net is None)
        if perth_net is None:
            self.perth_net = PerthNet.load(run_name, models_dir).to(device)
        else:
            self.perth_net = perth_net.to(device)

    def apply_watermark(self, signal,  sample_rate, **_):
        change_rate = sample_rate != self.perth_net.hp.sample_rate
        signal = resample(signal, orig_sr=sample_rate, target_sr=self.perth_net.hp.sample_rate) if change_rate \
            else signal

        # split signal into magnitude and phase
        signal = _to_tensor(signal, self.perth_net.device)
        magspec, phase = self.perth_net.ap.signal_to_magphase(signal)

        # encode the watermark
        magspec = magspec[None].to(self.perth_net.device)
        wm_magspec, _mask = self.perth_net.encoder(magspec)
        wm_magspec = wm_magspec[0]

        # assemble back into watermarked signal
        wm_signal = self.perth_net.ap.magphase_to_signal(wm_magspec, phase)
        wm_signal = wm_signal.detach().cpu().numpy()
        return resample(wm_signal, orig_sr=self.perth_net.hp.sample_rate, target_sr=sample_rate) if change_rate \
            else wm_signal

    def get_watermark(self, wm_signal, sample_rate, round=True, **_):
        change_rate = sample_rate != self.perth_net.hp.sample_rate
        if change_rate:
            wm_signal = resample(wm_signal, orig_sr=sample_rate, target_sr=self.perth_net.hp.sample_rate,
                                 res_type="polyphase")
        wm_signal = _to_tensor(wm_signal, self.perth_net.device)
        wm_magspec, _phase = self.perth_net.ap.signal_to_magphase(wm_signal)
        wm_magspec = wm_magspec.to(self.perth_net.device)
        wmark_pred = self.perth_net.decoder(wm_magspec[None])[0]
        wmark_pred = wmark_pred.clip(0., 1.)
        wmark_pred = wmark_pred.round() if round else wmark_pred
        return wmark_pred.detach().cpu().numpy()

    def remove_watermark(self, wm_signal, sample_rate, round=True, method='fixed_point', steps=None, **_):
        change_rate = sample_rate != self.perth_net.hp.sample_rate
        if change_rate:
            wm_signal = resample(wm_signal, orig_sr=sample_rate, target_sr=self.perth_net.hp.sample_rate,
                                 res_type="polyphase")
        wm_signal = _to_tensor(wm_signal, self.perth_net.device)
        wm_magspec, phase = self.perth_net.ap.signal_to_magphase(wm_signal)
        wm_magspec = wm_magspec.to(self.perth_net.device)

        if method == 'fixed_point':
            if steps is None:
                # change to do more fixed-point iterations if needed, 1 seems enough
                steps = 1
            for i in range(steps):
                # encode the watermark again to get direction
                wm2_magspec, _mask = self.perth_net.encoder(wm_magspec[None])
                wm2_magspec = wm2_magspec[0]
                wmark_dir = wm2_magspec - wm_magspec

                # run decoder to get amount
                wmark_pred = self.perth_net.decoder(wm_magspec[None])[0]
                wmark_pred = wmark_pred.clip(0., 1.)

                # go the opposite direction
                wm_magspec -= wmark_pred * wmark_dir
        elif method == 'adversarial':
            if steps is None:
                # change to do more adversarial iterations if needed, 5 seems enough
                steps = 5

            wm_magspec_orig = wm_magspec.clone()
            wm_magspec.requires_grad = True
            
            optimizer = torch.optim.Adam([ wm_magspec ], lr=1e-3)

            for i in range(steps):
                # XXX: could add augments here
                # to make adversarial sample robust to new attacks

                loss = self.perth_net.decoder(wm_magspec[None])[0]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 
        # assemble back into unwatermarked signal
        wm_signal = self.perth_net.ap.magphase_to_signal(wm_magspec, phase)
        wm_signal = wm_signal.detach().cpu().numpy()
        return resample(wm_signal, orig_sr=self.perth_net.hp.sample_rate, target_sr=sample_rate) if change_rate \
            else wm_signal
