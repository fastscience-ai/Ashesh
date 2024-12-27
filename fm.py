from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def expand_t(t, x):
    for _ in range(x.ndim - 1):
        t = t.unsqueeze(-1)
    return t


class FMScheduler(nn.Module):
    def __init__(self, num_train_timesteps=1000, sigma_min=0.001):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.sigma_min = sigma_min

    def uniform_sample_t(self, batch_size) -> torch.LongTensor:
        ts = (
            np.random.choice(np.arange(self.num_train_timesteps), batch_size)
            / self.num_train_timesteps
        )
        return torch.from_numpy(ts)

    def compute_psi_t(self, x1, t, x):
        """
        Compute the conditional flow psi_t(x | x_1).

        Note that time flows in the opposite direction compared to DDPM/DDIM.
        As t moves from 0 to 1, the probability paths shift from a prior distribution p_0(x)
        to a more complex data distribution p_1(x).

        Input:
            x1 (`torch.Tensor`): Data sample from the data distribution.
            t (`torch.Tensor`): Timestep in [0,1).
            x (`torch.Tensor`): The input to the conditional psi_t(x).
        Output:
            psi_t (`torch.Tensor`): The conditional flow at t.
        """
        t = expand_t(t, x1)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute psi_t(x)
        assert t.any() >= 0.0 and t.any() <= 1.0, f"t should be in [0, 1). Got {t}"

        mu_t = x1 * t 
        #print(mu_t.shape)
        sigma_t = 1 - (1 - self.sigma_min) * t
        psi_t = mu_t + sigma_t * x

        ######################

        return psi_t

    def step(self, xt, vt, dt):
        """
        The simplest ode solver as the first-order Euler method:
        x_next = xt + dt * vt
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        # implement each step of the first-order Euler method.
        #print(xt.shape, vt.shape, dt.shape)
        x_next = xt + dt.unsqueeze(-1) * vt
        ######################

        return x_next


class FlowMatching(nn.Module):
    def __init__(self, network: nn.Module, fm_scheduler: FMScheduler, **kwargs):
        super().__init__()
        self.network = network
        self.fm_scheduler = fm_scheduler

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    def get_loss(self, x1, class_label=None, x0=None):
        """
        The conditional flow matching objective, corresponding Eq. 23 in the FM paper.
        """
        batch_size = x1.shape[0]
        t = self.fm_scheduler.uniform_sample_t(batch_size).to(x1)
        if x0 is None:
            x0 = torch.randn_like(x1)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Implement the CFM objective.
        input_x = self.fm_scheduler.compute_psi_t(x1, t, x0)
        if class_label is not None:
            model_out = self.network(input_x, t, class_label=class_label)
        else:
            model_out = self.network(input_x, t)

        epsilon = 1e-5
        denominator = 1 - t + epsilon
        #print(x1.shape, x0.shape, t.shape)
        true_u_t = (x1 - (1 - self.fm_scheduler.sigma_min) * x0) #/ denominator.unsqueeze(-1)

        loss = F.mse_loss(model_out, true_u_t)
        ######################

        return loss

    def conditional_psi_sample(self, x1, t, x0=None):
        if x0 is None:
            x0 = torch.randn_like(x1)
        return self.fm_scheduler.compute_psi_t(x1, t, x0)

    #@torch.no_grad()
    def sample(
        self,
        shape,
        num_inference_timesteps=50,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 1.0,
        verbose=False,
    ):
        batch_size = shape[0]
        x_T = torch.randn(shape).to(self.device)
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            assert class_label is not None
            assert (
                len(class_label) == batch_size
            ), f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"

        traj = [x_T]

        timesteps = [
            i / num_inference_timesteps for i in range(num_inference_timesteps)
        ]
        timesteps = [torch.tensor([t] * x_T.shape[0]).to(x_T) for t in timesteps]
        #print("timesteps : ", timesteps)
        
        pbar = tqdm(timesteps) if verbose else timesteps
        xt = x_T
        for i, t in enumerate(pbar):
            t_next = timesteps[i + 1] if i < len(timesteps) - 1 else torch.ones_like(t)
            

            ######## TODO ########
            # Complete the sampling loop
            dt = (t_next - t  )

            if do_classifier_free_guidance:
                uncond_class_label = torch.zeros_like(class_label)
                u_t_uncond = self.network(xt, t, class_label=uncond_class_label)
                u_t_cond = self.network(xt, t, class_label=class_label)
                # guidence scaled u_t
                u_t = u_t_uncond + guidance_scale * (u_t_cond - u_t_uncond)
            else:
                if class_label is not None:
                    u_t = self.network(xt, t, class_label=class_label)
                else:
                    u_t = self.network(xt, t)

            # update state
            # print(xt.shape, u_t.shape, dt.shape)
            xt = self.fm_scheduler.step(xt, u_t, dt)

            ######################

            traj[-1] = traj[-1].cpu()
            traj.append(xt.clone().detach())
        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "fm_scheduler": self.fm_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.fm_scheduler = hparams["fm_scheduler"]

        self.load_state_dict(state_dict)
