from .functions import * 
from .args import get_parser

parser = get_parser()
args = parser.parse_args([])

# # Define beta schedule
# T = args.timesteps
# betas = linear_beta_schedule(timesteps=T)
# # Pre-calculate different terms for closed form
# alphas = 1. - betas
# alphas_cumprod = torch.cumprod(alphas, axis=0)
# alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
# posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# def sample_one_step(model, x_noisy, condition, t, temp_cond):

#     with torch.no_grad():
#         x_noisy = x_noisy.to(device)
#         condition = condition.to(device)
#         t = t.to(device)
#         temp_cond = temp_cond.to(device)

#         # (x_noisy-noise_pred) = label_batch 
#         # model_pred = X(K+1, 0) - X(K, t)
#         noise_pred = model(x_noisy, condition, t, temp_cond)
#         # X(K, t) - model_pred =X(K+1, 0)
#         # X(K, t) - X(K+1,0) = model_pred = -dX 
#         # will be revised: X(K+1, 0) - X(K, t) = model_pred = dX 
#         return x_noisy + noise_pred # 
#         # return x_noisy + noise_pred # TODO: Sign should be considered

cell_vector = torch.tensor([[[21.04, 0.0, 0.0], [0.0, 21.04, 0.0], [0.0, 0.0, 21.04]]])

def sample_one_step_unified(model, 
        loss_definition: str,
        X_noisy: torch.Tensor, 
        X_0: torch.Tensor, 
        t: torch.Tensor, 
        temp_cond: torch.Tensor,
        lattice: torch.Tensor):

  
        with torch.no_grad():
            x_noisy = x_noisy.to(device)
            X_noisy = X_noisy.to(device)
            X_0 = X_0.to(device)
            t = t.to(device)
            temp_cond = temp_cond.to(device)

            # noise_pred = model(x_noisy, x_0, t, temps)
            # difference = label_batch - x_0
            # model_pred = X(K+1, 0) - X(K, 0)

        match loss_definition:
            
            case 'one_step':
                
                X_t_orig = X_noisy[..., :3] # X(K,t) positoin

            case 'one_step_diff':
                
                X_t_orig = X_0[..., :3] # X(K,0) positoin
                
            case _:
                
                raise ValueError(f"Invalid loss definition: {loss_definition}")
                
        
        X_next_pred, dX, V_pred, F_pred = model_pbc_call(model,
                            X_t_orig,
                            X_noisy,
                            X_0,
                            t,
                            temp_cond,
                            lattice)
        
        # noiseX_next_pred # _pred = +dX
        return X_next_pred #condition + noise_pred # X + dx
        

def sample_next_frame(model, x_noisy, condition, tt, temp_cond):

    x_noisy = x_noisy.to(device)
    condition = condition.to(device)
    n_frames = x_noisy.shape[0]
    tt = tt.to(device)

    x_prev = x_noisy
    
    
    for t in range(tt[0].item(), -1, -1):
        t_tensor = torch.tensor([t], device=device).long()
        t_tensor = t_tensor.repeat(n_frames)
        noise_pred = model(x_noisy, condition, t_tensor, temp_cond)
        
        # Calculate the mean
        pred_mean = sqrt_recip_alphas[t] * \
                    (x_prev - betas[t] / sqrt_one_minus_alphas_cumprod[t] * noise_pred)
        
        # Add noise only for t > 0
        if t > 0:
            posterior_variance = betas[t] * (1 - alphas_cumprod_prev[t]) / (1 - alphas_cumprod[t])
            noise = torch.randn_like(x_prev) * 0.01
            x_prev = pred_mean + torch.sqrt(posterior_variance) * noise
        else:
            x_prev = pred_mean

    return x_prev