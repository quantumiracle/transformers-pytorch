import math
import torch


def create_targets_naive(images, batch_t, n_T, device):
    x_0 = torch.randn_like(images).to(device)
    x_1 = images
    x_t = (1 - (1 - 1e-5) * batch_t) * x_0 + batch_t * x_1
    v_t = x_1 - (1 - 1e-5) * x_0

    dt_flow = int(math.log2(n_T))
    dt_base = (torch.ones(images.shape[0], dtype=torch.int32) * dt_flow).to(device)

    return x_t, v_t, dt_base


# create batch, consisting of different timesteps and different dts(depending on total step sizes)
def create_targets(images, batch_t, labels, context_mask, model, n_T, device, bootstrap_every=8):
    current_batch_size = images.shape[0]

    # 1. create step sizes dt
    bootstrap_batch_size = current_batch_size // bootstrap_every
    log2_sections = int(math.log2(n_T))

    dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections), bootstrap_batch_size // log2_sections)
    # print(f"dt_base: {dt_base}")

    dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batch_size-dt_base.shape[0],)]).to(device)
    # print(f"dt_base: {dt_base}")
    
    dt = 1 / (2 ** (dt_base)) # [1, 1/2, 1/8, 1/16, 1/32]
    # print(f"dt: {dt}")

    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2 # [0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 0.5]
    # print(f"dt_bootstrap: {dt_bootstrap}")

    # 2. sample timesteps t
    dt_sections = 2**dt_base

    # print(f"dt_sections: {dt_sections}")

    t = torch.cat([
        torch.randint(low=0, high=int(val.item()), size=(1,)).float()
        for val in dt_sections
        ]).to(device)
    
    t = t / dt_sections
    t_full = t[:, None, None, None]

    # 3. generate bootstrap targets:
    x_1 = images[:bootstrap_batch_size]
    x_0 = torch.randn_like(x_1)

    # get dx at timestep t
    x_t = (1 - (1-1e-5) * t_full)*x_0 + t_full*x_1

    bst_labels = labels[:bootstrap_batch_size]


    with torch.no_grad():
        v_b1 = model(x_t, t, dt_base_bootstrap, bst_labels)

    t2 = t + dt_bootstrap
    x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
    x_t2 = torch.clip(x_t2, -4, 4)
    
    with torch.no_grad():
        v_b2 = model(x_t2, t2, dt_base_bootstrap, bst_labels)

    v_target = (v_b1 + v_b2) / 2

    v_target = torch.clip(v_target, -4, 4)
    
    bst_v = v_target
    bst_dt = dt_base
    bst_xt = x_t

    # 4. generate flow-matching targets
    # sample t(normalized)
    # sample flow pairs x_t, v_t
    x_0 = torch.randn_like(images).to(device)
    x_1 = images
    x_t = (1 - (1 - 1e-5) * batch_t) * x_0 + batch_t * x_1
    v_t = x_1 - (1 - 1e-5) * x_0

    dt_flow = int(math.log2(n_T))
    dt_base = (torch.ones(images.shape[0], dtype=torch.int32) * dt_flow).to(device)

    # 5. merge flow and bootstrap
    bst_size_data = current_batch_size - bootstrap_batch_size

    x_t = torch.cat([bst_xt, x_t[-bst_size_data:]], dim=0)
    t = torch.cat([t_full, batch_t[-bst_size_data:]], dim=0)
    dt_base = torch.cat([bst_dt, dt_base[-bst_size_data:]], dim=0)
    v_t = torch.cat([bst_v, v_t[-bst_size_data:]], dim=0)

    # set context mask to 1 for first bootstrap_batch_size samples
    unmask = torch.ones(context_mask[:bootstrap_batch_size].shape).to(context_mask.device)
    context_mask = torch.cat([unmask, context_mask[-bst_size_data:]], dim=0)

    return x_t, v_t, t, dt_base, context_mask



    
