import torch

def build_optimizer(optim_params, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (dict): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    optimizer_cfg = cfg['OPTIMIZER']
    if optimizer_cfg['OPTIMIZER_METHOD'] == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=optimizer_cfg['BASE_LR'],
            momentum=optimizer_cfg['MOMENTUM'],
            # dampening=optimizer_cfg['DAMPENING'],
            # weight_decay=optimizer_cfg['WEIGHT_DECAY'],
            # nesterov=optimizer_cfg['NESTEROV'],
        )
    elif optimizer_cfg['OPTIMIZER_METHOD'] == "rmsprop":
        return torch.optim.RMSprop(
            optim_params,
            lr=optimizer_cfg['BASE_LR'],
            alpha=optimizer_cfg['ALPHA'],
            eps=optimizer_cfg['EPS'],
            weight_decay=optimizer_cfg['WEIGHT_DECAY'],
            momentum=optimizer_cfg['MOMENTUM'],
        )
    elif optimizer_cfg['OPTIMIZER_METHOD'] == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=optimizer_cfg['BASE_LR'],
            betas=optimizer_cfg['ADAM_BETAS'],
            eps=optimizer_cfg['EPS'],
            weight_decay=optimizer_cfg['WEIGHT_DECAY'],
            amsgrad=optimizer_cfg['AMSGRAD'],
        )
    elif optimizer_cfg['OPTIMIZER_METHOD'] == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=optimizer_cfg['BASE_LR'],
            betas=optimizer_cfg['ADAM_BETAS'],
            eps=optimizer_cfg['EPS'],
            weight_decay=optimizer_cfg['WEIGHT_DECAY'],
            amsgrad=optimizer_cfg['AMSGRAD'],
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(
                optimizer_cfg['OPTIMIZER_METHOD']
            )
        )
