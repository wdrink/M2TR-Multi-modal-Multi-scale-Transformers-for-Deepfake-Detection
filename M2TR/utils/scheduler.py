
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.multistep_lr import MultiStepLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler


def build_scheduler(optimizer, cfg):
    num_epochs = cfg['TRAIN']['MAX_EPOCH']
    scheduler_cfg = cfg['SCHEDULER']

    if 'LR_NOISE' in scheduler_cfg:
        lr_noise = scheduler_cfg['LR_NOISE']
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=scheduler_cfg['LR_NOISE_PCT']
        if 'LR_NOISE_PCT' in scheduler_cfg
        else 0.67,
        noise_std=scheduler_cfg['LR_NOISE_STD']
        if 'LR_NOISE_STD' in scheduler_cfg
        else 1.0,
        noise_seed=scheduler_cfg['SEED']
        if 'SEED' in scheduler_cfg
        else 42,
    )
    cycle_args = dict(
        cycle_mul=scheduler_cfg['LR_CYCLE_MUL']
        if 'LR_CYCLE_MUL' in scheduler_cfg
        else 1.0,
        cycle_decay=scheduler_cfg['LR_CYCLE_DECAY']
        if 'LR_CYCLE_DECAY' in scheduler_cfg
        else 0.1,
        cycle_limit=scheduler_cfg['LR_CYCLE_LIMIT']
        if 'LR_CYCLE_LIMIT' in scheduler_cfg
        else 1,
    )

    lr_scheduler = None

    if scheduler_cfg['SCHEDULER_TYPE'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=scheduler_cfg['MIN_LR'],
            warmup_lr_init=scheduler_cfg['WARMUP_LR'],
            warmup_t=scheduler_cfg['WARMUP_EPOCHS'],
            k_decay=scheduler_cfg['LR_K_DECAY']
            if 'LR_K_DECAY' in scheduler_cfg
            else 1.0,
            **cycle_args,
            **noise_args,
        )
        num_epochs = (
            lr_scheduler.get_cycle_length() + scheduler_cfg['COOLDOWN_EPOCHS']
        )

    elif scheduler_cfg['SCHEDULER_TYPE'] == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=scheduler_cfg['MIN_LR'],
            warmup_lr_init=scheduler_cfg['WARMUP_LR'],
            warmup_t=scheduler_cfg['WARMUP_EPOCHS'],
            t_in_epochs=True,
            **cycle_args,
            **noise_args,
        )
        num_epochs = (
            lr_scheduler.get_cycle_length() + scheduler_cfg['COOLDOWN_EPOCHS']
        )
    elif scheduler_cfg['SCHEDULER_TYPE'] == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=scheduler_cfg['DECAY_EPOCHS'],
            decay_rate=scheduler_cfg['DECAY_RATE'],
            warmup_lr_init=scheduler_cfg['WARMUP_LR'],
            warmup_t=scheduler_cfg['WARMUP_EPOCHS'],
            **noise_args,
        )
    elif scheduler_cfg['SCHEDULER_TYPE'] == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=scheduler_cfg['DECAY_EPOCHS'],
            decay_rate=scheduler_cfg['DECAY_RATE'],
            warmup_lr_init=scheduler_cfg['WARMUP_LR'],
            warmup_t=scheduler_cfg['WARMUP_EPOCHS'],
            **noise_args,
        )

    return lr_scheduler, num_epochs
