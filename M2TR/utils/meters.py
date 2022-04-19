import datetime
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.distributed as dist
from fvcore.common.timer import Timer

import M2TR.utils.distributed as du
import M2TR.utils.logging as logging
from sklearn.metrics import roc_auc_score

logger = logging.get_logger(__name__)

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not du.is_dist_avail_and_initialized():
            return
        t = torch.tensor(
            [self.count, self.total], dtype=torch.float64, device='cuda'
        )
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class AucMetric():
    """
    
    """

    def __init__(self, num_gpus):
        self.labels = torch.Tensor().cuda()
        self.preds = torch.Tensor().cuda()
        self.num_gpus = num_gpus

    def update(self, labels, preds):
        self.labels = torch.cat([self.labels, labels], dim=0)
        self.preds = torch.cat([self.preds, preds], dim=0)

    def synchronize_between_processes(self):
        if not du.is_dist_avail_and_initialized():
            return
        labels = [torch.zeros(len(self.labels), dtype=self.labels[0].dtype).cuda() for _ in range(self.num_gpus)]
        preds = [torch.zeros(len(self.preds), dtype=self.preds[0].dtype).cuda() for _ in range(self.num_gpus)]
        dist.all_gather(labels, self.labels)
        dist.all_gather(preds, self.preds)
        labels = torch.cat(labels, dim=0).cpu()
        preds = torch.cat(preds, dim=0).cpu()
        self.auc = roc_auc_score(labels, preds)

    # def __str__(self):
    #     return str(self.auc)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr
            )
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(
            '{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)
            )
        )


class EpochTimer:
    """
    A timer which computes the epoch time.
    """

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer.
        """
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset()

    def epoch_toc(self):
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)
