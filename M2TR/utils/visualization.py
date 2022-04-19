from torch.utils.tensorboard import SummaryWriter

import M2TR.utils.distributed as du

class TensorBoardWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix=''):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        print('h')
        self.is_master_proc = du.is_master_proc(du.get_world_size())
        print('hereeee')
        print(self.is_master_proc)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False):
        if self.is_master_proc:
            super().add_scalar(tag, scalar_value, global_step, walltime, new_style, double_precision)
