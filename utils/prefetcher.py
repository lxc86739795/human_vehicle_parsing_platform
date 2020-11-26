# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485*255, 0.456*255, 0.406*255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229*255, 0.224*255, 0.225*255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_is_rotated = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_is_rotated = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_is_rotated = self.next_is_rotated.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        is_rotated = self.next_is_rotated
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if is_rotated is not None:
            is_rotated.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, is_rotated

    
class data_prefetcher_mask(): # Changed by Xinchen Liu
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485*255, 0.456*255, 0.406*255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229*255, 0.224*255, 0.225*255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_mask, self.next_is_rotated = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_mask = None
            self.next_is_rotated = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_mask = self.next_mask.cuda(non_blocking=True)
            self.next_is_rotated = self.next_is_rotated.cuda(non_blocking=True)

            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        mask = self.next_mask
        is_rotated = self.next_is_rotated
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if mask is not None:
            mask.record_stream(torch.cuda.current_stream())
        if is_rotated is not None:
            is_rotated.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, mask, is_rotated

