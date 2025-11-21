import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pos_utils import DeformNetwork, DeformNetwork_real_only, DeformNetwork_HumanSimple
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func

class DeformModel:
    def __init__(self, is_blender=True, is_6dof=False, train_man = False):
        self.train_man = train_man
        self.deform = None
        self.deform_man = None
        if not train_man:
            self.deform = DeformNetwork_real_only(is_blender=is_blender, is_6dof=is_6dof).cuda()
        if train_man:
            self.deform_man = DeformNetwork_HumanSimple().cuda()
            
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, rx_input):
        if self.deform is not None:
            raw = self.deform(xyz, rx_input)
        return raw

    def step_man(self, xyz, rx_pos, man_pos, man_orient):
        man_info = torch.cat([man_pos, man_orient], dim=-1)
        if self.deform_man is not None:
            return self.deform_man(xyz, rx_pos, man_info)


    def train_setting(self, training_args):
        if not self.train_man :
            l = [
                {'params': list(self.deform.parameters()),
                'lr': training_args.position_lr_init * self.spatial_lr_scale,
                "name": "deform"}
            ]
        else : 
            l = [
                {'params': list(self.deform_man.parameters()),
                'lr': training_args.position_lr_init * self.spatial_lr_scale,
                "name": "deform_man"}
            ]
            
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.deform_lr_final,
                                                       lr_delay_steps=training_args.deform_lr_delay_steps,
                                                       lr_delay_mult=training_args.deform_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        if not self.train_man:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "deform":
                    lr = self.deform_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr
        
        else:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "deform_man":
                    lr = self.deform_scheduler_args(iteration) 
                    param_group['lr'] = lr
                    return lr
            
    def parameters(self):
        if not self.train_man:
            return list(self.deform.parameters())
        else:
            return list(self.deform_man.parameters())
        