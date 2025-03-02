import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
from einops import rearrange, repeat
from torch import nn

import torch.nn.functional as F

def register_attention_control(model, controller, flag=True):
    
    
    def ca_forward(self, place_in_unet=None):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask
            
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            
            if flag:
                # if is_cross:
                #     layer_name = "cross_attn" + "_"+str(controller.cur_step)+"_"+ str(controller.cross_idx)
                #     controller.cross_idx += 1
                    
                # else:
                if not is_cross:
                    layer_name = "self_attn" + "_"+str(controller.cur_step)+"_"+ str(controller.self_idx)
                    controller.self_idx += 1
                    if controller.cur_step < controller.set_ddl_time:
                        controller(q, layer_name, is_cross)
            else:
                pass      
            
            # if q.shape[0]>2 and not is_cross and 0 <= controller.cur_step <= int(controller.self_output_replace_steps * 50):
            #     q[1, :, :] = q[0, :, :]
            #     q[3, :, :] = q[2, :, :]
            #     k[1, :, :] = k[0, :, :]
            #     k[3, :, :] = k[2, :, :]
            #     v[1, :, :] = v[0, :, :]
            #     v[3, :, :] = v[2, :, :]

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
                  
            
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return to_out(out)

        return forward

  
    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    
    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count




def register_attention_control_ostaf(model, controller,time):
    
    def ca_forward(self):
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            swap_controller = controller
            
            weight_wo = 1.0
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            
            query = F.linear(hidden_states, self.to_q.weight * (1 + weight_wo * self.wo_q()), bias=self.to_q.bias)
            
            
            
            flag_cross_attn = True
            if encoder_hidden_states is None:
                flag_cross_attn = False
                encoder_hidden_states = hidden_states
            elif self.cross_attention_norm:
                encoder_hidden_states = self.norm_cross(encoder_hidden_states)
                
            if flag_cross_attn:
                layer_name = "cross_attn" + "_"+str(controller.cur_step)+"_"+ str(controller.cross_idx)
                controller.cross_idx += 1
            else:
                layer_name = "self_attn" + "_"+str(controller.cur_step)+"_"+ str(controller.self_idx)
                controller.self_idx += 1
                
            if not flag_cross_attn:
                if swap_controller.cur_step < time:
                # self attention map 替换
                    # replace_self_attn_map = swap_controller.self_attn_map[layer_name].to(attention_probs.device)
                    q_swap = swap_controller.self_attn_map[layer_name].to(query.device)
                    # import ipdb; ipdb.set_trace()
                    # attention_probs[attention_probs.shape[0]//2:, :, :] = replace_self_attn_map
                    query[query.shape[0]//2:, :, :] = q_swap
                
            
            key = F.linear(encoder_hidden_states, self.to_k.weight * (1 + weight_wo * self.wo_k()), bias=self.to_k.bias)
            value = F.linear(encoder_hidden_states, self.to_v.weight * (1 + weight_wo * self.wo_v()), bias=self.to_v.bias)
            
            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)
            
            attention_probs = self.get_attention_scores(query, key, attention_mask)

            
            self.attn_map = query @ key.transpose(-2, -1).softmax(dim=-1)
          
            
            
        
            
            
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            
            # if not flag_cross_attn:
            #     setattr(self, "self_hidden_states", hidden_states)

            return hidden_states
        
        return forward
    
    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")