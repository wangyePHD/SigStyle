


from typing import Any


class Controller:
    def __init__(self):
        self.self_attn_map = {}
        self.cross_attn_map = {}
        
        self.self_idx = 0
        self.cross_idx = 0
        self.cur_step = 0
        self.set_ddl_time = 0
    
    
    def get_self_attn_map_by_layer_name(self, layer_name):
        return self.self_attn_map[layer_name]
    
    def set_self_attn_map_by_layer_name(self, self_attn_map, layer_name):    
        self.self_attn_map[layer_name] = self_attn_map.cpu()
    
    def get_cross_attn_map_by_layer_name(self, layer_name):
        return self.cross_attn_map[layer_name]
    
    def set_cross_attn_map_by_layer_name(self, cross_attn_map, layer_name):
        self.cross_attn_map[layer_name] = cross_attn_map.cpu()
    

    def set_cur_step(self, cur_step):
        self.cur_step = cur_step
        
    
    def forward(self, attn_maps, layer_name, is_cross):
        if is_cross:
            self.set_cross_attn_map_by_layer_name(attn_maps, layer_name)
        else:
            self.set_self_attn_map_by_layer_name(attn_maps, layer_name)
        
    def __call__(self, attn_maps, layer_name, is_cross):
        self.forward(attn_maps, layer_name, is_cross)