import lyngor as lyn
import torch
import sys

# /home/lynxi/Codes/mappo/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py
sys.path.append('../..')
from algorithms.r_mappo.algorithm.r_actor_critic import *

checkpoint = '/home/lynxi/Codes/mappo/onpolicy/lynxi/results/MPE/simple_reference/rmappo/check/run1/models/actor.pt'


if __name__ == "__main__":
    
    #1. 创建一个待加速的计算图（或对训练好的模型进行转换得到计算图）
    model = lyn.DLModel()
    model.load(checkpoint, model_type='Pytorch', inputs_dict={'input':(21,)})
    
    # #2. 创建一个Builder来编译计算图，并保存
    # offline_builder = lyn.Builder(target='apu')
    # out_path = offline_builder.build(model.graph, model.params, out_path='./lyn_actor')

    # #3. 直接Load即可得到runtime引擎
    # r_engine = lyn.load(path=out_path + "/Net_0/", device=0)

    