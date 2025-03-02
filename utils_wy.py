import os
import shutil
import json 
import time

from datetime import datetime

import subprocess

def save_pip_list(output_file):
    # 使用subprocess运行命令，并将输出重定向到指定文件
    with open(output_file, 'w') as file:
        subprocess.run(['pip', 'list'], stdout=file, text=True)



def backup_files(source_folder, backup_folder, extensions, args):
    """
    Backs up files from a source folder to a backup folder.
    
    Args:
        source_folder (str): The path to the folder containing the files to be backed up.
        backup_folder (str): The path to the folder where the backup files will be stored.
        extensions (list): A list of file extensions that should be backed up.
        args (object): An object containing additional arguments.
        
    Returns:
        None
    """
   
    # * 备份代码
    for foldername, _, filenames in os.walk(source_folder):
        if "Output" in foldername:
            continue
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                file_path = os.path.join(foldername, filename)
                relative_path = os.path.relpath(file_path, source_folder)
                backup_path = os.path.join(backup_folder, relative_path)

                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copyfile(file_path, backup_path)
          

def backup_params(backup_folder, args):
              
    # * 备份参数
    file_path = os.path.join(backup_folder, 'config_params.json')  # 可以修改文件名或路径
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=4)  # 设置 ensure_ascii=False 以保存非 ASCII 字符，indent 参数可选用于美化格式



def obtain_time():
     # 获取当前时间
    current_time = datetime.now()

    # 格式化时间为指定格式
    formatted_time = current_time.strftime('%Y-%m-%d-%H.%M')
    return formatted_time


def time_info(start_time, current_step, total_steps):
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    if current_step==0:
        estimated_remaining_time = 0
    else:
        estimated_remaining_time = elapsed_time / current_step * (total_steps - current_step)
    
    elapsed_time = format_time(elapsed_time)
    estimated_remaining_time = format_time(estimated_remaining_time)
    
    return elapsed_time, estimated_remaining_time

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = int(seconds % 60)
    
    return f"{hours}h-{minutes}min-{seconds}s"
    # return "{:.2f}h".format(hours)

# start_time = time.time()
# current_step = 4
# total_steps = 100
# time.sleep(2)
# elapsed_time, estimated_remaining_time = time_info(start_time, current_step, total_steps)
# print(elapsed_time, estimated_remaining_time)



def img2edgemap(file_path):
    import cv2

    # 读取图像
    image = cv2.imread(file_path, 0)  # 以灰度模式读取图像

    # 使用Canny边缘检测算法
    edges = cv2.Canny(image, 100, 200)  # 这里的参数100和200分别是Canny算法的低阈值和高阈值

    # 保存边缘图像
    out_img_name = os.path.basename(file_path)
    out_img_name = os.path.splitext(out_img_name)[0] + '_edge_map.jpg'
    cv2.imwrite(out_img_name, edges)

