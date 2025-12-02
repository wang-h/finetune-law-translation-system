import torch
import os
import glob
import shutil
from collections import OrderedDict

def average_checkpoints(folder_path, output_path, last_n=3):
    """
    平均最后 N 个 epoch 的模型权重 (增加对错误格式的兼容)
    """
    print(f"扫描目录: {folder_path}")
    try:
        subdirs = [d for d in os.listdir(folder_path) if d.startswith('epoch-') and os.path.isdir(os.path.join(folder_path, d))]
    except FileNotFoundError:
        print(f"❌ 目录不存在: {folder_path}")
        return

    subdirs.sort(key=lambda x: int(x.split('-')[1]))
    
    if not subdirs:
        print("❌ 未找到任何 epoch 文件夹")
        return

    candidate_dirs = subdirs[-min(len(subdirs), last_n + 2):] 
    print(f"🔍 候选模型: {candidate_dirs}")
    
    avg_state_dict = None
    count = 0
    valid_models = []

    for dirname in reversed(candidate_dirs):
        if count >= last_n:
            break
            
        full_dir_path = os.path.join(folder_path, dirname)
        bin_path = os.path.join(full_dir_path, "pytorch_model.bin")
        safetensors_path = os.path.join(full_dir_path, "model.safetensors")
        
        state_dict = None
        try:
            if os.path.exists(bin_path):
                print(f"   📖 加载 {dirname} (bin)...")
                state_dict = torch.load(bin_path, map_location='cpu')
            elif os.path.exists(safetensors_path):
                print(f"   📖 尝试加载 {dirname} (safetensors)...")
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(safetensors_path)
                except Exception as st_err:
                    print(f"      ⚠️ Safetensors 加载失败: {st_err}")
                    print(f"      🔄 尝试作为 PyTorch pickle 格式加载...")
                    # 关键修改：尝试用 torch.load 读取 .safetensors 文件
                    state_dict = torch.load(safetensors_path, map_location='cpu', weights_only=False)
                    print(f"      ✅ PyTorch 格式加载成功！(文件名后缀错误)")
            else:
                print(f"   ⚠️ 跳过 {dirname}: 找不到权重文件")
                continue
        except Exception as e:
            print(f"   ❌ 彻底失败 {dirname}: {str(e)}")
            continue

        print(f"   ✅ 成功加载 {dirname}")
        valid_models.append(dirname)

        if avg_state_dict is None:
            avg_state_dict = state_dict
        else:
            for key in state_dict:
                if key in avg_state_dict:
                    if isinstance(avg_state_dict[key], torch.Tensor):
                        avg_state_dict[key] = avg_state_dict[key].float() + state_dict[key].float()
        
        count += 1
            
    if avg_state_dict is None:
        print("❌ 没有加载到任何有效模型")
        return

    print(f"📚 最终合并了 {count} 个模型: {valid_models}")

    print("➗ 计算平均值...")
    for key in avg_state_dict:
        if isinstance(avg_state_dict[key], torch.Tensor):
            avg_state_dict[key] = avg_state_dict[key] / count

    os.makedirs(output_path, exist_ok=True)
    
    if valid_models:
        last_valid_model_dir = os.path.join(folder_path, valid_models[0])
        print(f"📋 从 {valid_models[0]} 复制配置文件...")
        
        files_to_copy = ['config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'spiece.model', 'generation_config.json']
        for filename in files_to_copy:
            src = os.path.join(last_valid_model_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, output_path)
    
    # 既然源文件其实是 pickle 格式，我们输出时最好也用 pickle (.bin)，避免混淆
    output_file = os.path.join(output_path, "pytorch_model.bin")
    print(f"💾 保存合并权重到: {output_file}")
    torch.save(avg_state_dict, output_file)
    print(f"✅ 完成! 合并模型位于: {output_path}")

if __name__ == "__main__":
    # 根据你的目录结构修改
    BASE_DIR = "/home/hao/law_translation_project/finetune_mt/checkpoints/mt5-zh2en"
    OUTPUT_DIR = "/home/hao/law_translation_project/finetune_mt/checkpoints/mt5-zh2en-avg"
    
    average_checkpoints(BASE_DIR, OUTPUT_DIR, last_n=3)

