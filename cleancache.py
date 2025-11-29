import os
import shutil

def clear_folder(folder_path):
    """简单版本：直接清空文件夹"""
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在")
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"删除 {filename} 时出错: {e}")
    
    print(f"已清空文件夹: {folder_path}")

# 直接执行
clear_folder("results_18emo")
clear_folder("emotion_model_18emo")