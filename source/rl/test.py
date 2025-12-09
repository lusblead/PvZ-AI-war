import os
import sys
import time

# ==========================================
# 路径配置
# ==========================================
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
source_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(source_dir)

# 切换工作目录到项目根目录 (这样才能加载 resources)
os.chdir(project_root)

# 添加搜索路径 (这样才能导入 source)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from source.rl.env import ZombiePvZEnv, ScriptedPlantPolicy


def main():
    # 1. 确定模型路径
    # 优先寻找最终模型，如果没有，则寻找中断保存的模型
    log_dir = os.path.join(project_root, "logs")
    final_model_path = os.path.join(log_dir, "zombie_final_model.zip")
    interrupted_model_path = os.path.join(log_dir, "zombie_interrupted_model.zip")

    model_path = None
    if os.path.exists(final_model_path):
        model_path = final_model_path
    elif os.path.exists(interrupted_model_path):
        print(f"提示: 未找到最终模型，将加载中断保存的模型: {interrupted_model_path}")
        model_path = interrupted_model_path
    else:
        print(f"错误: 在 {log_dir} 下未找到任何模型文件 (.zip)")
        return

    # 2. 加载模型
    print(f"正在加载模型: {model_path}")
    # 不需要传入 env，除非你需要继续训练
    model = PPO.load(model_path)

    # 3. 创建可视化环境
    # render_mode='human' 会弹出 Pygame 窗口
    plant_opponent = ScriptedPlantPolicy()
    env = ZombiePvZEnv(plant_policy=plant_opponent, render_mode='human')
    # 4. 开始游戏循环
    obs, _ = env.reset()

    print("开始演示 AI 对战... (按 Ctrl+C 退出)")
    try:
        while True:
            # 预测动作
            # deterministic=True: 使用确定性策略（测试时通常开启，表现更稳定）
            # deterministic=False: 带有随机性（如果你想看 AI 尝试不同操作）
            action, _State = model.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)

            # 渲染画面
            env.render()

            # 控制播放速度
            # 训练时是极速运行的，观看时加一点延迟以免太快
            time.sleep(0.05)

            if terminated or truncated:
                print("回合结束 (胜利/失败/超时)，重置环境...")
                # 稍微暂停一下再重开
                time.sleep(1)
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n演示结束")
    finally:
        env.close()


if __name__ == '__main__':
    main()