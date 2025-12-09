import os  # 导入操作系统接口模块，用于处理文件路径和环境变量
import sys  # 导入系统特定的参数和函数模块，用于修改Python搜索路径

# ==========================================
# 1. 设置无头模式 (Headless Mode)
# ==========================================
os.environ["SDL_VIDEODRIVER"] = "dummy"  # 设置SDL视频驱动为"dummy"（伪驱动），告诉Pygame不要弹出图形窗口，实现后台静默运行

# ==========================================
# 2. 路径与环境配置
# ==========================================
current_script_path = os.path.abspath(__file__)  # 获取当前脚本文件(train.py)的绝对路径
rl_dir = os.path.dirname(current_script_path)  # 获取当前脚本所在的目录，即 source/rl
source_dir = os.path.dirname(rl_dir)  # 获取上一级目录，即 source
project_root = os.path.dirname(source_dir)  # 获取再上一级目录，即项目根目录 PythonPlantsVsZombies-master

os.chdir(project_root)  # 将当前工作目录切换到项目根目录，确保程序能找到 'resources' 等资源文件夹
if project_root not in sys.path:  # 检查项目根目录是否已经在Python的模块搜索路径中
    sys.path.insert(0, project_root)  # 如果不在，将其插入到搜索路径的最前面，确保能正确导入 'source' 包

# ==========================================
# 3. 导入依赖
# ==========================================
from stable_baselines3 import PPO  # 从stable_baselines3库导入PPO（近端策略优化）算法，这是我们使用的强化学习核心算法
from stable_baselines3.common.vec_env import SubprocVecEnv  # 导入SubprocVecEnv，用于创建并行的多进程环境，加速数据采样
from stable_baselines3.common.callbacks import CheckpointCallback  # 导入CheckpointCallback，用于在训练过程中定期保存模型
from source.rl.env import ZombiePvZEnv, ScriptedPlantPolicy  # 从我们自定义的环境模块导入僵尸环境类和脚本控制的植物策略类


def make_env(rank, seed=0):  # 定义一个工厂函数，用于创建单个环境实例。rank是进程ID，seed是随机种子
    def _init():  # 内部初始化函数
        plant_opponent = ScriptedPlantPolicy()  # 实例化一个由脚本控制的植物方对手
        env = ZombiePvZEnv(plant_policy=plant_opponent, render_mode=None)  # 创建僵尸方环境，传入植物策略，不开启渲染模式
        env.reset(seed=seed + rank)  # 重置环境，并设置随机种子（基础种子 + 进程ID），保证每个进程的随机性不同
        return env  # 返回创建好的环境实例

    return _init  # 返回内部初始化函数


if __name__ == '__main__':  # Python程序的入口点保护，防止多进程启动时递归运行
    num_cpu = 4  # 设置并行环境的数量（即使用多少个CPU核心），这里设置为4个

    log_dir = os.path.join(project_root, "logs")  # 拼接日志保存目录的路径
    tensorboard_log = os.path.join(project_root, "pvz_zombie_tensorboard")  # 拼接Tensorboard日志的保存路径
    os.makedirs(log_dir, exist_ok=True)  # 创建日志目录，如果已存在则不报错

    print(f"Project Root: {project_root}")  # 打印项目根目录路径，用于调试
    print(f"Log Directory: {log_dir}")  # 打印日志目录路径

    # 创建并行环境
    # 列表推导式生成 num_cpu 个环境工厂函数，SubprocVecEnv 会在独立的进程中运行它们
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    final_model_path = os.path.join(log_dir, "zombie_final_model.zip")  # 定义最终模型保存的文件路径
    interrupted_model_path = os.path.join(log_dir, "zombie_interrupted_model.zip")  # 定义中断保存模型的文件路径

    model = None  # 初始化模型变量为空

    if os.path.exists(interrupted_model_path):  # 检查是否存在被中断保存的模型文件
        print("加载中断存档...")  # 打印提示信息
        # 加载中断的模型，绑定当前环境，并指定Tensorboard日志路径
        model = PPO.load(interrupted_model_path, env=env, tensorboard_log=tensorboard_log)
    elif os.path.exists(final_model_path):  # 如果没有中断存档，检查是否存在上次训练完成的模型
        print("加载历史模型接力训练...")  # 打印提示信息
        # 加载历史最终模型，用于继续训练（接力）
        model = PPO.load(final_model_path, env=env, tensorboard_log=tensorboard_log)
    else:  # 如果没有任何历史模型
        print("开始新训练...")  # 打印提示信息
        # 初始化一个新的PPO模型
        model = PPO(
            "MultiInputPolicy",  # 策略类型，这里使用多输入策略，因为我们的观测空间是字典（Grid + Stats）
            env,  # 传入并行环境
            verbose=1,  # 设置日志详细程度，1表示打印基本信息
            tensorboard_log=tensorboard_log,  # 设置Tensorboard日志路径
            learning_rate=3e-4,  # 设置学习率，0.0003
            # 减小 batch_size 和 n_steps，以适应更短的游戏回合，使更新更频繁
            batch_size=256,  # 每次梯度更新使用的样本批次大小
            n_steps=512,  # 每个环境采样多少步后进行一次更新
            gamma=0.99  # 折扣因子，决定AI看重长远利益的程度
        )

    # 创建检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # 每隔10000步保存一次模型
        save_path=log_dir,  # 保存路径
        name_prefix='zombie_ppo'  # 保存文件的前缀名
    )

    print("开始训练...")  # 打印开始信息
    try:  # 开始异常捕获块，用于处理用户中断
        # 开始训练模型
        # total_timesteps: 总训练步数，这里设为100万步
        # reset_num_timesteps=False: 关键参数，设为False表示接续之前的步数计数，而不是从0开始
        model.learn(total_timesteps=1_000_000, callback=checkpoint_callback, reset_num_timesteps=False)

        model.save(final_model_path)  # 训练完成后，保存最终模型
        print("训练完成")  # 打印完成信息

        # 如果训练顺利完成，且存在中断存档，则删除中断存档，以免下次误加载旧进度
        if os.path.exists(interrupted_model_path):
            os.remove(interrupted_model_path)

    except KeyboardInterrupt:  # 捕获 KeyboardInterrupt 异常（通常是用户按 Ctrl+C）
        print("中断保存...")  # 打印提示
        model.save(interrupted_model_path)  # 保存当前模型为中断存档
    finally:  # 无论是否发生异常，最后都会执行的代码块
        env.close()  # 关闭环境，释放资源（如关闭子进程）