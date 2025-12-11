import numpy as np  # 导入numpy库，用于高效的数组运算，处理观测数据
import pygame as pg  # 导入pygame库，用于游戏逻辑运行和画面绘制
import gymnasium as gym  # 导入gymnasium库，这是标准的强化学习环境接口库
from gymnasium import spaces  # 从gymnasium导入spaces模块，用于定义动作和观测空间
from stable_baselines3 import PPO  # 导入PPO算法（用于加载植物方AI模型，虽然当前主要用脚本）

from .. import constants as c  # 导入项目的常量配置模块，重命名为c
from ..state import level as level_module  # 导入关卡状态模块，重命名为level_module，这是游戏的核心逻辑
from .. import tool  # 导入工具模块，包含图像加载等功能
from ..component import menubar  # 导入菜单栏组件，用于获取植物卡片列表

# 定义僵尸造价字典，键是僵尸类型常量，值是消耗的阳光数
ZOMBIE_COSTS = {
    c.NORMAL_ZOMBIE: 50,  # 普通僵尸
    c.CONEHEAD_ZOMBIE: 75,  # 路障僵尸
    c.BUCKETHEAD_ZOMBIE: 125,  # 铁桶僵尸
    c.NEWSPAPER_ZOMBIE: 100  # 读报僵尸
}


class PlantPolicy:  # 定义植物策略的基类（接口）
    def get_action(self, game_state):  # 定义获取动作的方法接口
        raise NotImplementedError  # 抽象方法，子类必须实现，否则报错


class ScriptedPlantPolicy(PlantPolicy):  # 定义脚本控制的植物策略类
    def __init__(self):  # 初始化方法
        self.cooldown_timer = 0  # 初始化操作冷却计时器
        self.sun_points = 50  # 初始化植物方阳光资源为50

    def get_action(self, level_obj):  # 获取动作的具体逻辑
        self.cooldown_timer += 100  # 假设每次调用间隔100ms，计时器增加

        # 模拟植物方资源自然增长，每5秒增加25点
        if self.cooldown_timer % 5000 == 0:
            self.sun_points += 25

        # 操作频率限制：每2秒只能执行一次操作，防止脚本瞬间铺满全场
        if self.cooldown_timer < 2000:
            return

        # 构建当前场上的植物网格地图，用于决策判断
        grid_map = [[None for _ in range(9)] for _ in range(5)]  # 5行9列的空网格
        for y in range(5):  # 遍历行
            for plant in level_obj.plant_groups[y]:  # 遍历该行的植物组
                # 获取植物的网格坐标
                grid_x, grid_y = level_obj.map.getMapIndex(plant.rect.centerx, plant.rect.bottom)
                if 0 <= grid_x < 9 and 0 <= grid_y < 5:  # 确保坐标有效
                    grid_map[grid_y][grid_x] = plant.name  # 记录植物名称

        import random  # 导入随机模块
        action_taken = False  # 标记本回合是否执行了种植动作

        # 简单策略1：优先在第0列和第1列种向日葵
        if self.sun_points >= 50:  # 如果阳光足够
            for col in range(2):  # 遍历前两列
                for row in range(5):  # 遍历每一行
                    if grid_map[row][col] is None:  # 如果该位置为空
                        if random.random() < 0.4:  # 40%概率种植（增加随机性）
                            level_obj.addPlantStatic(col, row, c.SUNFLOWER)  # 调用level接口种植向日葵
                            self.sun_points -= 50  # 扣除资源
                            action_taken = True  # 标记已行动
                            break  # 跳出内层循环
                if action_taken: break  # 跳出外层循环

        # 简单策略2：如果还没行动且资源充足，随机在第2-6列种豌豆射手
        if not action_taken and self.sun_points >= 100:
            row = random.randint(0, 4)  # 随机选择一行
            col = random.randint(2, 6)  # 随机选择一列
            if grid_map[row][col] is None:  # 如果位置为空
                level_obj.addPlantStatic(col, row, c.PEASHOOTER)  # 种植豌豆射手
                self.sun_points -= 100  # 扣除资源
                action_taken = True  # 标记已行动

        if action_taken:  # 如果执行了动作
            self.cooldown_timer = 0  # 重置冷却计时器


class AIPlantPolicy(PlantPolicy):  # 定义AI控制的植物策略类（占位，暂未使用）
    def __init__(self, model_path):  # 初始化
        self.model = PPO.load(model_path)  # 加载PPO模型

    def get_action(self, obs):  # 获取动作接口
        pass  # 暂未实现


class ZombiePvZEnv(gym.Env):  # 定义僵尸方强化学习环境，继承自gym.Env
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}  # 定义环境元数据

    def __init__(self, plant_policy: PlantPolicy, render_mode=None):  # 初始化环境
        super(ZombiePvZEnv, self).__init__()  # 调用父类构造函数
        self.plant_policy = plant_policy  # 保存植物方策略
        self.render_mode = render_mode  # 保存渲染模式

        # 定义动作空间：离散空间，大小为21 (1个无操作 + 4种僵尸 * 5行)
        self.action_space = spaces.Discrete(1 + 4 * 5)

        # 定义观测空间：字典结构
        self.observation_space = spaces.Dict({
            # grid: 2个通道(植物血量, 僵尸密度) x 5行 x 9列
            "grid": spaces.Box(low=0, high=1, shape=(2, 5, 9), dtype=np.float32),
            # stats: 8个统计数值 (资源, 冷却等)
            "stats": spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        })

        pg.init()  # 初始化Pygame
        if render_mode == 'human':  # 如果是人类观看模式
            self.screen = pg.display.set_mode(c.SCREEN_SIZE)  # 创建图形窗口
            pg.display.set_caption("PvZ AI Battle")  # 设置窗口标题
        else:  # 如果是无头模式
            self.screen = None  # 不创建窗口
        self.level = None  # 初始化level对象为空

    def reset(self, seed=None, options=None):  # 环境重置函数，每回合开始时调用
        super().reset(seed=seed)  # 调用父类重置，设置随机种子

        persist = {c.CURRENT_TIME: 0.0, c.LEVEL_NUM: 1}  # 初始游戏状态参数
        self.level = level_module.Level()  # 实例化游戏关卡对象
        self.level.startup(0.0, persist)  # 启动关卡

        # 初始化植物方的卡片列表（将字符串转换为索引）
        default_cards_names = [c.SUNFLOWER, c.PEASHOOTER, c.WALLNUT, c.POTATOMINE]
        default_cards_indices = []
        for name in default_cards_names:
            if name in menubar.card_name_list:
                default_cards_indices.append(menubar.card_name_list.index(name))
        self.level.initPlay(default_cards_indices)  # 初始化战斗模式
        self.level.zombie_list = []  # 清空关卡预设的脚本僵尸，完全由AI控制

        # 【关键修改】开局直接给 500 阳光！解决冷启动难的问题，让AI开局就能买买买
        self.moonlight = 50

        self.time_seconds = 0.0  # 重置游戏时间
        self.frame_count = 0  # 重置帧数
        self.global_cd_timer = 0  # 重置全局冷却
        self.lane_cd_timers = [0] * 5  # 重置各行冷却

        # 【关键修改】将开局禁手时间设为0，AI一上来就能放僵尸
        self.start_delay_timer = 0
        self._resource_timer = 0.0  # 重置资源计时器

        return self._get_obs(), {}  # 返回初始观测值和空信息字典

    def step(self, action):  # 环境步进函数，执行动作并返回结果
        dt_ms = 100  # 每次step模拟100毫秒
        current_time = self.level.current_time + dt_ms  # 计算新的当前时间
        self.frame_count += 1  # 帧数加1
        self.time_seconds = current_time / 1000.0  # 更新秒数

        # 更新各种冷却时间（减少流逝的时间）
        if self.global_cd_timer > 0: self.global_cd_timer -= (dt_ms / 1000.0)
        if self.start_delay_timer > 0: self.start_delay_timer -= (dt_ms / 1000.0)
        for i in range(5):
            if self.lane_cd_timers[i] > 0: self.lane_cd_timers[i] -= (dt_ms / 1000.0)

        self._update_moonlight(dt_ms)  # 更新资源增长逻辑
        self.plant_policy.get_action(self.level)  # 执行植物方策略
        valid_action = self._apply_zombie_action(action)  # 执行僵尸方AI动作，并获取是否有效

        # 更新游戏物理逻辑
        # 传入 None 作为 surface，表示只计算逻辑不绘图，提高训练速度
        self.level.update(None, current_time, None, [False, False])

        reward = self._calculate_reward(valid_action)  # 计算奖励
        terminated = self._check_game_over()  # 检查游戏是否因胜负结束


        truncated = self.time_seconds >= 300

        return self._get_obs(), reward, terminated, truncated, {}  # 返回 (观测, 奖励, 终止, 截断, 信息)

    def _update_moonlight(self, dt_ms):  # 资源增长逻辑
        self._resource_timer += (dt_ms / 1000.0)  # 累加计时器
        # 为了加快训练节奏，资源获取速度设为较快：每2秒增加50点
        if self._resource_timer >= 4.0:
            self.moonlight += 50
            self._resource_timer = 0.0

    def _apply_zombie_action(self, action):  # 解析并执行僵尸动作
        if action == 0: return True  # 如果动作为0（不操作），始终有效

        # 解析动作ID -> 坐标和类型
        action_idx = action - 1
        row = action_idx % 5  # 行号
        zombie_type_idx = action_idx // 5  # 类型索引

        zombie_types = [c.NORMAL_ZOMBIE, c.CONEHEAD_ZOMBIE, c.BUCKETHEAD_ZOMBIE, c.NEWSPAPER_ZOMBIE]
        z_name = zombie_types[zombie_type_idx]  # 获取僵尸名称
        cost = ZOMBIE_COSTS[z_name]  # 获取僵尸造价

        # 规则检查：冷却中或钱不够则无效
        if self.start_delay_timer > 0: return False
        if self.global_cd_timer > 0: return False
        if self.lane_cd_timers[row] > 0: return False
        if self.moonlight < cost: return False

        # 执行生成
        self.moonlight -= cost  # 扣钱
        self.level.createZombie(z_name, row)  # 调用游戏接口生成僵尸
        self.global_cd_timer = 2.0  # 设置全局冷却2秒
        self.lane_cd_timers[row] = 5.0  # 设置该行冷却5秒
        return True  # 返回动作有效

    def _get_obs(self):  # 获取当前观测状态
        grid = np.zeros((2, 5, 9), dtype=np.float32)  # 初始化网格数据
        # 填充 Channel 0: 植物生命值 (归一化到 0-1)
        for y in range(self.level.map_y_len):
            for plant in self.level.plant_groups[y]:
                grid_x, grid_y = self.level.map.getMapIndex(plant.rect.centerx, plant.rect.bottom)
                if 0 <= grid_x < 9 and 0 <= grid_y < 5:
                    grid[0][grid_y][grid_x] = min(plant.health / 500.0, 1.0)
        # 填充 Channel 1: 僵尸密度 (归一化)
        for y in range(self.level.map_y_len):
            for zombie in self.level.zombie_groups[y]:
                grid_x, grid_y = self.level.map.getMapIndex(zombie.rect.centerx, zombie.rect.bottom)
                if 0 <= grid_x < 9:
                    grid[1][y][grid_x] = min(grid[1][y][grid_x] + 0.2, 1.0)

                    # 填充统计向量 (全部归一化处理，方便神经网络学习)
        stats = np.array([
            self.moonlight / 1000.0,  # 阳光 / 1000
            self.global_cd_timer / 2.0,  # 冷却 / 2
            self.lane_cd_timers[0] / 5.0,  # 行冷却 / 5
            self.lane_cd_timers[1] / 5.0,
            self.lane_cd_timers[2] / 5.0,
            self.lane_cd_timers[3] / 5.0,
            self.lane_cd_timers[4] / 5.0,
            self.time_seconds / 300.0  # 时间 / 60
        ], dtype=np.float32)
        return {"grid": grid, "stats": stats}  # 返回观测字典

    def _calculate_reward(self, valid_action):  # 计算奖励函数
        reward = 0

        # 1. 鼓励放置僵尸 (核心引导)
        if not valid_action:
            return 0  # 即使动作无效也不扣分，防止AI学会偷懒
        if valid_action and self.global_cd_timer == 2.0:
            reward += 10.0  # 只要成功花出去了钱，给予+10分的巨额奖励

        # 2. 推进奖励：僵尸越靠左，奖励越高
        for i in range(5):
            for zombie in self.level.zombie_groups[i]:
                if zombie.state != c.DIE:
                    # 计算进度百分比 (起点->0)
                    progress = (c.ZOMBIE_START_X - zombie.rect.centerx) / c.ZOMBIE_START_X
                    reward += progress * 0.1  # 持续给予小额奖励

        # 3. 胜利奖励
        if self.level.checkLose():  # checkLose返回True表示僵尸进屋了
            reward += 1000  # 给予终极胜利奖励

        return reward

    def _check_game_over(self):  # 检查是否满足终止条件
        # 仅当僵尸进屋（僵尸胜）时终止，或者由 step 函数中的 truncated (超时) 终止
        return self.level.checkLose()

    def render(self):  # 画面渲染函数
        if self.render_mode == "human" and self.screen:  # 仅在有屏幕时渲染
            # 处理事件队列，防止窗口在Windows下卡死
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    import sys;
                    sys.exit()
            self.level.draw(self.screen)  # 调用游戏本身的绘制方法
            pg.display.update()  # 刷新屏幕显示