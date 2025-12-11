"""
对战关卡：人类玩家控制植物 vs AI控制僵尸
"""
import os
import sys
import pygame as pg
import numpy as np
import random
from stable_baselines3 import PPO

# ==========================================
# 路径配置
# ==========================================
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
source_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(source_dir)

os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入游戏模块
import source.constants as c
import source.tool as tool
from source.component import menubar, map, plant, zombie

class HumanVsAILevel(tool.State):
    """人类玩家 vs 僵尸AI的对战关卡"""

    def __init__(self):
        tool.State.__init__(self)

    def startup(self, current_time, persist):
        """初始化关卡"""
        self.game_info = persist
        self.persist = self.game_info
        self.game_info[c.CURRENT_TIME] = current_time
        self.map_y_len = c.GRID_Y_LEN

        # 加载僵尸AI模型
        self.load_zombie_ai_model()

        # 加载对战专用地图
        self.map_data = self.loadBattleMap()
        self.setupBackground()
        self.initState()

    def load_zombie_ai_model(self):
        """加载训练好的僵尸AI模型"""
        log_dir = os.path.join(project_root, "logs")
        model_path = os.path.join(log_dir, "zombie_final_model.zip")

        if os.path.exists(model_path):
            print(f"加载僵尸AI模型: {model_path}")
            try:
                # 直接加载模型，不创建环境
                self.zombie_ai = PPO.load(model_path)
                print("僵尸AI模型加载成功")
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("将使用随机策略")
                self.zombie_ai = None
        else:
            print("警告: 未找到训练好的僵尸AI模型，将使用随机策略")
            self.zombie_ai = None

        # 初始化僵尸AI状态
        self.zombie_action_timer = 0
        self.zombie_action_interval = 2000  # AI每2秒做一次决策(毫秒)

    def loadBattleMap(self):
        """加载对战地图配置"""
        # 创建一个自定义地图配置
        return {
            c.BACKGROUND_TYPE: c.BACKGROUND_DAY,  # 白天背景
            c.INIT_SUN_NAME: 300,  # 初始阳光
            c.ZOMBIE_LIST: [],  # 清空预设僵尸，完全由AI控制
            c.CHOOSEBAR_TYPE: c.CHOOSEBAR_STATIC,  # 静态菜单栏
        }

    def setupBackground(self):
        """设置背景"""
        img_index = self.map_data[c.BACKGROUND_TYPE]
        self.background_type = img_index
        self.background = tool.GFX[c.BACKGROUND_NAME][img_index]
        self.bg_rect = self.background.get_rect()

        self.level = pg.Surface((self.bg_rect.w, self.bg_rect.h)).convert()
        self.viewport = tool.SCREEN.get_rect(bottom=self.bg_rect.bottom)
        self.viewport.x += c.BACKGROUND_OFFSET_X

        # 初始化地图
        self.map_obj = map.Map(c.GRID_X_LEN, self.map_y_len)

    def setupGroups(self):
        """设置精灵组"""
        self.sun_group = pg.sprite.Group()
        self.head_group = pg.sprite.Group()

        self.plant_groups = []
        self.zombie_groups = []
        self.hypno_zombie_groups = []
        self.bullet_groups = []
        for i in range(self.map_y_len):
            self.plant_groups.append(pg.sprite.Group())
            self.zombie_groups.append(pg.sprite.Group())
            self.hypno_zombie_groups.append(pg.sprite.Group())
            self.bullet_groups.append(pg.sprite.Group())

    def setupCars(self):
        """设置小推车"""
        self.cars = []
        for i in range(self.map_y_len):
            _, y = self.map_obj.getMapGridPos(0, i)
            self.cars.append(plant.Car(-25, y+20, i))

    def initState(self):
        """初始化游戏状态"""
        self.state = c.PLAY

        # 创建自定义卡片索引列表（使用menubar.py中定义的索引）
        # 根据menubar.py中的plant_name_list顺序，我们需要找到对应植物的索引
        battle_cards = []  # 存储卡片索引

        # 定义我们想要使用的植物及其在menubar.py中的索引
        # 根据menubar.py中的plant_name_list顺序：
        # 0: SUNFLOWER, 1: PEASHOOTER, 2: SNOWPEASHOOTER, 3: WALLNUT,
        # 4: CHERRYBOMB, 5: THREEPEASHOOTER, 6: REPEATERPEA, 7: CHOMPER,
        # 8: PUFFSHROOM, 9: POTATOMINE, 10: SQUASH, 11: SPIKEWEED,
        # 12: JALAPENO, 13: SCAREDYSHROOM, 14: SUNSHROOM, 15: ICESHROOM,
        # 16: HYPNOSHROOM, 17: WALLNUTBOWLING, 18: REDWALLNUTBOWLING

        plant_to_index = {
            c.SUNFLOWER: 0,
            c.PEASHOOTER: 1,
            c.SNOWPEASHOOTER: 2,
            c.WALLNUT: 3,
            c.CHERRYBOMB: 4,
            c.THREEPEASHOOTER: 5,
            c.REPEATERPEA: 6,
            c.CHOMPER: 7,
            c.PUFFSHROOM: 8,
            c.POTATOMINE: 9,
            c.SQUASH: 10,
            c.SPIKEWEED: 11,
            c.JALAPENO: 12,
            c.SCAREDYSHROOM: 13,
            c.SUNSHROOM: 14,
            c.ICESHROOM: 15,
            c.HYPNOSHROOM: 16,
            c.WALLNUTBOWLING: 17,
            c.REDWALLNUTBOWLING: 18
        }

        # 添加我们想要的植物索引
        for plant_name in [c.SUNFLOWER, c.PEASHOOTER, c.WALLNUT, c.SNOWPEASHOOTER,
                          c.CHERRYBOMB, c.POTATOMINE, c.SQUASH]:
            if plant_name in plant_to_index:
                battle_cards.append(plant_to_index[plant_name])
            else:
                print(f"警告: 植物 {plant_name} 不在plant_name_list中")

        # 手动创建菜单栏
        self.menubar = menubar.MenuBar(battle_cards, self.map_data[c.INIT_SUN_NAME])

        self.drag_plant = False
        self.hint_image = None
        self.hint_plant = False
        self.produce_sun = True
        self.sun_timer = self.current_time

        self.removeMouseImage()
        self.setupGroups()
        self.setupCars()

        # 僵尸AI资源初始化
        self.zombie_moonlight = 500  # 僵尸AI初始资源
        self.zombie_global_cd = 0  # 全局冷却
        self.zombie_lane_cds = [0] * 5  # 每行冷却
        self.zombie_resource_timer = 0  # 资源增长计时器

    def update(self, surface, current_time, mouse_pos, mouse_click):
        """更新游戏逻辑"""
        self.current_time = self.game_info[c.CURRENT_TIME] = current_time

        if self.state == c.PLAY:
            self.play(mouse_pos, mouse_click)

        self.draw(surface)

    def play(self, mouse_pos, mouse_click):
        """游戏主循环"""
        # 更新僵尸AI
        self._update_zombie_ai()

        # 更新游戏对象
        for i in range(self.map_y_len):
            self.bullet_groups[i].update(self.game_info)
            self.plant_groups[i].update(self.game_info)
            self.zombie_groups[i].update(self.game_info)
            self.hypno_zombie_groups[i].update(self.game_info)

        self.head_group.update(self.game_info)
        self.sun_group.update(self.game_info)

        # 处理玩家种植植物
        if not self.drag_plant and mouse_pos and mouse_click[0]:
            result = self.menubar.checkCardClick(mouse_pos)
            if result:
                self.setupMouseImage(result[0], result[1])
        elif self.drag_plant:
            if mouse_click[1]:  # 右键取消
                self.removeMouseImage()
            elif mouse_click[0]:  # 左键种植
                if self.menubar.checkMenuBarClick(mouse_pos):
                    self.removeMouseImage()
                else:
                    self.addPlant()
            elif mouse_pos is None:
                self.setupHintImage()

        # 生产阳光
        if self.produce_sun:
            if (self.current_time - self.sun_timer) > c.PRODUCE_SUN_INTERVAL:
                self.sun_timer = self.current_time
                map_x, map_y = self.map_obj.getRandomMapIndex()
                x, y = self.map_obj.getMapGridPos(map_x, map_y)
                self.sun_group.add(plant.Sun(x, 0, x, y))

        # 收集阳光
        if not self.drag_plant and mouse_pos and mouse_click[0]:
            for sun in self.sun_group:
                if sun.checkCollision(mouse_pos[0], mouse_pos[1]):
                    self.menubar.increaseSunValue(sun.sun_value)

        for car in self.cars:
            car.update(self.game_info)

        self.menubar.update(self.current_time)

        # 检查碰撞
        self.checkBulletCollisions()
        self.checkZombieCollisions()
        self.checkPlants()
        self.checkCarCollisions()
        self.checkGameState()

    def _update_zombie_ai(self):
        """更新僵尸AI决策"""
        # 更新冷却时间
        dt_ms = 100  # 假设每帧100ms
        if self.zombie_global_cd > 0:
            self.zombie_global_cd = max(0, self.zombie_global_cd - dt_ms/1000.0)
        for i in range(5):
            if self.zombie_lane_cds[i] > 0:
                self.zombie_lane_cds[i] = max(0, self.zombie_lane_cds[i] - dt_ms/1000.0)

        # 僵尸资源增长（每2秒增加50）
        self.zombie_resource_timer += dt_ms
        if self.zombie_resource_timer >= 2000:
            self.zombie_moonlight += 50
            self.zombie_resource_timer = 0

        # 僵尸AI决策（每2秒一次）
        if (self.current_time - self.zombie_action_timer) > self.zombie_action_interval:
            self.zombie_action_timer = self.current_time

            if self.zombie_ai:
                try:
                    # 获取观测
                    obs = self._get_zombie_ai_obs()

                    # AI决策
                    action, _ = self.zombie_ai.predict(obs, deterministic=True)

                    # 执行动作
                    self._execute_zombie_action(action)
                except Exception as e:
                    print(f"僵尸AI决策错误: {e}")
                    self._random_zombie_action()
            else:
                # 随机策略作为备用
                self._random_zombie_action()

    def _get_zombie_ai_obs(self):
        """为僵尸AI构建观测"""
        # 构建网格观测 (2通道, 5行, 9列)
        grid = np.zeros((2, 5, 9), dtype=np.float32)

        # 通道0: 植物存在和血量
        for y in range(self.map_y_len):
            for plant_obj in self.plant_groups[y]:
                grid_x, grid_y = self.map_obj.getMapIndex(plant_obj.rect.centerx, plant_obj.rect.bottom)
                if 0 <= grid_x < 9 and 0 <= grid_y < 5:
                    grid[0][grid_y][grid_x] = min(plant_obj.health / 500.0, 1.0)

        # 通道1: 僵尸密度
        for y in range(self.map_y_len):
            for zombie_obj in self.zombie_groups[y]:
                grid_x, grid_y = self.map_obj.getMapIndex(zombie_obj.rect.centerx, zombie_obj.rect.bottom)
                if 0 <= grid_x < 9:
                    grid[1][y][grid_x] = min(grid[1][y][grid_x] + 0.2, 1.0)

        # 构建统计向量
        stats = np.array([
            self.zombie_moonlight / 1000.0,
            max(0, self.zombie_global_cd / 2.0),
            max(0, self.zombie_lane_cds[0] / 5.0),
            max(0, self.zombie_lane_cds[1] / 5.0),
            max(0, self.zombie_lane_cds[2] / 5.0),
            max(0, self.zombie_lane_cds[3] / 5.0),
            max(0, self.zombie_lane_cds[4] / 5.0),
            self.current_time / 300000.0  # 归一化时间
        ], dtype=np.float32)

        return {"grid": grid, "stats": stats}

    def _execute_zombie_action(self, action):
        """执行僵尸AI的动作"""
        # 动作映射: 0-无操作, 1-20: 4种僵尸×5行
        if action == 0:
            return

        action_idx = action - 1
        row = action_idx % 5
        zombie_type_idx = action_idx // 5

        zombie_types = [c.NORMAL_ZOMBIE, c.CONEHEAD_ZOMBIE, c.BUCKETHEAD_ZOMBIE, c.NEWSPAPER_ZOMBIE]
        if zombie_type_idx >= len(zombie_types):
            return

        z_name = zombie_types[zombie_type_idx]
        cost = self._get_zombie_cost(z_name)

        # 检查条件
        if self.zombie_moonlight < cost:
            return
        if self.zombie_global_cd > 0:
            return
        if self.zombie_lane_cds[row] > 0:
            return

        # 生成僵尸
        self.createZombie(z_name, row)
        self.zombie_moonlight -= cost
        self.zombie_global_cd = 2.0
        self.zombie_lane_cds[row] = 5.0

    def _random_zombie_action(self):
        """随机僵尸策略（备用）"""
        # 随机选择僵尸类型
        zombie_types = [c.NORMAL_ZOMBIE, c.CONEHEAD_ZOMBIE, c.BUCKETHEAD_ZOMBIE, c.NEWSPAPER_ZOMBIE]
        z_name = random.choice(zombie_types)
        cost = self._get_zombie_cost(z_name)

        if self.zombie_moonlight >= cost and self.zombie_global_cd <= 0:
            row = random.randint(0, 4)
            if self.zombie_lane_cds[row] <= 0:
                self.createZombie(z_name, row)
                self.zombie_moonlight -= cost
                self.zombie_global_cd = 2.0
                self.zombie_lane_cds[row] = 5.0

    def _get_zombie_cost(self, zombie_name):
        """获取僵尸造价"""
        costs = {
            c.NORMAL_ZOMBIE: 50,
            c.CONEHEAD_ZOMBIE: 75,
            c.BUCKETHEAD_ZOMBIE: 125,
            c.NEWSPAPER_ZOMBIE: 100
        }
        return costs.get(zombie_name, 50)

    def createZombie(self, name, map_y):
        """创建僵尸"""
        x, y = self.map_obj.getMapGridPos(0, map_y)
        if name == c.NORMAL_ZOMBIE:
            self.zombie_groups[map_y].add(zombie.NormalZombie(c.ZOMBIE_START_X, y, self.head_group))
        elif name == c.CONEHEAD_ZOMBIE:
            self.zombie_groups[map_y].add(zombie.ConeHeadZombie(c.ZOMBIE_START_X, y, self.head_group))
        elif name == c.BUCKETHEAD_ZOMBIE:
            self.zombie_groups[map_y].add(zombie.BucketHeadZombie(c.ZOMBIE_START_X, y, self.head_group))
        elif name == c.NEWSPAPER_ZOMBIE:
            self.zombie_groups[map_y].add(zombie.NewspaperZombie(c.ZOMBIE_START_X, y, self.head_group))
        elif name == c.FLAG_ZOMBIE:
            self.zombie_groups[map_y].add(zombie.FlagZombie(c.ZOMBIE_START_X, y, self.head_group))
        else:
            print(f"未知僵尸类型: {name}")

    def canSeedPlant(self):
        x, y = pg.mouse.get_pos()
        return self.map_obj.showPlant(x, y)

    def addPlant(self):
        """种植植物"""
        pos = self.canSeedPlant()
        if pos is None:
            return

        if self.hint_image is None:
            self.setupHintImage()
        x, y = self.hint_rect.centerx, self.hint_rect.bottom
        map_x, map_y = self.map_obj.getMapIndex(x, y)

        # 根据植物名称创建植物
        if self.plant_name == c.SUNFLOWER:
            new_plant = plant.SunFlower(x, y, self.sun_group)
        elif self.plant_name == c.PEASHOOTER:
            new_plant = plant.PeaShooter(x, y, self.bullet_groups[map_y])
        elif self.plant_name == c.SNOWPEASHOOTER:
            new_plant = plant.SnowPeaShooter(x, y, self.bullet_groups[map_y])
        elif self.plant_name == c.WALLNUT:
            new_plant = plant.WallNut(x, y)
        elif self.plant_name == c.CHERRYBOMB:
            new_plant = plant.CherryBomb(x, y)
        elif self.plant_name == c.POTATOMINE:
            new_plant = plant.PotatoMine(x, y)
        elif self.plant_name == c.SQUASH:
            new_plant = plant.Squash(x, y)
        else:
            print(f"未知植物类型: {self.plant_name}")
            return

        if new_plant.can_sleep and self.background_type == c.BACKGROUND_DAY:
            new_plant.setSleep()

        self.plant_groups[map_y].add(new_plant)
        self.menubar.decreaseSunValue(self.select_plant.sun_cost)
        self.menubar.setCardFrozenTime(self.plant_name)
        self.map_obj.setMapGridType(map_x, map_y, c.MAP_EXIST)
        self.removeMouseImage()

    def setupHintImage(self):
        pos = self.canSeedPlant()
        if pos and self.mouse_image:
            if (self.hint_image and pos[0] == self.hint_rect.x and
                pos[1] == self.hint_rect.y):
                return
            width, height = self.mouse_rect.w, self.mouse_rect.h
            image = pg.Surface([width, height])
            image.blit(self.mouse_image, (0, 0), (0, 0, width, height))
            image.set_colorkey(c.BLACK)
            image.set_alpha(128)
            self.hint_image = image
            self.hint_rect = image.get_rect()
            self.hint_rect.centerx = pos[0]
            self.hint_rect.bottom = pos[1]
            self.hint_plant = True
        else:
            self.hint_plant = False

    def setupMouseImage(self, plant_name, select_plant):
        frame_list = tool.GFX[plant_name]
        if plant_name in tool.PLANT_RECT:
            data = tool.PLANT_RECT[plant_name]
            x, y, width, height = data['x'], data['y'], data['width'], data['height']
        else:
            x, y = 0, 0
            rect = frame_list[0].get_rect()
            width, height = rect.w, rect.h

        if (plant_name == c.POTATOMINE or plant_name == c.SQUASH):
            color = c.WHITE
        else:
            color = c.BLACK

        self.mouse_image = tool.get_image(frame_list[0], x, y, width, height, color, 1)
        self.mouse_rect = self.mouse_image.get_rect()
        pg.mouse.set_visible(False)
        self.drag_plant = True
        self.plant_name = plant_name
        self.select_plant = select_plant

    def removeMouseImage(self):
        pg.mouse.set_visible(True)
        self.drag_plant = False
        self.mouse_image = None
        self.hint_image = None
        self.hint_plant = False

    def checkBulletCollisions(self):
        collided_func = pg.sprite.collide_circle_ratio(0.7)
        for i in range(self.map_y_len):
            for bullet in self.bullet_groups[i]:
                if bullet.state == c.FLY:
                    zombie_obj = pg.sprite.spritecollideany(bullet, self.zombie_groups[i], collided_func)
                    if zombie_obj and zombie_obj.state != c.DIE:
                        zombie_obj.setDamage(bullet.damage, bullet.ice)
                        bullet.setExplode()

    def checkZombieCollisions(self):
        collided_func = pg.sprite.collide_circle_ratio(0.7)
        for i in range(self.map_y_len):
            for zombie_obj in self.zombie_groups[i]:
                if zombie_obj.state != c.WALK:
                    continue
                plant_obj = pg.sprite.spritecollideany(zombie_obj, self.plant_groups[i], collided_func)
                if plant_obj and plant_obj.name != c.SPIKEWEED:
                    zombie_obj.setAttack(plant_obj)

    def checkCarCollisions(self):
        collided_func = pg.sprite.collide_circle_ratio(0.8)
        for car in self.cars:
            zombies = pg.sprite.spritecollide(car, self.zombie_groups[car.map_y], False, collided_func)
            for zombie_obj in zombies:
                if zombie_obj and zombie_obj.state != c.DIE:
                    car.setWalk()
                    zombie_obj.setDie()
            if car.dead:
                self.cars.remove(car)

    def checkPlant(self, plant_obj, i):
        zombie_len = len(self.zombie_groups[i])

        if plant_obj.name == c.CHOMPER:
            for zombie_obj in self.zombie_groups[i]:
                if plant_obj.canAttack(zombie_obj):
                    plant_obj.setAttack(zombie_obj, self.zombie_groups[i])
                    break
        elif plant_obj.name == c.POTATOMINE:
            for zombie_obj in self.zombie_groups[i]:
                if plant_obj.canAttack(zombie_obj):
                    plant_obj.setAttack()
                    break
        elif plant_obj.name == c.SQUASH:
            for zombie_obj in self.zombie_groups[i]:
                if plant_obj.canAttack(zombie_obj):
                    plant_obj.setAttack(zombie_obj, self.zombie_groups[i])
                    break
        else:
            can_attack = False
            if (plant_obj.state == c.IDLE and zombie_len > 0):
                for zombie_obj in self.zombie_groups[i]:
                    if plant_obj.canAttack(zombie_obj):
                        can_attack = True
                        break
            if plant_obj.state == c.IDLE and can_attack:
                plant_obj.setAttack()
            elif (plant_obj.state == c.ATTACK and not can_attack):
                plant_obj.setIdle()

    def checkPlants(self):
        for i in range(self.map_y_len):
            for plant_obj in self.plant_groups[i]:
                if plant_obj.state != c.SLEEP:
                    self.checkPlant(plant_obj, i)
                if plant_obj.health <= 0:
                    self.killPlant(plant_obj)

    def killPlant(self, plant_obj):
        x, y = plant_obj.getPosition()
        map_x, map_y = self.map_obj.getMapIndex(x, y)
        self.map_obj.setMapGridType(map_x, map_y, c.MAP_EMPTY)
        plant_obj.kill()

    def checkGameState(self):
        # 检查失败条件（僵尸进屋）
        for i in range(self.map_y_len):
            for zombie_obj in self.zombie_groups[i]:
                if zombie_obj.rect.right < 0:
                    print("游戏失败：僵尸进屋了！")
                    self.next = c.GAME_LOSE
                    self.done = True
                    return

        # 检查胜利条件（所有僵尸都被消灭且AI资源耗尽）
        total_zombies = sum(len(group) for group in self.zombie_groups)
        if total_zombies == 0 and self.zombie_moonlight < 50:
            print("游戏胜利：成功抵御了僵尸AI的进攻！")
            self.next = c.GAME_VICTORY
            self.done = True

    def drawMouseShow(self, surface):
        if self.hint_plant:
            surface.blit(self.hint_image, self.hint_rect)
        x, y = pg.mouse.get_pos()
        self.mouse_rect.centerx = x
        self.mouse_rect.centery = y
        if self.mouse_image:
            surface.blit(self.mouse_image, self.mouse_rect)

    def draw(self, surface):
        """绘制游戏画面"""
        # 绘制背景
        self.level.blit(self.background, self.viewport, self.viewport)
        surface.blit(self.level, (0,0), self.viewport)

        # 绘制UI和游戏对象
        self.menubar.draw(surface)

        for i in range(self.map_y_len):
            self.plant_groups[i].draw(surface)
            self.zombie_groups[i].draw(surface)
            self.hypno_zombie_groups[i].draw(surface)
            self.bullet_groups[i].draw(surface)

        for car in self.cars:
            car.draw(surface)

        self.head_group.draw(surface)
        self.sun_group.draw(surface)

        # 绘制僵尸AI状态信息
        self.drawZombieAIInfo(surface)

        if self.drag_plant:
            self.drawMouseShow(surface)

    def drawZombieAIInfo(self, surface):
        """绘制僵尸AI的状态信息"""
        try:
            font = pg.font.Font(None, 28)

            # 绘制僵尸AI资源
            resource_text = font.render(f"僵尸AI资源: {int(self.zombie_moonlight)}", True, c.RED)
            surface.blit(resource_text, (650, 10))

            # 绘制冷却信息
            cool_text = font.render(f"全局CD: {self.zombie_global_cd:.1f}s", True, c.RED)
            surface.blit(cool_text, (650, 40))

            # 绘制行冷却
            for i in range(5):
                lane_text = font.render(f"第{i+1}行CD: {self.zombie_lane_cds[i]:.1f}s", True, c.RED)
                surface.blit(lane_text, (650, 70 + i*25))
        except Exception as e:
            # 如果绘制失败，忽略错误
            pass
