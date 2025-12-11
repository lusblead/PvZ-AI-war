"""
对战启动文件
"""
import os
import sys

# 设置路径
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
source_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(source_dir)

os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pygame as pg
import source.constants as c
import source.tool as tool

def main():
    """主函数"""
    # 初始化游戏
    pg.init()
    screen = pg.display.set_mode(c.SCREEN_SIZE)
    pg.display.set_caption("植物大战僵尸 - 人类玩家 vs 僵尸AI")
    clock = pg.time.Clock()

    # 创建对战关卡
    from source.rl.battle_level import HumanVsAILevel

    # 创建游戏控制器
    controller = tool.Control()

    # 设置状态字典
    state_dict = {
        c.LEVEL: HumanVsAILevel(),
        c.GAME_LOSE: tool.State(),
        c.GAME_VICTORY: tool.State()
    }

    # 启动游戏
    controller.setup_states(state_dict, c.LEVEL)

    # 主循环
    while not controller.done:
        # 事件处理
        for event in pg.event.get():
            if event.type == pg.QUIT:
                controller.done = True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    controller.done = True
                controller.keys = pg.key.get_pressed()
            elif event.type == pg.KEYUP:
                controller.keys = pg.key.get_pressed()
            elif event.type == pg.MOUSEBUTTONDOWN:
                controller.mouse_pos = pg.mouse.get_pos()
                controller.mouse_click[0], _, controller.mouse_click[1] = pg.mouse.get_pressed()

        # 更新游戏状态
        controller.update()

        # 刷新显示
        pg.display.flip()
        clock.tick(60)

    pg.quit()

if __name__ == "__main__":
    main()
