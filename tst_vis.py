import pygame
import sys

# 初始化Pygame
pygame.init()

# 设置屏幕大小
screen = pygame.display.set_mode((640, 480))

# 设置窗口标题
pygame.display.set_caption("Pygame Window")

# 主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 填充背景色
    screen.fill((0, 120, 230))

    # 更新显示
    pygame.display.flip()

# 退出Pygame
pygame.quit()
sys.exit()
