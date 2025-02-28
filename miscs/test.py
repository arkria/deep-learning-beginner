import matplotlib.pyplot as plt
import numpy as np

# 创建一些示例灰度图像数据
image1 = np.random.rand(100, 100)  # 第一张灰度图像
image2 = np.random.rand(100, 100)  # 第二张灰度图像

# 创建 2x3 的主子图布局
fig, axes = plt.subplots(2, 3, figsize=(12, 4))  # figsize 可以调整整个图的大小

# 遍历每个主子图
for i, ax in enumerate(axes.flatten()):
    # 在每个主子图中创建两个更小的子图（并列显示）
    ax1 = ax.inset_axes([0, 0, 0.45, 1])  # 左侧子图
    ax2 = ax.inset_axes([0.45, 0, 0.45, 1])  # 右侧子图

    # 在左侧子图中绘制第一张灰度图像
    ax1.imshow(image1, cmap='gray')
    ax1.set_title("Image 1", fontsize=8)
    ax1.axis('off')  # 关闭坐标轴

    # 在右侧子图中绘制第二张灰度图像
    ax2.imshow(image2, cmap='gray')
    ax2.set_title("Image 2", fontsize=8)
    ax2.axis('off')  # 关闭坐标轴

    # 设置主子图的标题
    ax.set_title(f'Subplot {i + 1}', fontsize=10)
    ax.axis('off')  # 关闭主子图的坐标轴

# 调整每个子图之间的间距
plt.tight_layout()
plt.savefig('./img.png')  # 保存整个图
