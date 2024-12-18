import torch
import torch.nn as nn

# 定义一个简单的3D卷积层
dilation = 1  # 可以更改此值来观察不同的扩展效果
conv3d_layer = nn.Conv3d(1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0)

# 创建一个示例输入张量 (1, 1, 10, 10, 10)
# 这里 (1, 1, 10, 10, 10) 分别是 (批次大小, 输入通道数, 深度, 高度, 宽度)
input_tensor = torch.randn(1, 1, 10, 10, 10)

# 通过3D卷积层得到输出
output_tensor = conv3d_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDBOQjpHRjEOmGXtWYA2g3w7DFVnqJd1iTM1fklDteIf/wL8IFkyZFYc1DB/abBjb7obHb1Kh1PY9k/xleHz3Shc+UOE6YWQZAB/Xbv9hvycEDd0OSellKkBM0K5MdUv4v7zi6A/H+xPPt8BG/eMApUGHtuahjN/0prqyy0LqnUiP4OLqYd5KwAvc8IPrb6
+1APfnzH2L/G1ShyWMdKMXeZvRjPf7Kj5IqjquiAn4NojN32V0vyEbsz04W3ESDrsjCYSbj0zaPPqe+A8zUEUzcgVeChxEYdvTcGF0o06T1ZnSd221wy334fJhowsn0UHjNfYCPryoUCoRjhcQs2T/FMZdZXPhHoDziCAblgCnXbsWrfavxW//VD0lgSGHP6LGIEgkyO2dJY541Vgyl1hzLlEiccK7ELeRhV
2R6qOLNr88Nn+M/oVUUp2iXKTo5nPCLnTSekfI8i/mPT0bTFGmbPxmfoQUVOcqFHK11hANFK5lbhsM/sxayixBEQhMgO70+D5IwH2k4hPeBmz4KutH766UDrL6ZS2eD6ZXfQT+fSuhCkSdaT9upBRT+3XGNSa+i1zjHN7xRa7MBMiuBYBE70oaSB2SMTFErN5SVwLdKB8grvvTVCZwUKWIi8F81x2U8SoqDZm0cSizm4Uz1rqGCWtpmBsfC9uTqeJLNyfafZQQ== a0979552111@gmail.com
