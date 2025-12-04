#!/usr/bin/env python3
"""
脚本功能：将高分辨率图片转换为1/2分辨率
使用方法：python downscale_image.py <输入图片路径> [--output <输出路径>]
"""

import cv2
import argparse
import os
from pathlib import Path


def downscale_image(input_path, output_path=None, scale=0.5):
    """
    将图片缩放到指定分辨率
    
    Args:
        input_path (str): 输入图片路径
        output_path (str): 输出图片路径，默认为None则保存为同名文件加_half后缀
        scale (float): 缩放因子，默认0.5表示1/2分辨率
    """
    # 读取图片
    if not os.path.exists(input_path):
        print(f"错误：文件不存在 - {input_path}")
        return False
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"错误：无法读取图片 - {input_path}")
        return False
    
    # 获取原始分辨率
    height, width = img.shape[:2]
    print(f"原始分辨率: {width}x{height}")
    
    # 计算新的分辨率
    new_width = int(width * scale)
    new_height = int(height * scale)
    print(f"目标分辨率: {new_width}x{new_height}")
    
    # 使用高质量缩放
    downscaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 确定输出路径
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}-4{input_file.suffix}"
    
    # 保存图片
    success = cv2.imwrite(str(output_path), downscaled_img)
    if success:
        print(f"✓ 成功保存到: {output_path}")
        return True
    else:
        print(f"错误：无法保存图片到 {output_path}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="将高分辨率图片转换为1/2分辨率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python downscale_image.py image.jpg
  python downscale_image.py image.jpg --output downscaled.jpg
  python downscale_image.py image.jpg --scale 0.125  # 转换为1/4分辨率
        """)
    
    parser.add_argument('input', help='输入图片路径')
    parser.add_argument('--output', '-o', default=None, help='输出图片路径（默认为原文件名加_half后缀）')
    parser.add_argument('--scale', '-s', type=float, default=0.25, 
                       help='缩放因子（默认0.5表示1/2分辨率）')
    
    args = parser.parse_args()
    
    downscale_image(args.input, args.output, args.scale)


if __name__ == '__main__':
    main()
