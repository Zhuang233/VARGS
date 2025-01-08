import re

# 输入文件名和输出文件名
input_file = "./datasets/shapenet_split/train.txt"  # 替换为你的输入文件名
output_file = "./datasets/shapenet_split/train_debug.txt"  # 替换为你的输出文件名

# 定义正则表达式
pattern = re.compile(r"02954340-\w*\.ply")

# 打开输入文件和输出文件
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # 检查每一行是否匹配正则表达式
        if pattern.search(line):
            # 如果匹配，写入到输出文件
            outfile.write(line)

print(f"提取完成，符合条件的行已写入 '{output_file}'")
