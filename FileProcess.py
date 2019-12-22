from langconv import *
import sys

print(sys.version)
print(sys.version_info)

# 转换繁体到简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line

# 转换简体到繁体
def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line

# line_chs='<>123asdasd把中文字符串进行繁体和简体中文的转换'
# line_cht='<>123asdasd把中文字符串進行繁體和簡體中文的轉換'
#
# ret_chs = "%s\n"%cht_to_chs(line_cht)
# ret_cht = "%s\n"%chs_to_cht(line_chs)
#
# print("chs='%s'",ret_cht)
# print("cht='%s'",ret_chs)
#
# file = open('ret.txt','w',encoding='utf-8')
# file.write(ret_chs)
# file.write(ret_cht)
# file.close()
with open('data/七绝1.txt', 'r', encoding='utf-8') as f1:
    with open('data/七绝.txt', 'w', encoding='utf-8') as f:
        for line in f1.readlines():
            one = line.split(':')[-1]
            one = cht_to_chs(one)
            f.write(one)
