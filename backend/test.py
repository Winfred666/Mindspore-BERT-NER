import re

def format_message(msg):
    """格式化消息，处理换行符以及引用消息"""
    formatted_msg = msg.replace('\n', '[换行]')
    
    # 检测并处理引用消息
    # 正则表达式修正为匹配 "[换行]引用 的消息 : " 模式
    pattern = r'\[换行\]引用\s+的消息\s*:\s*(.*)'
    quotation_match = re.search(pattern, formatted_msg)
    if quotation_match:
        # 提取引用部分
        referenced_message = quotation_match.group(1)
        # 删除引用部分
        formatted_msg = formatted_msg[:quotation_match.start()]
        # 标记为引用消息
        return formatted_msg, True, referenced_message
    
    return formatted_msg, False, None

# 示例消息
msg = "data/songjh/ONNX/[换行]引用  的消息 : 它的onnx文件是哪一个"

# 调用 format_message 函数
formatted_msg, is_quotation, referenced_message = format_message(msg)

print(f"格式化后的消息: {formatted_msg}")
print(f"是否为引用消息: {is_quotation}")
print(f"引用的消息: {referenced_message}")