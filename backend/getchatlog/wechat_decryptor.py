import os
import ctypes
import hashlib
import hmac
import shutil
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import sqlite3
import json
import argparse
import csv
import pandas as pd
from pymem import Pymem, process

# 默认参数
DEFAULT_ITER = 64000
KEY_SIZE = 32
DEFAULT_PAGESIZE = 4096
SQLITE_FILE_HEADER = "SQLite format 3"

# 加载Windows API函数
ReadProcessMemory = ctypes.windll.kernel32.ReadProcessMemory
ReadProcessMemory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
ReadProcessMemory.restype = ctypes.c_int

void_p = ctypes.c_void_p

def get_key(db_path, addr_len):
    """获取微信数据库加密密钥"""
    def read_key_bytes(h_process, address, address_len=8):
        array = ctypes.create_string_buffer(address_len)
        if ReadProcessMemory(h_process, void_p(address), array, address_len, 0) == 0: return "None"
        address = int.from_bytes(array, byteorder='little')  # 逆序转换为int地址（key地址）
        key = ctypes.create_string_buffer(32)
        if ReadProcessMemory(h_process, void_p(address), key, 32, 0) == 0: return "None"
        key_bytes = bytes(key)
        return key_bytes

    def verify_key(key, wx_db_path):
        """验证密钥是否正确"""
        if not wx_db_path or wx_db_path.lower() == "none":
            return True
        with open(wx_db_path, "rb") as file:
            blist = file.read(5000)
        salt = blist[:16]
        byteKey = hashlib.pbkdf2_hmac("sha1", key, salt, DEFAULT_ITER, KEY_SIZE)
        first = blist[16:DEFAULT_PAGESIZE]

        mac_salt = bytes([(salt[i] ^ 58) for i in range(16)])
        mac_key = hashlib.pbkdf2_hmac("sha1", byteKey, mac_salt, 2, KEY_SIZE)
        hash_mac = hmac.new(mac_key, first[:-32], hashlib.sha1)
        hash_mac.update(b'\x01\x00\x00\x00')

        if hash_mac.digest() != first[-32:-12]:
            return False
        return True

    phone_type1 = "iphone\x00"
    phone_type2 = "android\x00"
    phone_type3 = "ipad\x00"

    pm = Pymem("WeChat.exe")
    module_name = "WeChatWin.dll"

    MicroMsg_path = os.path.join(db_path, "msg", "MicroMsg.db")

    type1_addrs = pm.pattern_scan_module(phone_type1.encode(), module_name, return_multiple=True)
    type2_addrs = pm.pattern_scan_module(phone_type2.encode(), module_name, return_multiple=True)
    type3_addrs = pm.pattern_scan_module(phone_type3.encode(), module_name, return_multiple=True)
    type_addrs = type1_addrs if len(type1_addrs) >= 2 else type2_addrs if len(type2_addrs) >= 2 else type3_addrs if len(type3_addrs) >= 2 else "None"
    if type_addrs == "None":
        return "None"
    for i in type_addrs[::-1]:
        for j in range(i, i - 2000, -addr_len):
            key_bytes = read_key_bytes(pm.process_handle, j, addr_len)
            if key_bytes == "None":
                continue
            if db_path != "None" and verify_key(key_bytes, MicroMsg_path):
                return key_bytes.hex()
    return "None"

def decrypt_database(key: str, db_path, out_path):
    """通过密钥解密数据库"""
    if not os.path.exists(db_path) or not os.path.isfile(db_path):
        return False, f"[-] db_path:'{db_path}' File not found!"
    if not os.path.exists(os.path.dirname(out_path)):
        return False, f"[-] out_path:'{out_path}' File not found!"
    if len(key) != 64:
        return False, f"[-] key:'{key}' Len Error!"

    password = bytes.fromhex(key.strip())
    with open(db_path, "rb") as file:
        blist = file.read()

    salt = blist[:16]
    byteKey = hashlib.pbkdf2_hmac("sha1", password, salt, DEFAULT_ITER, KEY_SIZE)
    first = blist[16:DEFAULT_PAGESIZE]
    if len(salt) != 16:
        return False, f"[-] db_path:'{db_path}' File Error!"

    mac_salt = bytes([(salt[i] ^ 58) for i in range(16)])
    mac_key = hashlib.pbkdf2_hmac("sha1", byteKey, mac_salt, 2, KEY_SIZE)
    hash_mac = hmac.new(mac_key, first[:-32], hashlib.sha1)
    hash_mac.update(b'\x01\x00\x00\x00')

    if hash_mac.digest() != first[-32:-12]:
        return False, f"[-] Key Error! (key:'{key}'; db_path:'{db_path}'; out_path:'{out_path}' )"

    newblist = [blist[i:i + DEFAULT_PAGESIZE] for i in range(DEFAULT_PAGESIZE, len(blist), DEFAULT_PAGESIZE)]

    with open(out_path, "wb") as deFile:
        deFile.write(SQLITE_FILE_HEADER.encode())
        t = AES.new(byteKey, AES.MODE_CBC, first[-48:-32])
        decrypted = t.decrypt(first[:-48])
        deFile.write(decrypted)
        deFile.write(first[-48:])

        for i in newblist:
            t = AES.new(byteKey, AES.MODE_CBC, i[-48:-32])
            decrypted = t.decrypt(i[:-48])
            deFile.write(decrypted)
            deFile.write(i[-48:])
    return True, [db_path, out_path, key]

def parse_db(key, db_path, output_dir):
    """解析数据库文件"""
    os.makedirs(output_dir, exist_ok=True)
    tasks = []
    for root, dirs, files in os.walk(db_path):
        for file in files:
            if '.db' == file[-3:]:
                if 'xInfo.db' == file:
                    continue
                inpath = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                tasks.append([key, inpath, output_path])
            else:
                try:
                    name, suffix = file.split('.')
                    if suffix.startswith('db_SQLITE'):
                        inpath = os.path.join(root, file)
                        output_path = os.path.join(output_dir, name + '.db')
                        tasks.append([key, inpath, output_path])
                except:
                    continue
    for i, task in enumerate(tasks):
        flag, msg = decrypt_database(*task)
        print(f"[{i+1}/{len(tasks)}] {flag} {msg}")

def merge_databases(source_databases, target_database):
    """合并数据库"""
    if os.path.exists(target_database):
        os.remove(target_database)
    # 复制第一个数据库作为模板
    if source_databases and os.path.exists(source_databases[0]):
        shutil.copy2(source_databases[0], target_database)
    # 合并其他数据库
    for db in source_databases[1:]:
        if os.path.exists(db):
            conn = sqlite3.connect(target_database)
            cursor = conn.cursor()
            try:
                # 附加其他数据库
                cursor.execute(f"ATTACH DATABASE '{db}' AS source_db")
                # 获取所有表名
                cursor.execute("SELECT name FROM source_db.sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                for table in tables:
                    table_name = table[0]
                    # 复制表结构
                    cursor.execute(f"SELECT sql FROM source_db.sqlite_master WHERE type='table' AND name='{table_name}'")
                    create_sql = cursor.fetchone()[0]
                    try:
                        cursor.execute(f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                        if cursor.fetchone() is None:
                            cursor.execute(create_sql)
                    except:
                        cursor.execute(create_sql)
                    # 复制表数据
                    cursor.execute(f"INSERT INTO {table_name} SELECT * FROM source_db.{table_name}")
                conn.commit()
            except Exception as e:
                print(f"合并数据库时出错: {e}")
            finally:
                cursor.execute("DETACH DATABASE source_db")
                conn.close()

def export_messages(decrypted_db_path, output_path, output_format='csv'):
    """导出消息到文件"""
    try:
        conn = sqlite3.connect(decrypted_db_path)
        cursor = conn.cursor()
        
        # 查询消息表
        cursor.execute('SELECT * FROM Message')
        columns = [description[0] for description in cursor.description]
        
        messages = []
        for row in cursor.fetchall():
            message = {col: val for col, val in zip(columns, row)}
            messages.append(message)
        
        conn.close()
        
        if output_format == 'csv':
            df = pd.DataFrame(messages)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif output_format == 'excel':
            df = pd.DataFrame(messages)
            df.to_excel(output_path, index=False)
        elif output_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(messages, f, ensure_ascii=False, indent=4)
        else:
            print("不支持的输出格式，请使用 csv, excel 或 json")
            return False
            
        return True
    except Exception as e:
        print(f"导出消息失败：{e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='微信聊天记录解密工具')
    parser.add_argument('--wechat-path', default='E:\\Wechat_files\\WeChat Files', required=True, help='微信安装目录路径')
    parser.add_argument('--output-dir', default='E:\\chatdownloadtry', help='输出目录路径，默认为wechat_decrypt_output')
    parser.add_argument('--export-file', default='E:\\chatdownloadtry', help='导出文件路径，默认为wechat_messages.csv')
    parser.add_argument('--export-format', choices=['csv', 'excel', 'json'], default='csv', help='导出文件格式')
    
    args = parser.parse_args()
    
    wechat_dir = args.wechat_path
    output_dir = args.output_dir
    export_file = args.export_file
    export_format = args.export_format
    
    # 获取加密密钥
    key_hex = get_key(wechat_dir, 8)
    if key_hex == "None":
        print("未能获取加密密钥")
        sys.exit(1)
    
    print(f"获取到加密密钥：{key_hex}")
    
    # 解密所有数据库
    db_path = os.path.join(wechat_dir, 'msg')
    if not os.path.exists(db_path):
        print(f"未找到数据库目录：{db_path}")
        sys.exit(1)
    
    print(f"开始解密数据库到目录：{output_dir}")
    parse_db(key_hex, db_path, output_dir)
    
    # 合并数据库
    target_database = os.path.join(output_dir, 'MSG.db')
    source_databases = [os.path.join(output_dir, f"MSG{i}.db") for i in range(1, 100) if os.path.exists(os.path.join(output_dir, f"MSG{i}.db"))]
    merge_databases(source_databases, target_database)
    
    # 导出消息
    if export_file:
        print(f"开始导出消息到文件：{export_file}")
        if export_messages(target_database, export_file, export_format):
            print(f"成功导出消息到：{export_file}")
        else:
            print("消息导出失败")

if __name__ == '__main__':
    main()