import os
import wexpect
import subprocess

def transfer_json_to_remote(local_json_file, remote_host, remote_user, remote_dir, remote_password, remote_port=22):
    """使用 wexpect 库自动执行 scp 命令并输入密码"""
    try:
        # 构造 scp 命令
        scp_command = f'scp -P {remote_port} "{local_json_file}" {remote_user}@{remote_host}:"{remote_dir}"'
        
        # 创建 wexpect 子进程
        child = wexpect.spawn(scp_command)
        
        # 期望出现的提示信息
        index = child.expect(['password:', 'Are you sure you want to continue connecting (yes/no/[fingerprint])?'], timeout=30)
        
        # 如果是首次连接，会提示是否继续连接
        if index == 1:
            child.sendline('yes')
            child.expect('password:', timeout=30)
        
        # 输入密码
        child.sendline(remote_password)
        
        # 等待命令执行完成
        child.expect(wexpect.EOF, timeout=30)
        
        # 检查命令是否成功
        if child.exitstatus == 0:
            print(f"文件成功传输到远程主机：{remote_host}")
        else:
            print(f"文件传输失败，错误信息：{child.before}")
            
        child.close()
    except Exception as e:
        print(f"文件传输失败，错误信息：{str(e)}")

def fetch_json_from_remote(remote_json_file, local_dir, remote_host, remote_user, remote_password, remote_port=22):
    """使用 wexpect 库自动执行 scp 命令并输入密码"""
    try:
        # 确保本地目录存在
        os.makedirs(local_dir, exist_ok=True)
        
        # 构造 scp 命令
        scp_command = 'scp -P {} {}@{}:"{}" "{}"'.format(
            remote_port,
            remote_user,
            remote_host,
            remote_json_file,
            local_dir
        )
        
        # 创建 wexpect 子进程
        child = wexpect.spawn(scp_command)
        
        # 期望出现的提示信息
        index = child.expect(['password:', 'Are you sure you want to continue connecting (yes/no/[fingerprint])?'], timeout=30)
        
        # 如果是首次连接，会提示是否继续连接
        if index == 1:
            child.sendline('yes')
            child.expect('password:', timeout=30)
        
        # 输入密码
        child.sendline(remote_password)
        
        # 等待命令执行完成
        child.expect(wexpect.EOF, timeout=30)
        
        # 检查命令是否成功
        if child.exitstatus == 0:
            print(f"文件成功从远程主机拉取：{remote_host}")
        else:
            print(f"文件拉取失败，错误信息：{child.before}")
            
        child.close()
    except Exception as e:
        print(f"文件拉取失败，错误信息：{str(e)}")

if __name__ == '__main__':
    local_json_file = r'E:\SRTP_projects\Mindspore-BERT-NER\backend\user_chat\肖一鸣_chat_results.json'
    remote_host = '10.49.227.82'  # 远程主机的IP地址
    remote_user = 'songjh'        # 远程主机的用户名
    remote_dir = '/home/songjh/bert_1.9/user_chat'  # 远程主机的目标目录
    remote_password = 'bme106'  # 远程主机的密码
    remote_port = 22              # SSH端口，默认是22，如果不是请修改
    remote_json_file = '/home/songjh/bert_1.9/user_chat/大追追的观众姥爷们_chat_results.json'  # 远程 JSON 文件路径
    local_dir = 'E:\\SRTP_projects\\Mindspore-BERT-NER\\backend\\receive_file'  # 本地保存目录
    
    # transfer_json_to_remote(local_json_file, remote_host, remote_user, remote_dir, remote_password, remote_port)
    fetch_json_from_remote(remote_json_file, local_dir, remote_host, remote_user, remote_password, remote_port)