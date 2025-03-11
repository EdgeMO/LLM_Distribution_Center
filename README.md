
# LLM Distribution Center

This document describes the code implementation of the center. The main steps include:

1. Establish the connection between edge nodes and the center, register all edge nodes and keep the connection alive throughout the process.
2. Retrieve input data from the local environment and generate inputs for each round.
3. Perform 20 task offloading rounds before the model offloading round.
4. After the algorithm finishes, match the target edge ID with task offloading decisions and model caching decisions.
5. Call the transmission function to distribute specific tasks to the edge nodes.
6. Call the function for model transmission.
7. Once the model deployment is ready, prepare for the next round.

## Sample Data Structure

*(maintainer: @wxhfj)*

* **task_id** :
  The primary key.
* **task_type** :
* **0 TC (Text Classification)**
  * *reference_enum* :
  * 0: sad
  * 1: happy
  * 2: love
  * 3: angry
  * 4: scared
  * 5: surprise
* **1 NER (Named Entity Recognition)**
  * *reference_enum* :
  * O: None
  * B-`<ENTITY>`: Beginning part of certain words
  * I-`<ENTITY>`: Internal part of certain words
  * *Entity* : PER (Person Name), LOC (Location), ORG (Organization), MISC (Miscellaneous)
* **2 QA (Question Answering)**
  * *reference_value* : ideal standard answer
* **3 TS (Translation Chinese to English)**
  * *reference_value* : ideal standard answer
* **4 SG (Summarization Generation)**
  * *reference_value* : ideal standard answer
* **task_token** :
  Query words.

---

## Powershell Commands

Below are some commands to configure network settings using PowerShell.

### 1. Ping – 以管理员身份运行 PowerShell

```powershell
netsh advfirewall firewall add rule name="ICMP Allow incoming V4 echo request" protocol=icmpv4:8,any dir=in action=allow
```

### 2. wsl-port-forwarding.ps1

```powershell
# 获取 WSL2 IP 地址
$wslIP = (wsl hostname -I).Trim()
$port = 8888

# 删除现有规则
netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0

# 添加新规则
netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$wslIP

# 显示当前规则
netsh interface portproxy show all
```

### 3. 查看 WSL 内部绑定的 IP 地址

```bash
ip addr show eth0 | grep "inet\b" | awk '{print $2}' | cut -d/ -f1
```

### 4. 添加端口转发规则（使用手动获取的 WSL2 IP 地址）

```powershell
# 替换为您的 WSL2 IP 地址
$wslIP = "172.25.235.171"
$port = 8888

# 添加端口转发规则
netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$wslIP
```

#### 验证端口转发设置

```bash
netsh interface portproxy show all
```

#### 在 Windows 防火墙中添加入站规则

```powershell
New-NetFirewallRule -DisplayName "WSL2 TCP Port $port" -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port
```

---

## TCP Connection Test

The directory "close_test" is used for testing TCP connections. If the server is established but TCP connection is disconnected, you can check the firewall status with the following commands:

```bash
python tcp_server.py -p 8888
python tcp_server.py
```

---

## Firewalld Commands

Check if firewalld is running:

```bash
sudo systemctl status firewalld
```

Start firewalld:

```bash
sudo systemctl start firewalld
```

Enable firewalld at boot:

```bash
sudo systemctl enable firewalld
```

Stop firewalld:

```bash
sudo systemctl stop firewalld
```

Disable firewalld:

```bash
sudo systemctl disable firewalld
```

Open a single TCP port (e.g., 9999):

```bash
sudo firewall-cmd --permanent --add-port=9999/tcp
```

Open a range of ports (e.g., 8000-9000):

```bash
sudo firewall-cmd --permanent --add-port=8000-9000/tcp
```

Apply the changes:

```bash
sudo firewall-cmd --reload
```

Close a port:

```bash
sudo firewall-cmd --permanent --remove-port=9999/tcp
sudo firewall-cmd --reload
```

---

## UFW Commands

### 1. 安装 UFW

```bash
sudo apt update
sudo apt install ufw
```

### 2. 检查 UFW 状态

```bash
sudo ufw status
```

### 3. 启用 UFW

（首次启用前确保已添加允许 SSH 的规则）

```bash
sudo ufw enable
```

### 4. 禁用 UFW

```bash
sudo ufw disable
```

### 5. 开放单个 TCP 端口（如 9999）

```bash
sudo ufw allow 9999/tcp
```

### 6. 开放端口范围

```bash
sudo ufw allow 8000:9000/tcp
```

### 7. 删除规则

```bash
sudo ufw delete allow 9999/tcp
```

### 8. 配置基本规则

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 9999/tcp
sudo ufw enable
```

---

## 验证端口是否开放

1. 使用 netstat 检查本地监听端口：
   ```bash
   sudo netstat -tulpn | grep 9999
   ```
2. 使用 ss 命令检查（更现代的工具）：
   ```bash
   sudo ss -tulpn | grep 9999
   ```
3. 从另一台机器使用 nmap 检查：
   ```bash
   nmap -p 9999 <服务器IP地址>
   ```
