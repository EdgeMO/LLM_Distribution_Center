#!/bin/bash
# 保存为 open_port.sh

# 检查是否提供了端口号
if [ -z "$1" ]; then
    echo "用法: $0 <端口号>"
    exit 1
fi

PORT=$1

# 检查是否为 root 用户
if [ "$(id -u)" != "0" ]; then
   echo "错误: 此脚本需要 root 权限" 1>&2
   echo "请使用 sudo 运行" 1>&2
   exit 1
fi

# 检测防火墙类型并开放端口
if command -v firewall-cmd &> /dev/null; then
    echo "检测到 firewalld，正在配置..."
    firewall-cmd --permanent --add-port=${PORT}/tcp
    firewall-cmd --reload
    echo "端口 ${PORT}/tcp 已在 firewalld 中开放"
    
elif command -v ufw &> /dev/null && ufw status | grep -q "Status: active"; then
    echo "检测到活动的 ufw，正在配置..."
    ufw allow ${PORT}/tcp
    echo "端口 ${PORT}/tcp 已在 ufw 中开放"
    
else
    echo "使用 iptables 配置..."
    iptables -A INPUT -p tcp --dport ${PORT} -j ACCEPT
    
    # 尝试保存 iptables 规则
    if command -v netfilter-persistent &> /dev/null; then
        netfilter-persistent save
    elif [ -f /etc/redhat-release ]; then
        service iptables save
    else
        echo "警告: 无法自动保存 iptables 规则"
        echo "您的 iptables 规则可能在重启后丢失"
        echo "请手动保存 iptables 规则"
    fi
    
    echo "端口 ${PORT}/tcp 已在 iptables 中开放"
fi

# 验证端口是否开放
echo "验证端口状态..."
if command -v ss &> /dev/null; then
    echo "本地监听的 ${PORT} 端口:"
    ss -tulpn | grep ":${PORT}"
else
    echo "本地监听的 ${PORT} 端口:"
    netstat -tulpn | grep ":${PORT}"
fi

echo "完成！如果您的应用程序已在监听该端口，它现在应该可以从外部访问了。"


# ## 添加执行权限
# chmod +x open_port.sh

# # 开放端口 (如 9999)
# sudo ./open_port.sh 9999