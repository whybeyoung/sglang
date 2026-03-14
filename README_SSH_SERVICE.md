# SSH配置Web服务

这是一个简单的Python Web服务，用于通过HTTP接口执行SSH配置脚本。

## 功能

- 修改SSH端口为5022
- 允许密码登录
- 允许root登录
- 修改root密码
- 重启SSH服务

## 安装依赖

```bash
pip install -r requirements_ssh_service.txt
```

## 运行服务

```bash
# 直接运行
python ssh_config_service.py

# 或指定端口和主机
PORT=8080 HOST=0.0.0.0 python ssh_config_service.py
```

## API端点

### 1. 直接下载并执行脚本（推荐）
```bash
# 直接执行（最简单的方式）
curl http://server:port/script.sh | bash

# 或下载后执行
curl http://server:port/script.sh -o script.sh && bash script.sh
```

### 2. 服务端执行SSH配置脚本
```bash
# GET请求
curl http://localhost:5000/config-ssh

# POST请求
curl -X POST http://localhost:5000/config-ssh
```

### 3. 健康检查
```bash
curl http://localhost:5000/health
```

### 4. 查看API说明
```bash
curl http://localhost:5000/
```

## 注意事项

⚠️ **重要安全提示**：

1. **需要sudo权限**：此服务需要sudo权限来执行SSH配置脚本
2. **安全风险**：此服务会修改SSH配置和root密码，请确保：
   - 仅在受信任的网络环境中使用
   - 添加适当的身份验证（建议在生产环境中添加API密钥或Token验证）
   - 考虑使用HTTPS而不是HTTP
   - 限制访问IP地址

3. **建议改进**：
   - 添加API密钥验证
   - 添加IP白名单
   - 使用HTTPS
   - 添加请求日志和审计

## 示例响应

成功响应：
```json
{
  "success": true,
  "message": "SSH配置脚本执行成功",
  "stdout": "修改完成，原文件已备份为 /etc/ssh/sshd_config.bak.2024-01-01-12:00:00\n",
  "stderr": ""
}
```

失败响应：
```json
{
  "success": false,
  "message": "SSH配置脚本执行失败",
  "stdout": "",
  "stderr": "错误信息...",
  "returncode": 1
}
```




