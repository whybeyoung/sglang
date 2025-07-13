# FastAPI 健康状态接口

这是一个使用 FastAPI 实现的健康状态报告接口，对应你提供的 `report_health` 函数。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务器

```bash
python simple_health_api.py
```

服务器将在 `http://localhost:8000` 启动。

## API 文档

启动服务器后，可以在以下地址查看自动生成的 API 文档：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 接口说明

### POST /health

接收健康状态报告。

**请求体：**
```json
{
    "status": "healthy",  // 可选值: healthy, unhealthy, starting, stopping
    "msg": "服务运行正常"   // 可选的消息描述
}
```

**响应：**
```json
{
    "success": true,
    "received_status": "healthy",
    "message": "Health status healthy received successfully"
}
```

## 客户端使用示例

```python
from simple_health_api import ServerStatus, report_health

# 报告健康状态
result = report_health(
    status=ServerStatus.HEALTHY, 
    host="localhost", 
    http_port=8000, 
    msg="服务运行正常"
)
print(result)
```

## 测试

运行测试脚本：

```bash
# 先启动服务器
python simple_health_api.py

# 在另一个终端运行测试
python test_health_api.py
```

## 服务器状态类型

- `HEALTHY`: 健康
- `UNHEALTHY`: 不健康
- `STARTING`: 启动中
- `STOPPING`: 停止中

## 高级功能版本

如果需要更多功能（如状态持久化、告警、监控等），可以查看 `health_api_example.py` 文件中的完整实现。
