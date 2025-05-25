import httpx
import json
import asyncio

url = "http://127.0.0.1:8002/generate"
headers = {"Content-Type": "application/json"}
data = {
    "text": "The capital of France is ",
    "rid": "cccccc",
    "stream": True,  # ✅ 启用流式响应
    "sampling_params": {
        "temperature": 0.5
    }
}

async def main():
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=data) as response:
            async for line in response.aiter_lines():
                if line.strip():
                    print("Received:", line)

if __name__ == "__main__":
    asyncio.run(main())
