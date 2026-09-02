import httpx
import asyncio
import os

os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

async def test():
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post("http://127.0.0.1:8001/api/v1/search", json={"query": "test", "workspace_id": 0})
        print(resp.status_code)
        print(resp.text)

asyncio.run(test())
