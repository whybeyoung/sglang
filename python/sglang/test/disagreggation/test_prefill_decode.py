import asyncio
import json
import sys
import aiohttp
import random
import urllib.parse
import time

def _generate_bootstrap_room():
    return random.randint(0, 2**63 - 1)

def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None

def add_bootstrap_info(request, prefill_url, decode_url):
    """Add bootstrap information to the request with the same room_id for both servers"""
    prefill_parsed = urllib.parse.urlparse(prefill_url)
    prefill_hostname = prefill_parsed.hostname

    # Generate room_id once to be used by both servers
    room_id = _generate_bootstrap_room()

    batch_size = _get_request_batch_size(request)

    if batch_size is not None:
        prefill_request = request.copy()
        decode_request = request.copy()

        prefill_request.update({
            "bootstrap_host": [prefill_hostname] * batch_size,
            "bootstrap_room": [room_id] * batch_size
        })

        decode_request.update({
            "bootstrap_host": [prefill_hostname] * batch_size,  # Use prefill hostname for decode server too
            "bootstrap_room": [room_id] * batch_size
        })
    else:
        prefill_request = request.copy()
        decode_request = request.copy()

        prefill_request.update({
            "bootstrap_host": prefill_hostname,
            "bootstrap_room": room_id
        })

        decode_request.update({
            "bootstrap_host": prefill_hostname,  # Use prefill hostname for decode server too
            "bootstrap_room": room_id
        })

    return prefill_request, decode_request

async def test_generate(prefill_url, decode_url):
    """Test the generate endpoint by sending requests to both prefill and decode servers"""
    test_request = {
        "text": "Hello, how are you?",
        "max_new_tokens": 10,
        "temperature": 0.7
    }

    # Add bootstrap information for both servers with the same room_id
    prefill_request, decode_request = add_bootstrap_info(test_request, prefill_url, decode_url)

    print(f"Sending generate request to prefill server {prefill_url} with data:", json.dumps(prefill_request, indent=2))
    print(f"Sending generate request to decode server {decode_url} with data:", json.dumps(decode_request, indent=2))

    timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            start_time = time.time()
            # Send requests to both servers simultaneously
            tasks = [
                session.post(f"{prefill_url}/generate", json=prefill_request),
                session.post(f"{decode_url}/generate", json=decode_request)
            ]
            prefill_response, decode_response = await asyncio.gather(*tasks)

            print(f"Got responses after {time.time() - start_time:.2f} seconds")
            print(f"Prefill server response status: {prefill_response.status}")
            print(f"Decode server response status: {decode_response.status}")

            if decode_response.status == 200:
                result = await decode_response.json()
                print(f"Generate response from decode server:", json.dumps(result, indent=2))
            else:
                print(f"Decode server error: {await decode_response.text()}")

        except asyncio.TimeoutError:
            print("Generate request timeout")
            raise
        except Exception as e:
            print(f"Generate request failed: {str(e)}")
            raise

async def test_chat_completions(prefill_url, decode_url):
    """Test the chat completions endpoint by sending requests to both prefill and decode servers"""
    test_request = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 10,
        "temperature": 0.7
    }

    # Add bootstrap information for both servers with the same room_id
    prefill_request, decode_request = add_bootstrap_info(test_request, prefill_url, decode_url)

    print(f"Sending chat completions request to prefill server {prefill_url} with data:", json.dumps(prefill_request, indent=2))
    print(f"Sending chat completions request to decode server {decode_url} with data:", json.dumps(decode_request, indent=2))

    timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            start_time = time.time()
            # Send requests to both servers simultaneously
            tasks = [
                session.post(f"{prefill_url}/v1/chat/completions", json=prefill_request),
                session.post(f"{decode_url}/v1/chat/completions", json=decode_request)
            ]
            prefill_response, decode_response = await asyncio.gather(*tasks)

            print(f"Got responses after {time.time() - start_time:.2f} seconds")
            print(f"Prefill server response status: {prefill_response.status}")
            print(f"Decode server response status: {decode_response.status}")

            if decode_response.status == 200:
                result = await decode_response.json()
                print(f"Chat completions response from decode server:", json.dumps(result, indent=2))
            else:
                print(f"Decode server error: {await decode_response.text()}")

        except asyncio.TimeoutError:
            print("Chat completions request timeout")
            raise
        except Exception as e:
            print(f"Chat completions request failed: {str(e)}")
            raise

async def run_tests(prefill_addrs, decode_addrs):
    """Run tests for all server pairs"""
    for prefill_url, decode_url in zip(prefill_addrs, decode_addrs):
        print(f"\nTesting server pair:")
        print(f"Prefill server: {prefill_url}")
        print(f"Decode server: {decode_url}")

        try:
            # Choose one of these to test:
            await test_generate(prefill_url, decode_url)
            # await test_chat_completions(prefill_url, decode_url)
        except Exception as e:
            print(f"Failed to test server pair: {str(e)}")
            continue

def main():
    if len(sys.argv) != 3:
        print("Usage: python test_lb.py <prefill_addrs> <decode_addrs>")
        print("Example: python test_lb.py http://localhost:8001 http://localhost:8002")
        sys.exit(1)

    prefill_addrs = sys.argv[1].split(",")
    decode_addrs = sys.argv[2].split(",")

    if len(prefill_addrs) != len(decode_addrs):
        print("Error: Number of prefill servers must match number of decode servers")
        sys.exit(1)

    asyncio.run(run_tests(prefill_addrs, decode_addrs))

if __name__ == "__main__":
    main()
