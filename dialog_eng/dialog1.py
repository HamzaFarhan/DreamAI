system = """
Extract structured details about gadgets in this format:
{
  "device_type": "",
  "brand": "",
  "model": "",
  "key_features": []
}
"""

messages = [
    {
        "role": "user",
        "content": "Just picked up the new MacBook Air M3, love how thin it is and the battery lasts forever!",
    },
    {
        "role": "assistant",
        "content": {
            "device_type": "laptop",
            "brand": "Apple",
            "model": "MacBook Air M3",
            "key_features": ["thin design", "long battery life"],
        },
    },
    {
        "role": "user",
        "content": "My roommate's got this Sony headphone, the WH-1000XM4. The noise canceling is top-notch and it connects to multiple devices",
    },
    {
        "role": "assistant",
        "content": {
            "device_type": "headphones",
            "brand": "Sony",
            "model": "WH-1000XM4",
            "key_features": ["noise cancellation", "multi-device connectivity"],
        },
    },
    {
        "role": "user",
        "content": "Just got the new Samsung S24 Ultra, the AI features are mind-blowing and the zoom on the camera is insane!",
    },
]
