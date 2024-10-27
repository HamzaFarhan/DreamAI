system = """
Extract structured details about gadgets in this format:
{
  "device_type": "",
  "brand": "",
  "model": "",
  "key_features": []
}

Examples:

input: Just picked up the new MacBook Air M3, love how thin it is and the battery lasts forever!
output: {
  "device_type": "laptop",
  "brand": "Apple",
  "model": "MacBook Air M3",
  "key_features": ["thin design", "long battery life"]
}

input: My roommate's got this Sony headphone, the WH-1000XM4. The noise canceling is top-notch and it connects to multiple devices
output: {
  "device_type": "headphones",
  "brand": "Sony",
  "model": "WH-1000XM4",
  "key_features": ["noise cancellation", "multi-device connectivity"]
}
"""

messages = [
    {
        "role": "user",
        "content": "Just got the new Samsung S24 Ultra, the AI features are mind-blowing and the zoom on the camera is insane!",
    }
]



