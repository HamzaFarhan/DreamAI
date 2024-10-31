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
    #  ALL previous messages
    {
        "role": "user",
        "content": "Just checked out the iPad Mini 7 but I'm really disappointed. The jelly scroll issue hasn't been fixed and there's barely any meaningful upgrades. No need to upgrade.",
    },
    {
        "role": "assistant",
        "content": {
            "device_type": "tablet",
            "brand": "Apple",
            "model": "iPad Mini 7",
            "key_features": ["jelly scroll issue persists", "minimal upgrades"],
            "sentiment": "negative",
        },
    },
]


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
        "content": "Got these amazing AirPods Pro 2 yesterday, the noise canceling and spatial audio are fantastic!",
    },
    {
        "role": "assistant",
        "content": {
            "device_type": "headphones",
            "brand": "Apple",
            "model": "AirPods 2",
            "key_features": ["noise cancellation", "spatial audio"],
        },
    },
    {
        "role": "user",
        "content": "That's not right - they're AirPods Pro 2, not AirPods 2. Those are different models.",
    },
    {
        "role": "assistant",
        "content": {
            "device_type": "earbuds",
            "brand": "Apple",
            "model": "AirPods Pro 2",
            "key_features": ["noise cancellation", "spatial audio"],
        },
    },
    {
        "role": "user",
        "content": "Just got the new Samsung S24 Ultra, the AI features are mind-blowing and the zoom on the camera is insane!",
    },
    {
        "role": "assistant",
        "content": {
            "device_type": "smartphone",
            "brand": "Samsung",
            "model": "S24 Ultra",
            "key_features": ["AI features", "powerful zoom camera"],
        },
    },
    {
        "role": "user",
        "content": "Could you also add a sentiment field to show how positive or negative the user feels about the product? Add that field moving forward",
    },
    {
        "role": "assistant",
        "content": {
            "device_type": "smartphone",
            "brand": "Samsung",
            "model": "S24 Ultra",
            "key_features": ["AI features", "powerful zoom camera"],
            "sentiment": "very positive",
        },
    },
    {
        "role": "user",
        "content": "Just checked out the iPad Mini 7 but I'm really disappointed. The jelly scroll issue hasn't been fixed and there's barely any meaningful upgrades. No need to upgrade.",
    },
    {
        "role": "assistant",
        "content": {
            "device_type": "tablet",
            "brand": "Apple",
            "model": "iPad Mini 7",
            "key_features": ["jelly scroll issue persists", "minimal upgrades"],
            "sentiment": "negative",
        },
    },
]
