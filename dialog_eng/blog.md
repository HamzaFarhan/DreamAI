Dialog Engineering vs Prompt Engineering.

Instead of wasting time crafting the perfect system prompt and few-shot examples, just _talk_ to the LLM.
We all know what a system prompt is and what few-shot examples are, so let's get started:

```python
system = """
Extract structured details about gadgets in this format:
{
  "device_type": "",
  "brand": "",
  "model": "",
  "key_features": []
}
"""
```

Add some examples:

```python
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
```

The Anthropic API expects a system prompt and a list of messages. So along with the `system` defined above, this will be our `messages` argument:

```python
messages = [
    {
        "role": "user",
        "content": "Just got the new Samsung S24 Ultra, the AI features are mind-blowing and the zoom on the camera is insane!",
    }
]
```

This is what we get:

```python
response = {
    "role": "assistant",
    "content": {
        "device_type": "smartphone",
        "brand": "Samsung",
        "model": "S24 Ultra",
        "key_features": ["AI capabilities", "powerful zoom camera"],
    },
}
```

Now how would we do this in Dialog Engineering? Instead of "Input" and "Output", just think of the few-shot examples as "user" and "assistant" messages. Because that's what they actually are. An expected user message followed by an expected LLM response.
This is our `dialog`:

```python
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
```

We are basically tricking the LLM into thinking that we are continuing a conversation that has been going well and that it should continue just like this. We will keep adding the latest user assistant messages to the overall messages list.
This is much more intuitive than a system prompt with examples.
Now what if we want to add an example with user feedback? So a user message, then an assistant message that is incorrect, then a user message providing feedback, and then an assistant message that is correct.

```python
messages = [
    
    # previous messages
    
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
]
```

By adding this example, the LLM will make sure to get things right moving forward, because it thinks it already got it wrong once before but was able to correct itself with some help.
Now what if we want to add some more guidelines? Something we forgot to add in the system prompt or the initial chat history? Something that came up in our experiments? Or a new idea? Rewrite it all? No, just _talk_ to the LLM. Let's say we want to know the sentiment as well.

```python
messages = [
    
    # previous messages
    
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
]
```

This is what we get:

```python
response = {
    "role": "assistant",
    "content": {
        "device_type": "smartphone",
        "brand": "Samsung",
        "model": "S24 Ultra",
        "key_features": ["AI features", "powerful zoom camera"],
        "sentiment": "very positive",
    },
}
```

Now what would happen going forward?
Let's just add a new user message and see:

```python
messages = [
    
    # previous messages
    
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
]
```

This is what we get:

```python
response = {
    "role": "assistant",
    "content": {
        "device_type": "tablet",
        "brand": "Apple",
        "model": "iPad Mini 7",
        "key_features": ["jelly scroll issue persists", "minimal upgrades"],
        "sentiment": "negative",
    },
}
```

And now here are our arguments for this whole use case:

```python
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

```

We will load this `dialog` whenever we need to "Extract Structured Details About Gadgets". That could be in a node/action in an agentic application, or just a standalone script.