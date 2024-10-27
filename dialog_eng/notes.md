Dialog Engineering vs Prompt Engineering using Claude by @Anthropic

Instead of wasting time crafting the perfect system prompt and few-shot examples, just _talk_ to the LLM.
We all know what a system prompt is what few-shot examples are, so let's get started:

<system.py>

Add some examples:

<few_shot.py only system>

The Anthropic API expects a system prompt and a list of messages. So these are our arguments:

<few_shot.py>

This is what we get:

<few_shot_res.py>

Now how would we do this in Dialog Engineering? Instead of "Input" and "Output", just think of the few-shot examples as "user" and "assistant" messages. Because that's what they actually are. An expected user message followed by an expected LLM response.

This is how our "dialog" looks:

<system.py>
<dialog1.py>


We are basically tricking the model into thinking that we are continuing a conversation that has been going well and that it should go on just like this. And we will keep adding the latest user assistant messages to the overall messages list. This is much more intuitive than the system prompt and few-shot examples.
Now what if we want to add an example with user feedback? So a user message, then an assistant message that is incorrect, then a user message providing feedback, and then an assistant message that is correct.

Here you go:

<system.py>
<dialog2.py>

By adding this example, the LLM will make sure to get the model right moving forward, because it thinks it already got it wrong once before but was able to correct itself with some help.

Now what if we want to add some more guidelines? Something that we forgot to add in the system prompt or the initial chat history? Something that maybe came up in our experiments? Or a new idea? Rewrite it all? No, just _talk_ to the LLM. Let's say we want to know the sentiment as well.

Here you go:

<system.py>
<dialog3.py>

This is what we get:

<system.py>
<dialog3_res.py>

Now what would happen going forward?
Let's just add a new user message and see:

<system.py>
<dialog4.py>

This is what we get:

<dialog4_res.py>

And now here are our arguments for this use case:

<system.py>
<dialog.py>

We will load this "dialog" whenever we need to extract structured details about gadgets. That could be in a node/action in a @burr app, or just a standalone script.