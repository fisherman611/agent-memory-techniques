---
title: Agent Memory Techniques
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
hf_oauth: true
python_version: 3.11
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

# **AI Agent Chatbot with Multiple Memory Techniques**

An advanced AI chatbot project that implements and compares different conversation-memory strategies. The goal is to maintain long-term context while optimizing cost, speed, and response quality.

## **Overview**

Long conversations create several challenges for chatbots:

* token limits get exceeded,
* API costs increase,
* responses slow down.

However, preserving context is essential for coherent and relevant replies.

This project experiments with multiple memory techniques — from simple buffer storage to event-triggered summaries and hybrid selective memory — to explore how each method affects:

* context retention,
* computational efficiency,
* cost,
* overall quality of the conversation.

The project shows how different memory architectures behave in real scenarios such as short support chats, long problem-solving sessions, or assistants that need richer context.


## **Memory Techniques**

### **1. Buffer Memory**

The simplest approach that stores the complete conversation history.

**Advantages:**
- Preserves all context and details
- No information loss
- Simple to implement

**Disadvantages:**
- High token consumption
- May exceed context window limits in long conversations
- Slower processing with large histories

### **2. Sliding Window**

Maintains only the K most recent messages from the conversation history.

**Advantages:**
- Fixed memory size
- Keeps most recent and relevant context
- Predictable token usage

**Disadvantages:**
- Loses older context that might be relevant
- May lose important information from earlier in the conversation

### **3. Recursive Summarization**

Continuously summarizes the conversation history using a dynamic approach:

```
(current_summary, new_question) → LLM → updated_summary
```

The system maintains a rolling summary that gets updated with each new interaction.

**Advantages:**
- Compact representation of conversation history
- Captures key information from the entire conversation
- Scalable to very long conversations

**Disadvantages:**
- Some details may be lost in summarization
- Requires additional LLM calls for summarization
- Quality depends on summarization prompts

### **4. Recursive Summarization + Sliding Window**

Combines both approaches for optimal balance:
- Maintains a summary of older conversation history
- Keeps the K most recent messages in full detail

**Advantages:**
- Balances detail and efficiency
- Recent context preserved in full
- Historical context available through summary
- More robust than either technique alone

**Disadvantages:**
- More complex to implement
- Requires tuning the window size parameter

### **5. Recursive Summarization + Key Messages**

Maintains a summary alongside explicitly marked important messages:
- Automatically or manually identifies key messages
- Preserves critical information that shouldn't be summarized
- Summarizes less important conversational content

**Advantages:**
- Ensures important context is never lost
- More intelligent than simple windowing
- Good balance of efficiency and completeness

**Disadvantages:**
- Requires logic to identify key messages
- Slightly more complex implementation
- May need manual message flagging for best results

## **Use Cases**

- **Buffer Memory**: Short conversations, debugging, or when complete history is required
- **Sliding Window**: Chatbots with natural conversation flow where recent context matters most
- **Recursive Summarization**: Long-running conversations, customer support sessions
- **Recursive + Sliding Window**: General-purpose chatbots requiring both efficiency and context
- **Recursive + Key Messages**: Task-oriented conversations where specific details must be preserved

## **Installation**

**1. Clone the repository:**
```bash
git clone https://github.com/fisherman611/agent-memory-techniques.git
cd agent-memory-techniques
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Configure environment variables:**
```bash
# Create .env file with your API keys
GOOGLE_API_KEY=your_google_api_key
```

**4. Launch the application:**
```bash
python demo/app.py
```

## **License** 
This project is licensed under the [MIT License](LICENSE).