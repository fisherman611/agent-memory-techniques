import os
import sys
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

from techniques.sliding_window.message_history import BufferWindowMessageHistory
from techniques.recursive_summarization.message_history import ConversationSummaryMessageHistory
from techniques.recursive_summarization_sliding_window.message_history import ConversationSummaryBufferMessageHistory
from callbacks.manager import get_gemini_callback

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Global state for managing chat sessions
chat_stores = {}

def get_session_history(session_id: str, memory_type: str, window_size: int, llm):
    """Get or create chat history for a session."""
    key = f"{session_id}_{memory_type}_{window_size}"
    
    if key not in chat_stores:
        if memory_type == "In-Memory (No Limit)":
            chat_stores[key] = InMemoryChatMessageHistory()
        elif memory_type == "Sliding Window":
            chat_stores[key] = BufferWindowMessageHistory(k=window_size)
        elif memory_type == "Recursive Summarization":
            chat_stores[key] = ConversationSummaryMessageHistory(llm=llm)
        elif memory_type == "Summary + Sliding Window":
            chat_stores[key] = ConversationSummaryBufferMessageHistory(llm=llm, k=window_size)
    
    return chat_stores[key]

def create_chain(memory_type: str, window_size: int, temperature: float):
    """Create a LangChain runnable with the specified memory configuration."""
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
        google_api_key=GEMINI_API_KEY
    )
    
    # Create conversational chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Be concise, friendly, and informative in your responses. 
You can help answer questions, have conversations, and assist with various tasks.
When asked about the current time, provide it based on your knowledge cutoff.
You can also help with basic calculations if asked."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: get_session_history(session_id, memory_type, window_size, llm),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return chain_with_history, llm

def format_message_history(messages):
    """Format message history for display as a beautiful chat conversation."""
    if not messages:
        return """
<div style="padding: 10px; text-align: center; color: #666;">
    <i>No messages in history yet</i>
</div>
"""
    
    formatted = []
    for i, msg in enumerate(messages):
        # Determine role and styling
        class_name = msg.__class__.__name__
        
        if 'Human' in class_name:
            role = "USER"
            role_color = "#2563eb"  # Blue
            bg_color = "#eff6ff"
            border_color = "#bfdbfe"
            icon = "👤"
        elif 'AI' in class_name or 'Assistant' in class_name:
            role = "AI"
            role_color = "#059669"  # Green
            bg_color = "#f0fdf4"
            border_color = "#bbf7d0"
            icon = "🤖"
        elif 'System' in class_name:
            role = "SYSTEM"
            role_color = "#7c3aed"  # Purple
            bg_color = "#faf5ff"
            border_color = "#e9d5ff"
            icon = "⚙️"
        else:
            role = class_name.upper()
            role_color = "#6b7280"  # Gray
            bg_color = "#f9fafb"
            border_color = "#e5e7eb"
            icon = "💬"
        
        # Truncate long messages for display
        content = msg.content
        is_truncated = len(content) > 300
        display_content = content[:300] + "..." if is_truncated else content
        
        # Escape HTML special characters
        display_content = (display_content
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>"))
        
        # Create message bubble
        message_html = f"""
<div style="margin-bottom: 10px; padding: 10px; background-color: {bg_color}; border-left: 3px solid {border_color}; border-radius: 6px;">
    <div style="display: flex; align-items: center; margin-bottom: 4px;">
        <span style="font-size: 16px; margin-right: 6px;">{icon}</span>
        <strong style="color: {role_color}; font-size: 13px;">[{role}]</strong>
        <span style="margin-left: auto; color: #9ca3af; font-size: 10px;">#{i+1}</span>
    </div>
    <div style="color: #374151; font-size: 12px; line-height: 1.4; padding-left: 22px;">
        {display_content}
    </div>
</div>
"""
        formatted.append(message_html)
    
    return "\n".join(formatted)

def chat(message, history, memory_type, window_size, temperature, session_id):
    """Process a chat message and return the response."""
    
    if not message.strip():
        return history, "", "", ""
    
    # Create chain with current settings
    chain, llm = create_chain(memory_type, window_size, temperature)
    
    # Use callback to track token usage
    with get_gemini_callback() as cb:
        try:
            # Invoke the chain
            response = chain.invoke(
                {"input": message},
                config={"configurable": {"session_id": session_id}, "callbacks": [cb]}
            )
            
            # Extract response text
            if isinstance(response, dict):
                response_text = response.get("output", str(response))
            else:
                response_text = str(response)
            
            # Update history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response_text})
            
            # Get statistics
            usage = cb.get_total_usage()
            stats = f"""
<div style="display: flex; gap: 20px; padding: 10px; background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 8px; border-left: 4px solid #0284c7;">
    <div style="flex: 1; text-align: center;">
        <div style="font-size: 24px; font-weight: bold; color: #0369a1;">{usage['total_prompt_tokens']:,}</div>
        <div style="font-size: 11px; color: #64748b;">📥 Prompt Tokens</div>
    </div>
    <div style="flex: 1; text-align: center;">
        <div style="font-size: 24px; font-weight: bold; color: #0369a1;">{usage['total_completion_tokens']:,}</div>
        <div style="font-size: 11px; color: #64748b;">📤 Completion Tokens</div>
    </div>
    <div style="flex: 1; text-align: center;">
        <div style="font-size: 24px; font-weight: bold; color: #0c4a6e;">{usage['total_tokens_used']:,}</div>
        <div style="font-size: 11px; color: #64748b;">🔢 Total Tokens</div>
    </div>
</div>
"""
            
            # Get current memory state
            key = f"{session_id}_{memory_type}_{window_size}"
            if key in chat_stores:
                current_history = chat_stores[key].messages
                
                # Create header info
                header_info = f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px; border-radius: 8px; margin-bottom: 12px; color: white;">
    <div style="font-size: 14px; font-weight: bold; margin-bottom: 6px;">📊 Overview</div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 12px;">
        <div><strong>Type:</strong> {memory_type.split()[0]}</div>
        <div><strong>Messages:</strong> {len(current_history)}</div>
        <div><strong>Window:</strong> {window_size if 'Window' in memory_type else 'N/A'}</div>
        <div><strong>ID:</strong> {session_id}</div>
    </div>
</div>
"""
                
                # Create conversation history display
                history_html = f"""
<div style="height: 480px; overflow-y: auto; padding: 8px; background-color: #ffffff; border-radius: 8px; border: 1px solid #e5e7eb;">
    {format_message_history(current_history)}
</div>
"""
                
                memory_info = header_info + history_html
            else:
                memory_info = """
<div style="padding: 20px; text-align: center; color: #9ca3af; height: 550px; display: flex; align-items: center; justify-content: center; flex-direction: column; background-color: #f9fafb; border-radius: 8px;">
    <div style="font-size: 48px; margin-bottom: 8px;">📭</div>
    <div>No memory state available</div>
</div>
"""
            
        except Exception as e:
            response_text = f"Error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response_text})
            stats = """
<div style="display: flex; gap: 20px; padding: 10px; background: linear-gradient(90deg, #fef2f2 0%, #fee2e2 100%); border-radius: 8px; border-left: 4px solid #dc2626;">
    <div style="flex: 1; text-align: center;">
        <div style="font-size: 24px; font-weight: bold; color: #dc2626;">❌</div>
        <div style="font-size: 11px; color: #64748b;">Error occurred during processing</div>
    </div>
</div>
"""
            memory_info = """
<div style="padding: 20px; text-align: center; color: #dc2626; height: 550px; display: flex; align-items: center; justify-content: center; flex-direction: column; background-color: #fef2f2; border-radius: 8px;">
    <div style="font-size: 48px; margin-bottom: 8px;">❌</div>
    <div>Error occurred</div>
</div>
"""
    
    return history, stats, memory_info, ""

def clear_chat(session_id, memory_type, window_size):
    """Clear the chat history."""
    key = f"{session_id}_{memory_type}_{window_size}"
    if key in chat_stores:
        chat_stores[key].clear()
    
    empty_memory_state = """
<div style="padding: 20px; text-align: center; color: #9ca3af; height: 550px; display: flex; align-items: center; justify-content: center; flex-direction: column; background-color: #f9fafb; border-radius: 8px;">
    <div style="font-size: 48px; margin-bottom: 8px;">📭</div>
    <div>No memory state yet</div>
    <div style="font-size: 12px; margin-top: 8px; color: #9ca3af;">Start chatting to see memory updates</div>
</div>
"""
    
    empty_stats = """
<div style="display: flex; gap: 20px; padding: 10px; background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 8px; border-left: 4px solid #0284c7;">
    <div style="flex: 1; text-align: center;">
        <div style="font-size: 24px; font-weight: bold; color: #94a3b8;">-</div>
        <div style="font-size: 11px; color: #64748b;">📥 Prompt Tokens</div>
    </div>
    <div style="flex: 1; text-align: center;">
        <div style="font-size: 24px; font-weight: bold; color: #94a3b8;">-</div>
        <div style="font-size: 11px; color: #64748b;">📤 Completion Tokens</div>
    </div>
    <div style="flex: 1; text-align: center;">
        <div style="font-size: 24px; font-weight: bold; color: #94a3b8;">-</div>
        <div style="font-size: 11px; color: #64748b;">🔢 Total Tokens</div>
    </div>
</div>
<div style="text-align: center; margin-top: 8px; font-size: 12px; color: #64748b;">
    <em>Start chatting to see token usage statistics</em>
</div>
"""
    
    return [], empty_stats, empty_memory_state, ""

def reset_session():
    """Generate a new session ID."""
    import uuid
    return str(uuid.uuid4())[:8]

# Create Gradio interface
custom_css="""
    .stats-box { 
        margin-top: 10px; 
    }
    @media (max-width: 1024px) {
        .gr-row { 
            flex-direction: column !important; 
        }
    }
    .gr-button {
        transition: all 0.3s ease;
    }
    .gr-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .full-height-btn {
    height: 100% !important;
    min-height: 80px !important;
}
"""

with gr.Blocks(title="AI Chatbot with Multiple Memory Techniques") as demo:
    
    gr.Markdown("""
    # 🤖 AI Chatbot with Multiple Memory Techniques
    
    This demo showcases different memory management techniques for conversational AI.
    Watch how different techniques store and manage conversation history in real-time!
    """)
    
    # Settings Section at the top
    with gr.Group():
        gr.Markdown("### ⚙️ Settings & Controls")
        
        with gr.Row():
            # Memory Technique - now as dropdown for better responsiveness
            memory_type = gr.Dropdown(
                choices=[
                    "In-Memory (No Limit)",
                    "Sliding Window",
                    "Recursive Summarization",
                    "Summary + Sliding Window"
                ],
                value="Sliding Window",
                label="💾 Memory Technique",
                info="Choose how conversation history is managed",
                scale=2,
            )
            
            # Window Size
            window_size = gr.Slider(
                minimum=2,
                maximum=20,
                value=6,
                step=1,
                label="📏 Window Size",
                info="Number of messages to keep",
                scale=1,
            )
            
            # Temperature
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="🌡️ Temperature",
                info="0=focused, 1=creative",
                scale=1,
            )
        
        # Stats box with better formatting
        with gr.Row():
            stats_box = gr.HTML(
                value="""
<div style="display: flex; gap: 20px; padding: 10px; background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 8px; border-left: 4px solid #0284c7;">
    <div style="flex: 1; text-align: center;">
        <div style="font-size: 24px; font-weight: bold; color: #94a3b8;">-</div>
        <div style="font-size: 11px; color: #64748b;">📥 Prompt Tokens</div>
    </div>
    <div style="flex: 1; text-align: center;">
        <div style="font-size: 24px; font-weight: bold; color: #94a3b8;">-</div>
        <div style="font-size: 11px; color: #64748b;">📤 Completion Tokens</div>
    </div>
    <div style="flex: 1; text-align: center;">
        <div style="font-size: 24px; font-weight: bold; color: #94a3b8;">-</div>
        <div style="font-size: 11px; color: #64748b;">🔢 Total Tokens</div>
    </div>
</div>
<div style="text-align: center; margin-top: 8px; font-size: 12px; color: #64748b;">
    <em>Start chatting to see token usage statistics</em>
</div>
""",
                elem_classes="stats-box"
            )
    
    session_id = gr.State(value=reset_session())
    
    gr.Markdown("---")
    
    # Main content: Chat and Memory side by side
    with gr.Row(equal_height=True):
        # Left side: Chat conversation
        with gr.Column(scale=1):
            gr.Markdown("### 💬 Chat Conversation")
            chatbot = gr.Chatbot(
                label="",
                height=550,
                avatar_images=(None, None),
                show_label=False,
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="",
                    placeholder="Type your message here...",
                    scale=4,
                    lines=2,
                    show_label=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=0.5, size="lg", elem_classes="full-height-btn")
        
        # Right side: Memory State
        with gr.Column(scale=1):
            gr.Markdown("### 💾 Memory State")
            memory_box = gr.HTML(
                value="""
<div style="padding: 20px; text-align: center; color: #9ca3af; height: 550px; display: flex; align-items: center; justify-content: center; flex-direction: column; background-color: #f9fafb; border-radius: 8px;">
    <div style="font-size: 48px; margin-bottom: 8px;">📭</div>
    <div>No memory state yet</div>
    <div style="font-size: 12px; margin-top: 8px; color: #9ca3af;">Start chatting to see memory updates</div>
</div>
""",
                show_label=False,
            )
    
    # Control buttons at the bottom
    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="lg", scale=1)
        new_session_btn = gr.Button("🔄 New Session", variant="secondary", size="lg", scale=1)
    
    gr.Markdown("""
    ---
    
    <details>
    <summary><strong>📚 Memory Techniques Explained</strong></summary>
    
    - **In-Memory (No Limit)**: Stores all conversation history. Best for short conversations, can get expensive for long ones.
    - **Sliding Window**: Keeps only the last N messages. Efficient but may lose context from earlier in the conversation.
    - **Recursive Summarization**: Continuously summarizes all previous messages. Maintains context but summaries may lose details.
    - **Summary + Sliding Window**: Keeps recent messages AND a summary of older ones. Best balance of context and efficiency.
    
    </details>
    
    <details>
    <summary><strong>💡 Example Prompts</strong></summary>
    
    - "Tell me a story about a robot"
    - "What is machine learning?"
    - "Explain quantum computing in simple terms"
    - "Let's have a conversation about space exploration" (tests memory retention)
    - "What did we just talk about?" (tests memory recall)
    
    </details>
    
    <details>
    <summary><strong>📝 Tips</strong></summary>
    
    - **Watch the side-by-side view**: See how your conversation (left) is stored in memory (right)
    - **Try different techniques**: Notice how each handles the same conversation differently
    - **Adjust window size**: See how it affects what's kept in memory
    - **Temperature**: Higher (0.8-1.0) = creative, Lower (0.1-0.3) = focused
    
    </details>
    """)
    
    # Event handlers
    def submit_message(message, history, memory_type, window_size, temperature, session_id):
        return chat(message, history, memory_type, window_size, temperature, session_id)
    
    send_btn.click(
        submit_message,
        inputs=[msg_input, chatbot, memory_type, window_size, temperature, session_id],
        outputs=[chatbot, stats_box, memory_box, msg_input],
    )
    
    msg_input.submit(
        submit_message,
        inputs=[msg_input, chatbot, memory_type, window_size, temperature, session_id],
        outputs=[chatbot, stats_box, memory_box, msg_input],
    )
    
    clear_btn.click(
        clear_chat,
        inputs=[session_id, memory_type, window_size],
        outputs=[chatbot, stats_box, memory_box, msg_input],
    )
    
    new_session_btn.click(
        reset_session,
        outputs=[session_id],
    ).then(
        clear_chat,
        inputs=[session_id, memory_type, window_size],
        outputs=[chatbot, stats_box, memory_box, msg_input],
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Default(), css=custom_css)
