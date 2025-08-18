"""
Interactive Prompt System

This module provides the InteractivePrompt class for interactive conversation management.
"""

import torch
from typing import List, Optional, Dict, Any
from .generation import generate_text


class InteractivePrompt:
    """
    Interactive prompt system that maintains conversation context.
    
    Args:
        model: The AGI2 model to use for generation
        max_context_length: Maximum length of conversation context
        tokenizer: Tokenizer for text processing
    """
    
    def __init__(
        self, 
        model, 
        max_context_length: int = 1024,
        tokenizer = None
    ):
        self.model = model
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        
        # Conversation context
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_context_length = 0
        
        if tokenizer is None:
            raise ValueError("Tokenizer is required for InteractivePrompt")
    
    def send_message(self, text: str) -> str:
        """
        Send a message and get a response, maintaining conversation context.
        
        Args:
            text: User's message text
            
        Returns:
            Model-generated response text
        """
        # Add user message to conversation history
        user_message = {
            'role': 'user',
            'content': text,
            'tokens': len(self.tokenizer.encode(text))
        }
        self.conversation_history.append(user_message)
        self.current_context_length += user_message['tokens']
        
        # Build context for the model
        context = self._build_context()
        
        # Generate response
        response = generate_text(
            model=self.model,
            prompt=context,
            max_length=100,
            temperature=0.8,
            tokenizer=self.tokenizer
        )
        
        # Add model response to conversation history
        model_message = {
            'role': 'assistant',
            'content': response,
            'tokens': len(self.tokenizer.encode(response))
        }
        self.conversation_history.append(model_message)
        self.current_context_length += model_message['tokens']
        
        # Manage context length
        self._manage_context_length()
        
        return response
    
    def _build_context(self) -> str:
        """
        Build conversation context from history.
        
        Returns:
            Formatted conversation context string
        """
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for message in self.conversation_history:
            role = message['role']
            content = message['content']
            
            if role == 'user':
                context_parts.append(f"User: {content}")
            elif role == 'assistant':
                context_parts.append(f"Assistant: {content}")
        
        return "\n".join(context_parts)
    
    def _manage_context_length(self) -> None:
        """Manage context length by truncating old messages if needed."""
        while self.current_context_length > self.max_context_length and len(self.conversation_history) > 2:
            # Remove oldest message (keep at least the last user-assistant pair)
            removed_message = self.conversation_history.pop(0)
            self.current_context_length -= removed_message['tokens']
    
    def clear_context(self) -> None:
        """Clear the conversation history and reset context."""
        self.conversation_history.clear()
        self.current_context_length = 0
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        user_messages = [msg for msg in self.conversation_history if msg['role'] == 'user']
        assistant_messages = [msg for msg in self.conversation_history if msg['role'] == 'assistant']
        
        return {
            'total_messages': len(self.conversation_history),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'current_context_length': self.current_context_length,
            'max_context_length': self.max_context_length,
            'context_utilization': self.current_context_length / self.max_context_length
        }
    
    def get_recent_messages(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent messages from the conversation.
        
        Args:
            n: Number of recent messages to return
            
        Returns:
            List of recent message dictionaries
        """
        return self.conversation_history[-n:] if self.conversation_history else []
    
    def export_conversation(self) -> str:
        """
        Export the conversation to a formatted string.
        
        Returns:
            Formatted conversation string
        """
        if not self.conversation_history:
            return "No conversation history."
        
        export_lines = ["Conversation History", "=" * 20, ""]
        
        for i, message in enumerate(self.conversation_history, 1):
            role = message['role'].title()
            content = message['content']
            export_lines.append(f"{i}. {role}: {content}")
            export_lines.append("")
        
        return "\n".join(export_lines)
    
    def set_max_context_length(self, new_length: int) -> None:
        """
        Update the maximum context length.
        
        Args:
            new_length: New maximum context length
        """
        if new_length < 100:
            raise ValueError("Context length must be at least 100 tokens")
        
        self.max_context_length = new_length
        self._manage_context_length()
    
    def get_context_info(self) -> str:
        """
        Get a human-readable summary of context usage.
        
        Returns:
            String describing current context usage
        """
        summary = self.get_conversation_summary()
        utilization = summary['context_utilization']
        
        if utilization < 0.5:
            status = "Low"
        elif utilization < 0.8:
            status = "Medium"
        else:
            status = "High"
        
        return (
            f"Context: {summary['current_context_length']}/{summary['max_context_length']} tokens "
            f"({utilization:.1%} utilization) - {status}"
        )
