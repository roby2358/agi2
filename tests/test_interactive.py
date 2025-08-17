"""
Tests for InteractivePrompt class.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from src.interactive import InteractivePrompt


class TestInteractivePrompt:
    """Test cases for InteractivePrompt class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        
        # Mock tokenizer methods
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        self.mock_tokenizer.vocab = {'<EOS>': 0}
        
        self.prompt = InteractivePrompt(
            model=self.mock_model,
            max_context_length=100,
            tokenizer=self.mock_tokenizer
        )
    
    def test_initialization(self):
        """Test InteractivePrompt initialization."""
        assert self.prompt.model == self.mock_model
        assert self.prompt.max_context_length == 100
        assert self.prompt.tokenizer == self.mock_tokenizer
        assert self.prompt.conversation_history == []
        assert self.prompt.current_context_length == 0
    
    def test_initialization_without_tokenizer(self):
        """Test that initialization without tokenizer raises error."""
        with pytest.raises(ValueError, match="Tokenizer is required"):
            InteractivePrompt(model=self.mock_model, tokenizer=None)
    
    def test_send_message(self):
        """Test sending a message and getting a response."""
        # Mock the generate_text function
        with pytest.patch('src.interactive.generate_text') as mock_generate:
            mock_generate.return_value = "Hello! How can I help you?"
            
            response = self.prompt.send_message("Hi there!")
            
            # Check response
            assert response == "Hello! How can I help you?"
            
            # Check conversation history
            assert len(self.prompt.conversation_history) == 2
            
            user_msg = self.prompt.conversation_history[0]
            assert user_msg['role'] == 'user'
            assert user_msg['content'] == "Hi there!"
            assert user_msg['tokens'] == 5
            
            assistant_msg = self.prompt.conversation_history[1]
            assert assistant_msg['role'] == 'assistant'
            assert assistant_msg['content'] == "Hello! How can I help you?"
            assert assistant_msg['tokens'] == 5
            
            # Check context length
            assert self.prompt.current_context_length == 10
    
    def test_build_context(self):
        """Test building conversation context."""
        # Add some messages to history
        self.prompt.conversation_history = [
            {'role': 'user', 'content': 'Hello', 'tokens': 3},
            {'role': 'assistant', 'content': 'Hi!', 'tokens': 2},
            {'role': 'user', 'content': 'How are you?', 'tokens': 4}
        ]
        
        context = self.prompt._build_context()
        
        expected = "User: Hello\nAssistant: Hi!\nUser: How are you?"
        assert context == expected
    
    def test_build_context_empty(self):
        """Test building context with empty history."""
        context = self.prompt._build_context()
        assert context == ""
    
    def test_manage_context_length(self):
        """Test context length management."""
        # Set up conversation history that exceeds max length
        self.prompt.conversation_history = [
            {'role': 'user', 'content': 'Old message 1', 'tokens': 30},
            {'role': 'assistant', 'content': 'Old response 1', 'tokens': 25},
            {'role': 'user', 'content': 'Old message 2', 'tokens': 30},
            {'role': 'assistant', 'content': 'Old response 2', 'tokens': 25},
            {'role': 'user', 'content': 'Recent message', 'tokens': 20}
        ]
        self.prompt.current_context_length = 130  # Exceeds max of 100
        
        self.prompt._manage_context_length()
        
        # Should keep at least the last user-assistant pair
        assert len(self.prompt.conversation_history) >= 2
        assert self.prompt.current_context_length <= 100
        
        # Check that recent message is still there
        recent_msg = self.prompt.conversation_history[-1]
        assert recent_msg['content'] == 'Recent message'
    
    def test_clear_context(self):
        """Test clearing conversation context."""
        # Add some messages
        self.prompt.conversation_history = [
            {'role': 'user', 'content': 'Hello', 'tokens': 5},
            {'role': 'assistant', 'content': 'Hi!', 'tokens': 3}
        ]
        self.prompt.current_context_length = 8
        
        self.prompt.clear_context()
        
        assert self.prompt.conversation_history == []
        assert self.prompt.current_context_length == 0
    
    def test_get_conversation_summary(self):
        """Test getting conversation summary."""
        # Add some messages
        self.prompt.conversation_history = [
            {'role': 'user', 'content': 'Hello', 'tokens': 5},
            {'role': 'assistant', 'content': 'Hi!', 'tokens': 3},
            {'role': 'user', 'content': 'How are you?', 'tokens': 7}
        ]
        self.prompt.current_context_length = 15
        
        summary = self.prompt.get_conversation_summary()
        
        assert summary['total_messages'] == 3
        assert summary['user_messages'] == 2
        assert summary['assistant_messages'] == 1
        assert summary['current_context_length'] == 15
        assert summary['max_context_length'] == 100
        assert summary['context_utilization'] == 0.15
    
    def test_get_recent_messages(self):
        """Test getting recent messages."""
        # Add several messages
        for i in range(10):
            self.prompt.conversation_history.append({
                'role': 'user' if i % 2 == 0 else 'assistant',
                'content': f'Message {i}',
                'tokens': 5
            })
        
        # Get last 5 messages
        recent = self.prompt.get_recent_messages(5)
        assert len(recent) == 5
        
        # Check that we get the most recent messages
        assert recent[-1]['content'] == 'Message 9'
        assert recent[0]['content'] == 'Message 5'
    
    def test_export_conversation(self):
        """Test exporting conversation to string."""
        # Add some messages
        self.prompt.conversation_history = [
            {'role': 'user', 'content': 'Hello', 'tokens': 5},
            {'role': 'assistant', 'content': 'Hi!', 'tokens': 3}
        ]
        
        export = self.prompt.export_conversation()
        
        assert "Conversation History" in export
        assert "1. User: Hello" in export
        assert "2. Assistant: Hi!" in export
    
    def test_export_conversation_empty(self):
        """Test exporting empty conversation."""
        export = self.prompt.export_conversation()
        assert export == "No conversation history."
    
    def test_set_max_context_length(self):
        """Test updating maximum context length."""
        # Test valid length
        self.prompt.set_max_context_length(200)
        assert self.prompt.max_context_length == 200
        
        # Test invalid length
        with pytest.raises(ValueError, match="must be at least 100"):
            self.prompt.set_max_context_length(50)
    
    def test_get_context_info(self):
        """Test getting context information string."""
        # Add some messages
        self.prompt.conversation_history = [
            {'role': 'user', 'content': 'Hello', 'tokens': 50},
            {'role': 'assistant', 'content': 'Hi!', 'tokens': 30}
        ]
        self.prompt.current_context_length = 80
        
        info = self.prompt.get_context_info()
        
        assert "Context: 80/100 tokens" in info
        assert "80.0% utilization" in info
        assert "Medium" in info  # Should be medium utilization
    
    def test_context_utilization_levels(self):
        """Test different context utilization levels."""
        # Low utilization
        self.prompt.current_context_length = 30
        info = self.prompt.get_context_info()
        assert "Low" in info
        
        # High utilization
        self.prompt.current_context_length = 90
        info = self.prompt.get_context_info()
        assert "High" in info
