import { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from "uuid";
import {
    useChatInteract,
    useChatMessages,
} from "@chainlit/react-client";

function ChatbotComponent() {
    const [inputValue, setInputValue] = useState([]);
    const { sendMessage } = useChatInteract();
    const { messages } = useChatMessages();
    const conversationRef = useRef(null);

    const handleSendMessage = () => {
        if (!inputValue.trim()) return;
        // Simulate a bot response
        if (inputValue) {
            const message = {
                id: uuidv4(),
                name: "user",
                type: "user_message",
                output: inputValue,
                createdAt: new Date().toISOString(),
            };
            sendMessage(message, []);
            setInputValue('');
        }
    };

    const handleInputChange = (event) => {
        setInputValue(event.target.value);
    };

    const handleKeyDown = (event) => {
        if (event.key === 'Enter') {
            handleSendMessage();
        }
    };

    useEffect(() => {
        conversationRef.current.scrollTop = conversationRef.current.scrollHeight;
    }, [messages]);

    return (
        <div className="chatbot-component">
            <div className="conversation" ref={conversationRef}>
                {messages.map((message, index) => (
                    <div key={index} className={`message ${message.name}`}>
                        {message.output}
                    </div>
                ))}
            </div>
            <div className="message-input">
                <input
                    type="text"
                    value={inputValue}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    placeholder="Type here..."
                />
                <button onClick={handleSendMessage}>Send</button>
            </div>
            <div style={{ clear: 'both' }}></div>
        </div>
    );
}

export default ChatbotComponent;
