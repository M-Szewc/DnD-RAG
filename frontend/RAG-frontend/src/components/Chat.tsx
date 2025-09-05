import { useState } from "react";
import ChatbotIcon from "./ChatbotIcon";
import ChatForm from "./ChatForm";
import ChatMessage from "./ChatMessage";


type ChatProps = {
    children: React.ReactNode;
}

type ChatLog = {
    role: string;
    text: string;
}

export type {ChatLog};

export default function ChatBox({children} : ChatProps) {
    const [chatHistory, setChatHistory] = useState<Array<ChatLog>>([]);
    
    const generateBotResponse = async (chatHistory: Array<ChatLog>) => {
        const updateHistory = (text: string) => {
            setChatHistory(history => [...history.filter(msg => msg.text !== "Thinking..."), { role: "model", text: text }]);
        };
        
        const requestOptions: RequestInit = {
            method: "POST",
            mode: "cors",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({input: chatHistory[chatHistory.length-1].text})
        };

        try {
            const response = await fetch("http://localhost:5678/generator/invoke", requestOptions);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error.message || "Something went wrong!");

            const botResponseText = data.output.replace(/\*\*(.*?)\*\*/g, "$1").trim();
            updateHistory(botResponseText);
        } catch (error) {
            console.log(error);
        }
    };

    return (
        <div className="container">
            <div className="chatbot-popup">
                <div className="chat-header">
                    <div className="header-info">
                        <ChatbotIcon/>
                        <h2 className="logo-text">{children}</h2>
                    </div>
                    <button className="material-symbols-rounded">
                        keyboard_arrow_down
                    </button>
                </div>
                {/* Chatbot body */}
                <div className="chat-body">
                    <div className="message bot-message">
                        <ChatbotIcon/>
                        <p className="message-text">
                            Hi c:
                        </p>
                    </div>
                    {chatHistory.map((chat, index)=> (
                        <ChatMessage key={index} chat={chat}/>
                    ))}
                    
                </div>

                {/* Chatbot Footer */}
                <div className="chat-footer">
                    <ChatForm chatHistory={chatHistory} setChatHistory={setChatHistory} generateBotResponse={generateBotResponse}/>
                </div>
            </div>
        </div>
    );
}