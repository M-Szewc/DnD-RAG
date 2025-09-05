import {useRef, type Dispatch, type SetStateAction} from "react";
import type { ChatLog } from "./Chat";

type ChatFormProps = {
    chatHistory: ChatLog[];
    setChatHistory: Dispatch<SetStateAction<ChatLog[]>>;
    generateBotResponse: (message: ChatLog[]) => void;
}

export default function ChatForm({chatHistory, setChatHistory, generateBotResponse} : ChatFormProps) {
    const inputRef = useRef<HTMLInputElement>(null);
    
    const handleFormSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if(!inputRef.current) return;
        const userMessage = inputRef.current.value.trim();
    
        if (!userMessage) return;
        inputRef.current.value = "";

        setChatHistory((history: ChatLog[]) => [...history, { role: "user", text: userMessage }]);
    
        setTimeout(() => {
                // Add a "thinking..." placeholder and generate response
                setChatHistory((history: ChatLog[]) => [...history, { role: "model", text: "Thinking..." }]);
                
                // Call the function to generate bot's response
                generateBotResponse([...chatHistory, { role: "user", text: userMessage }]);
            },
            600
        );

    }

    return (
        <form action="#" className="chat-form" onSubmit={handleFormSubmit}>
            <input ref={inputRef} type="text" placeholder="Message..."
            className="message-input" required />
            <button className="material-symbols-rounded">arrow_upward</button>
        </form>
    );
}