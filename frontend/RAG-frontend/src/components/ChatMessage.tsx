import ChatbotIcon from "./ChatbotIcon";


type ChatMessagePrompts = {
    chat: {
        role: string;
        text: string;
    };
}

export default function ChatMessage({chat} : ChatMessagePrompts) {
    return (
        <div className={`message ${chat.role === "model" ? 'bot' : 'user'}-message`}>
            {chat.role === "model" && <ChatbotIcon/>}
            <p className="message-text">{chat.text}</p>
        </div>
    );
}