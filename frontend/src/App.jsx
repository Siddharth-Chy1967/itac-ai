import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AppHeader from "./components/AppHeader.jsx";
import Chat from "./pages/Chat.jsx";
import KnowledgeBase from "./pages/KnowledgeBase.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-full flex-col">
        <AppHeader />
        <main className="min-h-0 flex-1">
          <Routes>
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/knowledge" element={<KnowledgeBase />} />
            <Route path="*" element={<Navigate to="/chat" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
