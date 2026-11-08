import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import AppHeader from "./components/AppHeader.jsx";
import Bills from "./pages/Bills.jsx";
import Chat from "./pages/Chat.jsx";

// Both pages stay mounted and the inactive one is only hidden. Unmounting on a tab
// switch threw away a running bill analysis (and the chat transcript) mid-job.
const PAGES = [
  { path: "/chat", Page: Chat },
  { path: "/bills", Page: Bills },
];

function Pages() {
  const { pathname } = useLocation();
  return (
    <>
      {PAGES.map(({ path, Page }) => (
        <div key={path} className={pathname === path ? "h-full" : "hidden"}>
          <Page />
        </div>
      ))}
      <Routes>
        {PAGES.map(({ path }) => <Route key={path} path={path} element={null} />)}
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-full flex-col">
        <AppHeader />
        <main className="min-h-0 flex-1">
          <Pages />
        </main>
      </div>
    </BrowserRouter>
  );
}
