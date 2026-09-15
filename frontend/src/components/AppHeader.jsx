import { NavLink } from "react-router-dom";

// Nav appears on every page; the ITAC logo is a required header element sitewide.
const NAV = [
  { to: "/chat", label: "Chat" },
  { to: "/knowledge", label: "Knowledge Base" },
];

export default function AppHeader() {
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3">
        <img src="/itac-logo.png" alt="ITAC logo" className="h-9 w-auto" />
        <span className="text-lg font-semibold text-itac-blue">ITAC AI Assistant</span>
        <nav className="ml-auto flex flex-wrap gap-1">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-itac-blue text-white"
                    : "text-slate-600 hover:bg-itac-blue-light hover:text-itac-blue"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
