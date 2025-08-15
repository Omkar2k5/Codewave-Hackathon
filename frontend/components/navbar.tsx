"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "Home" },
  { href: "/working", label: "Working" },
  { href: "/dashboard", label: "Dashboard" },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="w-full flex justify-center mt-6 z-50">
      <div
        className="flex items-center justify-between px-8 py-3 rounded-full bg-white/10 backdrop-blur-md border border-white/20 shadow-lg max-w-2xl w-full"
        style={{ maxWidth: "700px" }}
      >
        {/* Logo and Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-500 to-blue-400 flex items-center justify-center">
            {/* Example SVG icon, replace as needed */}
            <svg
              width="22"
              height="22"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path d="M12 2L15 8H9L12 2Z" fill="#fff" />
              <circle cx="12" cy="15" r="5" fill="#fff" fillOpacity="0.7" />
            </svg>
          </div>
          <span className="text-lg font-semibold text-white tracking-wide">
            Drishti
          </span>
        </div>
        {/* Nav Links */}
        <div className="flex items-center space-x-8">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`text-white font-medium transition-all duration-300 ${
                  isActive
                    ? "underline decoration-2 underline-offset-4 decoration-blue-400"
                    : "hover:underline hover:decoration-2 hover:underline-offset-4 hover:decoration-white/70"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
