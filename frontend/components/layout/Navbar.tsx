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
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-purple-600 flex items-center justify-center shadow-lg">
            {/* Professional Crowd Detection Icon */}
            <svg
              width="20"
              height="20"
              viewBox="0 0 32 32"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              {/* Background circle */}
              <circle cx="16" cy="16" r="15" fill="url(#gradient)" stroke="url(#strokeGradient)" strokeWidth="2"/>
              
              {/* Camera lens */}
              <circle cx="16" cy="12" r="4" fill="none" stroke="white" strokeWidth="1.5"/>
              <circle cx="16" cy="12" r="2.5" fill="none" stroke="white" strokeWidth="1"/>
              
              {/* Camera body */}
              <rect x="12" y="16" width="8" height="6" rx="1" fill="white" opacity="0.9"/>
              
              {/* People silhouettes (crowd) */}
              <g opacity="0.8">
                {/* Person 1 */}
                <circle cx="10" cy="20" r="1" fill="white"/>
                <rect x="9.5" y="21" width="1" height="2" fill="white"/>
                <rect x="9" y="23" width="2" height="0.5" fill="white"/>
                
                {/* Person 2 */}
                <circle cx="13" cy="19" r="0.8" fill="white"/>
                <rect x="12.7" y="19.8" width="0.6" height="1.5" fill="white"/>
                <rect x="12.5" y="21.3" width="1" height="0.4" fill="white"/>
                
                {/* Person 3 */}
                <circle cx="19" cy="20" r="0.8" fill="white"/>
                <rect x="18.7" y="20.8" width="0.6" height="1.5" fill="white"/>
                <rect x="18.5" y="22.3" width="1" height="0.4" fill="white"/>
                
                {/* Person 4 */}
                <circle cx="22" cy="19" r="0.7" fill="white"/>
                <rect x="21.8" y="19.7" width="0.4" height="1.3" fill="white"/>
                <rect x="21.6" y="21" width="0.8" height="0.3" fill="white"/>
              </g>
              
              {/* Safety shield overlay */}
              <path d="M16 6 L20 8 L20 12 C20 16 16 20 16 20 C16 20 12 16 12 12 L12 8 Z" fill="none" stroke="white" strokeWidth="1" opacity="0.6"/>
              
              {/* Gradients */}
              <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#1e3a8a" stopOpacity="1" />
                  <stop offset="50%" stopColor="#3730a3" stopOpacity="1" />
                  <stop offset="100%" stopColor="#1e1b4b" stopOpacity="1" />
                </linearGradient>
                <linearGradient id="strokeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#60a5fa" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#a855f7" stopOpacity="0.8" />
                </linearGradient>
              </defs>
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
