"use client";

import type React from "react";

import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/navbar";
import Footer from "@/components/footer";
import { usePathname } from "next/navigation";
import DarkVeil from "@/components/DarkVeil";

const inter = Inter({ subsets: ["latin"] });

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const isDashboard = pathname === "/dashboard";

  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        <link rel="icon" href="/icon.svg" type="image/svg+xml" />
        <meta name="description" content="Advanced Crowd Detection and Surveillance Dashboard" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body>
        {/* Background */}
        {!isDashboard && (
          <div className="fixed inset-0 z-0">
            <DarkVeil />
          </div>
        )}

        {/* Navbar */}
        {!isDashboard && (
          <div className="fixed top-0 left-0 right-0 z-50">
            <Navbar />
          </div>
        )}

        <main className={`${isDashboard ? "" : "relative z-10"}`}>
          {children}
        </main>

        {/* Footer */}
        {!isDashboard && (
          <div className="relative z-20">
            <Footer />
          </div>
        )}
      </body>
    </html>
  );
}
