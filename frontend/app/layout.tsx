"use client"

import type React from "react"

import { Inter } from "next/font/google"
import "./globals.css"
import Navbar from "@/components/navbar"
import Footer from "@/components/footer"
import { usePathname } from "next/navigation"
import DarkVeil from "@/components/DarkVeil"

const inter = Inter({ subsets: ["latin"] })

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const pathname = usePathname()
  const isDashboard = pathname === "/dashboard"

  return (
    <html lang="en">
      <body>
        {/* Background */}
        <div style={{ position: "absolute", inset: 0, zIndex: 0 }}>
          <DarkVeil />
        </div>
        {!isDashboard && <Navbar />}
        <main className={`${isDashboard ? "" : "pt-16"} relative z-[1]`}>
          {children}
        </main>
        {!isDashboard && <Footer />}
      </body>
    </html>
  )
}