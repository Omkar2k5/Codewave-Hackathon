"use client"

import type React from "react"

import { Inter } from "next/font/google"
import "./globals.css"
import Navbar from "@/components/navbar"
import Footer from "@/components/footer"
import { usePathname } from "next/navigation"

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
      <body className={`${inter.className} bg-gray-900 text-white min-h-screen`}>
        {!isDashboard && <Navbar />}
        <main className={isDashboard ? "" : "pt-16"}>{children}</main>
        {!isDashboard && <Footer />}
      </body>
    </html>
  )
}