"use client"

import { useEffect, useRef } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { Sphere, MeshDistortMaterial } from "@react-three/drei"
import { motion } from "framer-motion"
import * as THREE from "three"
import { ArrowRight, Users, MapPin, Shield } from "lucide-react"
import Link from "next/link"
import DarkVeil from "./DarkVeil"
import SplitText from "./SplitText"

export default function LandingPage() {
  const challenges = [
    {
      icon: Users,
      title: "Mass Gatherings",
      description: "Managing crowds in festivals, religious events, and public spaces",
    },
    {
      icon: MapPin,
      title: "Real-time Tracking",
      description: "Instant detection and monitoring of crowd density patterns",
    },
    {
      icon: Shield,
      title: "Safety Assurance",
      description: "Preventing stampedes and ensuring public safety through AI",
    },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br via-blue-900/20 to-purple-900/20 relative overflow-hidden">
      
      {/* Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20">
        <div className="text-center mb-16">
          <div className="mb-6">
            <SplitText
              text="Drishti"
              className="text-6xl md:text-8xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent"
              splitType="chars"
              delay={50}
              duration={0.8}
              from={{ opacity: 0, y: 60 }}
              to={{ opacity: 1, y: 0 }}
              threshold={0.1}
              rootMargin="-50px"
            />
          </div>
          
          <div className="mb-8">
            <SplitText
              text="Revolutionizing crowd management in India through real-time AI detection, intelligent analysis, and proactive safety measures"
              className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto leading-relaxed"
              splitType="words"
              delay={80}
              duration={0.6}
              from={{ opacity: 0, y: 30 }}
              to={{ opacity: 1, y: 0 }}
              threshold={0.1}
              rootMargin="-50px"
            />
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            <Link href="/working">
              <button className="group bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 px-8 py-4 rounded-full text-lg font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-2xl flex items-center space-x-2 mx-auto">
                <span>Explore the System</span>
                <ArrowRight className="group-hover:translate-x-1 transition-transform" size={20} />
              </button>
            </Link>
          </motion.div>
        </div>

        {/* Problem Statement */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mb-20"
        >
          <div className="bg-gray-800/30 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50">
            <h2 className="text-3xl font-bold text-center mb-8 text-blue-400">The Challenge We're Solving</h2>
            <p className="text-lg text-gray-300 text-center max-w-4xl mx-auto leading-relaxed">
              India hosts some of the world's largest gatherings - from Kumbh Mela to religious festivals. Traditional
              crowd management relies on manual observation, leading to delayed responses and safety risks.
              <span className="text-blue-400 font-semibold">
                {" "}
                Drishti transforms this with AI-powered real-time detection
              </span>
              , providing instant insights and proactive safety measures.
            </p>
          </div>
        </motion.div>

        {/* Key Features */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.9 }}
          className="grid md:grid-cols-3 gap-8 mb-20"
        >
          {challenges.map((challenge, index) => {
            const Icon = challenge.icon
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 1.1 + index * 0.2 }}
                className="bg-gray-800/40 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50 hover:border-blue-500/50 transition-all duration-300 hover:transform hover:scale-105"
              >
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center mb-4">
                  <Icon size={24} className="text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-blue-400">{challenge.title}</h3>
                <p className="text-gray-300 leading-relaxed">{challenge.description}</p>
              </motion.div>
            )
          })}
        </motion.div>
      </div>
    </div>
  )
}
