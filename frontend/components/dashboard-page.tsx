"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Home, Route, Brain, Camera, Activity } from "lucide-react"
import Link from "next/link"

export default function DashboardPage() {
  const [currentMapView, setCurrentMapView] = useState(0) // 0: heatmap, 1: normal map
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setIsLoaded(true), 500)
    return () => clearTimeout(timer)
  }, [])

  const mapViews = ["Heatmap View", "Normal Map View"]

  // Placeholder camera feeds
  const cameraFeeds = [
    { id: 1, name: "Gate A", status: "active" },
    { id: 2, name: "Main Hall", status: "active" },
    { id: 3, name: "Exit B", status: "active" },
    { id: 4, name: "Corridor C", status: "active" },
  ]

  return (
    <div className="h-screen bg-gradient-to-br from-gray-900 via-blue-900/10 to-purple-900/10 overflow-hidden">
      {/* Home Icon */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6 }}
        className="absolute top-4 left-4 z-50"
      >
        <Link href="/">
          <button className="bg-gray-800/80 backdrop-blur-sm hover:bg-gray-700/80 p-3 rounded-full border border-gray-700/50 transition-all duration-300 hover:scale-110">
            <Home size={20} className="text-white" />
          </button>
        </Link>
      </motion.div>

      {/* Main Content */}
      <div className="h-full flex">
        {/* Left Side - Camera Feeds */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: isLoaded ? 1 : 0, x: isLoaded ? 0 : -50 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="w-1/2 p-6 pr-3"
        >
          <h2 className="text-2xl font-bold text-blue-400 mb-6 flex items-center space-x-2">
            <Camera size={24} />
            <span>Live Camera Feeds</span>
          </h2>

          <div className="grid grid-cols-2 gap-4 h-[calc(100vh-120px)]">
            {cameraFeeds.map((feed, index) => (
              <motion.div
                key={feed.id}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: isLoaded ? 1 : 0, scale: isLoaded ? 1 : 0.8 }}
                transition={{ duration: 0.6, delay: 0.4 + index * 0.1 }}
                className="bg-gray-800/40 backdrop-blur-sm rounded-xl border border-gray-700/50 overflow-hidden relative group hover:border-blue-500/50 transition-all duration-300"
              >
                {/* Camera Feed Placeholder */}
                <div className="aspect-video bg-gradient-to-br from-gray-700 to-gray-800 relative">
                  <div className="absolute inset-0 shimmer opacity-20"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <Activity size={32} className="text-gray-500 mx-auto mb-2" />
                      <p className="text-gray-400 text-sm">Camera {feed.id}</p>
                    </div>
                  </div>

                  {/* Status Indicator */}
                  <div className="absolute top-3 right-3 flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-xs text-green-400 bg-black/50 px-2 py-1 rounded">LIVE</span>
                  </div>
                </div>

                {/* Feed Info */}
                <div className="p-3">
                  <h3 className="font-semibold text-white">{feed.name}</h3>
                  <p className="text-xs text-gray-400">Status: {feed.status}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Right Side - Maps and Controls */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: isLoaded ? 1 : 0, x: isLoaded ? 0 : 50 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="w-1/2 p-6 pl-3 flex flex-col"
        >
          {/* Map Section */}
          <div className="flex-1 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-blue-400 flex items-center space-x-2">
                <Brain size={24} />
                <span>Crowd Analysis</span>
              </h2>

              {/* Map Navigation */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setCurrentMapView(0)}
                  className={`px-3 py-1 rounded-lg text-sm transition-all duration-300 ${
                    currentMapView === 0
                      ? "bg-blue-600 text-white"
                      : "bg-gray-700/50 text-gray-300 hover:bg-gray-600/50"
                  }`}
                >
                  Heatmap
                </button>
                <button
                  onClick={() => setCurrentMapView(1)}
                  className={`px-3 py-1 rounded-lg text-sm transition-all duration-300 ${
                    currentMapView === 1
                      ? "bg-blue-600 text-white"
                      : "bg-gray-700/50 text-gray-300 hover:bg-gray-600/50"
                  }`}
                >
                  Normal
                </button>
              </div>
            </div>

            {/* Map Display */}
            <motion.div
              key={currentMapView}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
              className="bg-gray-800/40 backdrop-blur-sm rounded-xl border border-gray-700/50 h-[60vh] relative overflow-hidden"
            >
              <div className="absolute inset-0 shimmer opacity-10"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div
                    className={`w-16 h-16 rounded-full mx-auto mb-4 flex items-center justify-center ${
                      currentMapView === 0
                        ? "bg-gradient-to-br from-red-500 to-orange-500"
                        : "bg-gradient-to-br from-blue-500 to-purple-500"
                    }`}
                  >
                    <Brain size={32} className="text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">{mapViews[currentMapView]}</h3>
                  <p className="text-gray-400 text-sm">
                    {currentMapView === 0
                      ? "Real-time crowd density visualization"
                      : "Standard venue layout with crowd markers"}
                  </p>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Controls Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: isLoaded ? 1 : 0, y: isLoaded ? 0 : 20 }}
            transition={{ duration: 0.6, delay: 0.8 }}
            className="space-y-4"
          >
            {/* Escape Routes Button */}
            <button className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 p-4 rounded-xl transition-all duration-300 transform hover:scale-105 flex items-center justify-center space-x-3">
              <Route size={20} />
              <span className="font-semibold">Show Escape Routes</span>
            </button>

            {/* AI Situation Summariser */}
            <div className="bg-gray-800/40 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4">
              <h3 className="font-semibold text-white mb-3 flex items-center space-x-2">
                <Brain size={18} />
                <span>AI Situation Summary</span>
              </h3>
              <div className="bg-gray-700/30 rounded-lg p-3 mb-3">
                <p className="text-gray-300 text-sm leading-relaxed">
                  Current crowd density: <span className="text-green-400 font-semibold">Moderate</span>
                  <br />
                  Peak areas: Gate A, Main Hall
                  <br />
                  Recommended action: Monitor exit flows
                </p>
              </div>
              <button className="w-full bg-blue-600 hover:bg-blue-700 py-2 rounded-lg text-sm font-medium transition-all duration-300">
                Generate New Summary
              </button>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  )
}
