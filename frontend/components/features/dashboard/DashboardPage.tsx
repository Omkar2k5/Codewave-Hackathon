"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Home, Route, Brain, Camera, Activity } from "lucide-react"
import Link from "next/link"
import CrowdMap from "@/components/features/crowd-monitoring/CrowdMap"

// --- TYPES ---
interface Camera {
    id: string;
    name: string;
    lat: number;
    lng: number;
    status: string;
    fov: number; // Field of view in degrees
    direction: number; // Direction in degrees (0-360)
    fovRadius: number; // FOV triangle radius in meters
}

interface CameraCoverageCircle {
    center: { lat: number; lng: number };
    radius: number;
}

export default function DashboardPage() {
  const [currentMapView, setCurrentMapView] = useState(1) // 0: heatmap, 1: escape routes (default)
  const [isLoaded, setIsLoaded] = useState(false)
  const [cameraPlacementMode, setCameraPlacementMode] = useState(false)
  
  // Shared camera state between both map views
  const [selectedCameraPositions, setSelectedCameraPositions] = useState<Camera[]>([])
  const [cameraCoverageCircle, setCameraCoverageCircle] = useState<CameraCoverageCircle | null>(null)

  useEffect(() => {
    const timer = setTimeout(() => setIsLoaded(true), 500)
    return () => clearTimeout(timer)
  }, [])

  const mapViews = ["Heatmap View", "Escape Routes"]

  const openCameraPlacementMode = () => {
    setCameraPlacementMode(true)
  }

  const closeCameraPlacementMode = () => {
    setCameraPlacementMode(false)
  }

  const handleCameraPositionsUpdate = (cameras: Camera[]) => {
    setSelectedCameraPositions(cameras)
  }

  const handleCameraCoverageUpdate = (coverage: CameraCoverageCircle | null) => {
    setCameraCoverageCircle(coverage)
  }

  return (
    <div className="h-screen bg-gradient-to-br from-gray-900 via-blue-900/10 to-purple-900/10 overflow-hidden flex flex-col">
      {/* Top Bar: Home Icon + Place Cameras Button + Map Switcher */}
      <div className="flex items-center justify-between px-8 pt-6 pb-2">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
          className="z-50 flex items-center space-x-3"
        >
          <Link href="/">
            <button className="bg-gray-800/80 backdrop-blur-sm hover:bg-gray-700/80 p-3 rounded-full border border-gray-700/50 transition-all duration-300 hover:scale-110">
              <Home size={20} className="text-white" />
            </button>
          </Link>
          
                          {/* Place Cameras Button - Only show when in Escape Routes view */}
                {currentMapView === 1 && (
                  <motion.button
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.4, delay: 0.2 }}
                    onClick={openCameraPlacementMode}
                    className="bg-gray-800/80 backdrop-blur-sm hover:bg-gray-700/80 p-3 rounded-full border border-gray-700/50 transition-all duration-300 hover:scale-110"
                  >
                    <Camera size={20} />
                  </motion.button>
                )}
        </motion.div>
        
        {/* Map Navigation */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setCurrentMapView(1)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold border transition-all duration-300 ${
              currentMapView === 1
                ? "bg-blue-600 text-white border-blue-400 shadow"
                : "bg-gray-900/80 text-gray-200 border-gray-700 hover:bg-gray-800"
            }`}
          >
            Escape Routes
          </button>
          <button
            onClick={() => setCurrentMapView(0)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold border transition-all duration-300 ${
              currentMapView === 0
                ? "bg-blue-600 text-white border-blue-400 shadow"
                : "bg-gray-900/80 text-gray-200 border-gray-700 hover:bg-gray-800"
            }`}
          >
            Heatmap
          </button>
        </div>
      </div>

      {/* Main Content: Camera + Map */}
      <div className="flex flex-1 px-6 pb-6 gap-6 min-h-0">
        {/* Left Side - Two Camera Feeds */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: isLoaded ? 1 : 0, x: isLoaded ? 0 : -50 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="w-1/3 flex flex-col min-h-0"
        >
          {/* Camera Overlay 1 */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: isLoaded ? 1 : 0, scale: isLoaded ? 1 : 0.8 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="bg-gray-800/40 backdrop-blur-sm rounded-xl border border-gray-700/50 overflow-hidden relative group hover:border-blue-500/50 transition-all duration-300 flex flex-col mb-2"
          >
            {/* Camera Feed Placeholder - 16:9 ratio */}
            <div className="aspect-video bg-gradient-to-br from-gray-700 to-gray-800 relative flex-1 flex flex-col justify-center">
              <div className="absolute inset-0 shimmer opacity-20"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <Activity size={32} className="text-gray-500 mx-auto mb-2" />
                  <p className="text-gray-400 text-sm">Camera Overlay 1</p>
                </div>
              </div>
              {/* Status Indicator */}
              <div className="absolute top-3 right-3 flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-xs text-green-400 bg-black/50 px-2 py-1 rounded">LIVE</span>
              </div>
            </div>
            {/* Feed Info */}
            <div className="p-2">
              <h3 className="text-white">Camera Overlay 1</h3>
            </div>
          </motion.div>

          {/* Camera Overlay 2 */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: isLoaded ? 1 : 0, scale: isLoaded ? 1 : 0.8 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            className="bg-gray-800/40 backdrop-blur-sm rounded-xl border border-gray-700/50 overflow-hidden relative group hover:border-blue-500/50 transition-all duration-300 flex flex-col flex-1"
          >
            {/* Camera Feed Placeholder - 16:9 ratio */}
            <div className="aspect-video bg-gradient-to-br from-gray-700 to-gray-800 relative flex-1 flex flex-col justify-center">
              <div className="absolute inset-0 shimmer opacity-20"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <Activity size={32} className="text-gray-500 mx-auto mb-2" />
                  <p className="text-gray-400 text-sm">Camera Overlay 2</p>
                </div>
              </div>
              {/* Status Indicator */}
              <div className="absolute top-3 right-3 flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-xs text-green-400 bg-black/50 px-2 py-1 rounded">LIVE</span>
              </div>
            </div>
            {/* Feed Info */}
            <div className="p-2">
              <h3 className="text-white">Camera Overlay 2</h3>
            </div>
          </motion.div>
        </motion.div>

        {/* Right Side - Large Map Section */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: isLoaded ? 1 : 0, x: isLoaded ? 0 : 50 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="w-2/3 flex flex-col min-h-0"
        >
          {/* Map Display */}
          <motion.div
            key={currentMapView}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
            className="bg-gray-800/40 backdrop-blur-sm rounded-xl border border-gray-700/50 flex-1 relative overflow-hidden"
          >
            {/* Both views now use the CrowdMap component with different mapType props */}
            <CrowdMap 
              cameraPlacementMode={cameraPlacementMode}
              onCloseCameraPlacementMode={closeCameraPlacementMode}
              mapType={currentMapView === 0 ? 'heatmap' : 'escape-routes'}
              selectedCameraPositions={selectedCameraPositions}
              onCameraPositionsUpdate={handleCameraPositionsUpdate}
              cameraCoverageCircle={cameraCoverageCircle}
              onCameraCoverageUpdate={handleCameraCoverageUpdate}
            />
          </motion.div>
        </motion.div>
      </div>
    </div>
  )
}
