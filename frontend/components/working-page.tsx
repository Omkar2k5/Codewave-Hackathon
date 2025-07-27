"use client"

import { motion } from "framer-motion"
import { Play, Camera, Brain, Map, Route } from "lucide-react"
import Link from "next/link"

export default function WorkingPage() {
  const workflowSteps = [
    {
      icon: Camera,
      title: "YOLOv8 Detection",
      description: "Real-time person detection and tracking from multiple camera feeds",
      color: "from-green-500 to-emerald-600",
    },
    {
      icon: Brain,
      title: "Crowd Clustering",
      description: "AI-powered analysis to identify crowd density patterns and hotspots",
      color: "from-blue-500 to-cyan-600",
    },
    {
      icon: Map,
      title: "Heatmap Generation",
      description: "Dynamic visualization of crowd distribution and movement patterns",
      color: "from-purple-500 to-pink-600",
    },
    {
      icon: Route,
      title: "Route Optimization",
      description: "Intelligent escape route suggestions based on real-time crowd data",
      color: "from-orange-500 to-red-600",
    },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900/10 to-purple-900/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            How Drishti Works
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            A sophisticated AI pipeline that transforms raw camera feeds into actionable crowd intelligence
          </p>
        </motion.div>

        {/* Demo Video Section */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <div className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50">
            <h2 className="text-3xl font-bold text-center mb-8 text-blue-400">System in Action</h2>
            <div className="relative aspect-video bg-gray-800 rounded-xl overflow-hidden group cursor-pointer">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 to-purple-600/20 flex items-center justify-center">
                <div className="bg-white/10 backdrop-blur-sm rounded-full p-6 group-hover:scale-110 transition-transform duration-300">
                  <Play size={48} className="text-white ml-2" />
                </div>
              </div>
              <div className="absolute bottom-4 left-4 right-4">
                <div className="bg-black/50 backdrop-blur-sm rounded-lg p-3">
                  <p className="text-white text-sm">
                    Live demonstration: YOLOv8 detection → Person tracking → Crowd clustering → Heatmap overlay
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Workflow Steps */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mb-16"
        >
          <h2 className="text-3xl font-bold text-center mb-12 text-blue-400">Data Pipeline</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {workflowSteps.map((step, index) => {
              const Icon = step.icon
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.8 + index * 0.2 }}
                  className="relative"
                >
                  <div className="bg-gray-800/40 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50 hover:border-blue-500/50 transition-all duration-300 h-full">
                    <div
                      className={`w-12 h-12 bg-gradient-to-br ${step.color} rounded-lg flex items-center justify-center mb-4`}
                    >
                      <Icon size={24} className="text-white" />
                    </div>
                    <h3 className="text-lg font-semibold mb-3 text-white">{step.title}</h3>
                    <p className="text-gray-300 text-sm leading-relaxed">{step.description}</p>
                  </div>

                  {/* Arrow connector */}
                  {index < workflowSteps.length - 1 && (
                    <div className="hidden lg:block absolute top-1/2 -right-3 transform -translate-y-1/2 z-10">
                      <div className="w-6 h-0.5 bg-gradient-to-r from-blue-500 to-purple-500"></div>
                      <div className="absolute right-0 top-1/2 transform -translate-y-1/2 w-0 h-0 border-l-4 border-l-purple-500 border-t-2 border-b-2 border-t-transparent border-b-transparent"></div>
                    </div>
                  )}
                </motion.div>
              )
            })}
          </div>
        </motion.div>

        {/* Technical Highlights */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.4 }}
          className="bg-gray-800/30 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50 mb-12"
        >
          <h2 className="text-2xl font-bold text-center mb-6 text-blue-400">Key Technical Features</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                <p className="text-gray-300">Real-time processing with sub-second latency</p>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
                <p className="text-gray-300">Multi-camera feed synchronization via Kafka</p>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                <p className="text-gray-300">Advanced clustering algorithms for crowd analysis</p>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-orange-500 rounded-full mt-2"></div>
                <p className="text-gray-300">Dynamic heatmap generation and visualization</p>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-pink-500 rounded-full mt-2"></div>
                <p className="text-gray-300">Intelligent route optimization algorithms</p>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-cyan-500 rounded-full mt-2"></div>
                <p className="text-gray-300">Scalable architecture for large-scale deployments</p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.6 }}
          className="text-center"
        >
          <Link href="/dashboard">
            <button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 px-8 py-4 rounded-full text-lg font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-2xl">
              View Live Dashboard
            </button>
          </Link>
        </motion.div>
      </div>
    </div>
  )
}
