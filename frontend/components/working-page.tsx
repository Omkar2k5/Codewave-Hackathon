"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import {
  Play,
  Camera,
  Map,
  Route,
  Cloud,
  BarChart3,
  Monitor,
  ArrowRight,
} from "lucide-react";
import Link from "next/link";
import { useRef, useEffect, useState } from "react";

export default function WorkingPage() {
  const pipelineRef = useRef<HTMLDivElement>(null);
  const [windowWidth, setWindowWidth] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(800);

  const { scrollYProgress } = useScroll({
    target: pipelineRef,
    offset: ["start 0.1", "end 0.9"],
  });

  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
      setViewportHeight(window.innerHeight);
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const workflowSteps = [
    {
      icon: Camera,
      title: "Camera Feed & AI Models",
      description:
        "YOLOv8 detection and DeepSORT tracking from multiple camera feeds",
      color: "from-green-500 to-emerald-600",
      svg: (
        <svg
          className="w-16 h-16 text-green-400"
          fill="currentColor"
          viewBox="0 0 24 24"
        >
          <path d="M17 10.5V7a1 1 0 00-1-1H4a1 1 0 00-1 1v10a1 1 0 001 1h12a1 1 0 001-1v-3.5l4 2v-7l-4 2z" />
        </svg>
      ),
    },
    {
      icon: Cloud,
      title: "Cloud Clustering",
      description:
        "Advanced clustering algorithms to group and analyze crowd patterns",
      color: "from-blue-500 to-cyan-600",
      svg: (
        <svg
          className="w-16 h-16 text-blue-400"
          fill="currentColor"
          viewBox="0 0 24 24"
        >
          <path d="M18.5 12A2.5 2.5 0 0016 9.5a3 3 0 00-5.5-1.5A4 4 0 006 12a2 2 0 000 4h12.5a2.5 2.5 0 000-5z" />
          <circle cx="8" cy="10" r="1" />
          <circle cx="12" cy="8" r="1" />
          <circle cx="16" cy="10" r="1" />
        </svg>
      ),
    },
    {
      icon: Map,
      title: "Heatmap Generation",
      description:
        "Dynamic visualization of crowd distribution and movement patterns",
      color: "from-purple-500 to-pink-600",
      svg: (
        <svg
          className="w-16 h-16 text-purple-400"
          fill="currentColor"
          viewBox="0 0 24 24"
        >
          <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" />
          <circle cx="12" cy="9" r="2.5" />
          <rect x="6" y="16" width="3" height="3" rx="1" opacity="0.7" />
          <rect x="10" y="17" width="4" height="2" rx="1" opacity="0.5" />
          <rect x="15" y="16" width="2" height="3" rx="1" opacity="0.8" />
        </svg>
      ),
    },
    {
      icon: BarChart3,
      title: "AI Summarizer",
      description:
        "Intelligent analysis and insights generation from crowd data",
      color: "from-indigo-500 to-purple-600",
      svg: (
        <svg
          className="w-16 h-16 text-indigo-400"
          fill="currentColor"
          viewBox="0 0 24 24"
        >
          <path d="M9 2a1 1 0 000 2h6a1 1 0 100-2H9z" />
          <path d="M10.5 4.5h3l.5 2h-4l.5-2z" />
          <rect
            x="6"
            y="6"
            width="12"
            height="12"
            rx="2"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          />
          <path
            d="M9 10h6M9 12h4M9 14h5"
            stroke="currentColor"
            strokeWidth="1.5"
            fill="none"
          />
          <circle cx="18" cy="6" r="3" className="text-yellow-400" />
          <path
            d="M17 5l1 1 2-2"
            stroke="white"
            strokeWidth="1.5"
            fill="none"
          />
        </svg>
      ),
    },
    {
      icon: Route,
      title: "Route Optimization",
      description:
        "Smart escape route suggestions based on real-time crowd analysis",
      color: "from-orange-500 to-red-600",
      svg: (
        <svg
          className="w-16 h-16 text-orange-400"
          fill="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"
            opacity="0.3"
          />
          <path
            d="M3 12h3l2-4 4 8 2-4h7"
            stroke="currentColor"
            strokeWidth="2"
            fill="none"
          />
          <circle cx="5" cy="12" r="2" />
          <circle cx="19" cy="12" r="2" />
        </svg>
      ),
    },
    {
      icon: Monitor,
      title: "Real-time Dashboard",
      description:
        "Live monitoring interface with actionable insights and alerts",
      color: "from-pink-500 to-rose-600",
      svg: (
        <svg
          className="w-16 h-16 text-pink-400"
          fill="currentColor"
          viewBox="0 0 24 24"
        >
          <rect
            x="2"
            y="4"
            width="20"
            height="12"
            rx="2"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          />
          <path d="M8 20h8M12 16v4" />
          <rect x="4" y="6" width="4" height="3" rx="1" opacity="0.7" />
          <rect x="9" y="6" width="6" height="2" rx="1" opacity="0.5" />
          <rect x="16" y="6" width="4" height="4" rx="1" opacity="0.8" />
          <circle cx="6" cy="12" r="1" className="text-green-400" />
          <circle cx="9" cy="12" r="1" className="text-yellow-400" />
          <circle cx="12" cy="12" r="1" className="text-red-400" />
        </svg>
      ),
    },
  ];

  // Calculate transforms for synchronized scroll animation
  const stepProgresses = workflowSteps.map((_, index) => {
    // Each step gets 1/6 of the scroll progress (since we have 6 steps)
    const stepStart = index / workflowSteps.length;
    const stepEnd = (index + 1) / workflowSteps.length;

    return useTransform(scrollYProgress, [stepStart, stepEnd], [0, 1]);
  });

  // Path progress should complete just as the next block reaches center
  const pathProgresses = workflowSteps.slice(0, -1).map((_, index) => {
    const pathStart = index / workflowSteps.length;
    const pathEnd = (index + 1) / workflowSteps.length;

    return useTransform(scrollYProgress, [pathStart, pathEnd], [0, 1]);
  });

  const stepTransforms = workflowSteps.map((_, index) => {
    const progress = stepProgresses[index];
    const isLeft = index % 2 === 0;

    return {
      // First card is always visible, others fade in with scroll
      opacity: index === 0 ? 1 : useTransform(progress, [0, 0.5], [0, 1]),
      x: index === 0 ? 0 : useTransform(progress, [0, 0.5], [isLeft ? -200 : 200, 0]),
      scale: index === 0 ? 1 : useTransform(progress, [0, 0.5], [0.8, 1]),
    };
  });

  return (
    <div className="min-h-screen from-gray-900 via-blue-900/10 to-purple-900/10 relative z-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 pt-32 pb-32">
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
            A sophisticated AI pipeline that transforms raw camera feeds into
            actionable crowd intelligence
          </p>
        </motion.div>

        {/* Demo Video Section */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-12 md:mb-16"
        >
          <div className="bg-gray-800/40 backdrop-blur-sm rounded-2xl p-4 md:p-6 border border-gray-700/50 max-w-xs sm:max-w-2xl md:max-w-3xl lg:max-w-4xl mx-auto">
            <h2 className="text-xl md:text-2xl font-bold text-center mb-4 md:mb-6 text-blue-400">
              System in Action
            </h2>
            <div className="relative aspect-video bg-gray-800 rounded-xl overflow-hidden group cursor-pointer">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 to-purple-600/20 flex items-center justify-center">
                <div className="bg-white/10 backdrop-blur-sm rounded-full p-3 md:p-4 group-hover:scale-110 transition-transform duration-300">
                  <Play className="w-6 h-6 md:w-9 md:h-9 text-white ml-0.5" />
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Data Pipeline */}
        <motion.div
          ref={pipelineRef}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="relative"
        >
            <h2 className="text-3xl font-bold text-center mb-8 text-blue-400">
              Data Pipeline
            </h2>

            {/* Pipeline Container - restored height to prevent overlap */}
           <div className="min-h-[400vh] overflow-hidden">
            {/* Animated Path */}
            {windowWidth > 0 && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                style={{ zIndex: 1 }}
              >
                <defs>
                  <filter id="glow">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                    <feMerge>
                      <feMergeNode in="coloredBlur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                </defs>

                {/* Create connecting paths between blocks - exactly as shown in image */}
                {workflowSteps.map((_, index) => {
                  if (index === workflowSteps.length - 1) return null;

                  const progress = pathProgresses[index];
                  const cardWidth = 384;
                  const cardHeight = 280; // Approximate card height
                  const margin = 80;

                                     // Calculate positions for each block (centered in viewport)
                   const currentBlockY = index === 0
                     ? viewportHeight * 0.35
                     : viewportHeight * (0.5 + index * 0.6);
                   const nextBlockY = index === 0
                     ? viewportHeight * (0.5 + 1 * 0.6)
                     : viewportHeight * (0.5 + (index + 1) * 0.6);

                  const isCurrentLeft = index % 2 === 0;
                  const isNextLeft = (index + 1) % 2 === 0;

                  // Position blocks on alternating sides - ensure they stay within viewport
                  const containerWidth = Math.min(windowWidth, 1200); // More conservative max width
                  const currentBlockX = isCurrentLeft
                    ? 100 + cardWidth / 2 // Left side with fixed margin
                    : containerWidth - 100 - cardWidth / 2; // Right side with fixed margin
                  const nextBlockX = isNextLeft
                    ? 100 + cardWidth / 2 // Left side with fixed margin
                    : containerWidth - 100 - cardWidth / 2; // Right side with fixed margin

                  // Start from the right side of left blocks, left side of right blocks
                  const startX = isCurrentLeft
                    ? currentBlockX + cardWidth / 2 // Right edge of left block
                    : currentBlockX - cardWidth / 2; // Left edge of right block
                  const startY = currentBlockY + cardHeight / 2; // Bottom center of current block

                  // End at the top center of the next block
                  const endX = nextBlockX;
                  const endY = nextBlockY - cardHeight / 2; // Top center of next block

                  // Create curved path connecting the blocks
                  const controlX1 = startX + (isCurrentLeft ? 100 : -100);
                  const controlY1 = startY + 50;
                  const controlX2 = endX + (isNextLeft ? -100 : 100);
                  const controlY2 = endY - 50;

                  const pathData = `M ${startX} ${startY} 
                                   C ${controlX1} ${controlY1} ${controlX2} ${controlY2} ${endX} ${endY}`;

                  return (
                    <g key={index}>
                      <motion.path
                        d={pathData}
                        stroke="white"
                        strokeWidth="3"
                        fill="none"
                        filter="url(#glow)"
                        style={{
                          pathLength: progress,
                        }}
                        strokeLinecap="round"
                        strokeDasharray="8 8"
                        opacity="0.7"
                      />
                      {/* Animated dot following the path */}
                      <motion.circle
                        r="4"
                        fill="white"
                        filter="url(#glow)"
                        style={{
                          offsetDistance: progress,
                          offsetPath: `path('${pathData}')`,
                          opacity: progress,
                        }}
                      />
                    </g>
                  );
                })}
              </svg>
            )}

            {/* Pipeline Steps */}
            {workflowSteps.map((step, index) => {
              const Icon = step.icon;
              const transforms = stepTransforms[index];

                             // Position blocks to be centered in viewport when visible
               const isLeft = index % 2 === 0;
               const topPosition = (index === 0
                 ? viewportHeight * 0.35
                 : viewportHeight * (0.5 + index * 0.6)) - 140; // First block closer, others original spacing

              // Position blocks on alternating sides - ensure they stay within viewport
              const cardWidth = 384;
              const containerWidth = Math.min(windowWidth, 1200); // More conservative max width

              const leftPosition = isLeft
                ? 100 // Left side with fixed margin
                : containerWidth - cardWidth - 100; // Right side with fixed margin

              return (
                <motion.div
                  key={index}
                  className="absolute w-96"
                  style={{
                    top: `${topPosition}px`,
                    left: `${leftPosition}px`,
                    zIndex: 2,
                    opacity: transforms.opacity,
                    x: transforms.x,
                    scale: transforms.scale,
                  }}
                >
                  <motion.div
                    className="relative bg-gray-900/90 backdrop-blur-xl rounded-3xl p-8 border border-gray-700/40 hover:border-blue-400/60 transition-all duration-500 shadow-2xl"
                    whileHover={{
                      scale: 1.02,
                      boxShadow: "0 30px 60px -12px rgba(0, 0, 0, 0.6)",
                    }}
                  >
                    {/* Glowing background effect matching step color */}
                    <div
                      className={`absolute inset-0 bg-gradient-to-br ${step.color} opacity-5 rounded-3xl`}
                    />

                    {/* Header with icon and title */}
                    <div className="relative flex items-start mb-6">
                      <motion.div
                        className={`w-16 h-16 bg-gradient-to-br ${step.color} rounded-2xl flex items-center justify-center mr-6 shadow-lg`}
                        whileHover={{ rotate: 5, scale: 1.1 }}
                        transition={{ type: "spring", stiffness: 300 }}
                      >
                        <Icon size={28} className="text-white" />
                      </motion.div>
                      <div className="flex-1">
                        <motion.div className="text-sm text-blue-400 font-semibold mb-2 tracking-wider">
                          STEP {index + 1}
                        </motion.div>
                        <h3 className="text-2xl font-bold text-white leading-tight mb-2">
                          {step.title}
                        </h3>
                      </div>
                    </div>

                    {/* SVG Illustration */}
                    <div className="flex justify-center mb-6">
                      <motion.div
                        className="p-4 bg-gray-800/50 rounded-2xl"
                        whileHover={{ scale: 1.05 }}
                        transition={{ type: "spring", stiffness: 300 }}
                      >
                        {step.svg}
                      </motion.div>
                    </div>

                    {/* Description */}
                    <p className="text-gray-300 leading-relaxed text-lg mb-6 text-center">
                      {step.description}
                    </p>


                  </motion.div>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* Technical Highlights */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.4 }}
                     className="relative bg-gray-900/90 backdrop-blur-xl rounded-3xl p-8 border border-gray-700/40 hover:border-blue-400/60 transition-all duration-500 shadow-2xl mb-8"
        >
          {/* Glowing background effect */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 rounded-3xl" />
          <h2 className="text-2xl font-bold text-center mb-8 text-blue-400">
            Key Technical Features
          </h2>
          
          {/* Infinite Scrolling Features */}
          <div className="space-y-8">
            {/* First Row - Moving Left to Right */}
            <div className="relative overflow-hidden">
              <motion.div
                className="flex space-x-6"
                animate={{
                  x: [0, -1000],
                }}
                transition={{
                  duration: 20,
                  repeat: Infinity,
                  ease: "linear",
                }}
              >
                {/* Repeat the features 3 times for seamless loop */}
                {[...Array(3)].map((_, repeatIndex) => (
                  <div key={`row1-${repeatIndex}`} className="flex space-x-6">
                    {[
                      {
                        color: "from-blue-500 to-cyan-500",
                        title: "Real-time Processing",
                        description: "Sub-second latency processing",
                        icon: "⚡"
                      },
                      {
                        color: "from-purple-500 to-pink-500",
                        title: "Multi-camera Sync",
                        description: "Kafka-based synchronization",
                        icon: "📹"
                      },
                      {
                        color: "from-green-500 to-emerald-500",
                        title: "Advanced Clustering",
                        description: "Crowd pattern analysis",
                        icon: "🔍"
                      }
                    ].map((feature, index) => (
                      <motion.div
                        key={`feature1-${repeatIndex}-${index}`}
                        className="relative bg-gray-800/50 backdrop-blur-sm rounded-2xl p-6 border border-gray-700/40 hover:border-blue-400/60 transition-all duration-300 shadow-lg min-w-[280px]"
                        whileHover={{
                          scale: 1.05,
                          boxShadow: "0 20px 40px -12px rgba(0, 0, 0, 0.4)",
                        }}
                      >
                        {/* Glowing background effect */}
                        <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-5 rounded-2xl`} />
                        
                        <div className="relative">
                          <div className="flex items-center mb-4">
                            <div className={`w-12 h-12 bg-gradient-to-br ${feature.color} rounded-xl flex items-center justify-center mr-4 text-2xl`}>
                              {feature.icon}
                            </div>
                            <h3 className="text-lg font-semibold text-white">
                              {feature.title}
                            </h3>
                          </div>
                          <p className="text-gray-300 text-sm">
                            {feature.description}
                          </p>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                ))}
              </motion.div>
            </div>

            {/* Second Row - Moving Right to Left */}
            <div className="relative overflow-hidden">
              <motion.div
                className="flex space-x-6"
                animate={{
                  x: [-1000, 0],
                }}
                transition={{
                  duration: 25,
                  repeat: Infinity,
                  ease: "linear",
                }}
              >
                {/* Repeat the features 3 times for seamless loop */}
                {[...Array(3)].map((_, repeatIndex) => (
                  <div key={`row2-${repeatIndex}`} className="flex space-x-6">
                    {[
                      {
                        color: "from-orange-500 to-red-500",
                        title: "Heatmap Generation",
                        description: "Dynamic visualization",
                        icon: "🗺️"
                      },
                      {
                        color: "from-pink-500 to-rose-500",
                        title: "Route Optimization",
                        description: "Smart escape routes",
                        icon: "🛣️"
                      },
                      {
                        color: "from-cyan-500 to-blue-500",
                        title: "Scalable Architecture",
                        description: "Large-scale deployments",
                        icon: "🏗️"
                      }
                    ].map((feature, index) => (
                      <motion.div
                        key={`feature2-${repeatIndex}-${index}`}
                        className="relative bg-gray-800/50 backdrop-blur-sm rounded-2xl p-6 border border-gray-700/40 hover:border-blue-400/60 transition-all duration-300 shadow-lg min-w-[280px]"
                        whileHover={{
                          scale: 1.05,
                          boxShadow: "0 20px 40px -12px rgba(0, 0, 0, 0.4)",
                        }}
                      >
                        {/* Glowing background effect */}
                        <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-5 rounded-2xl`} />
                        
                        <div className="relative">
                          <div className="flex items-center mb-4">
                            <div className={`w-12 h-12 bg-gradient-to-br ${feature.color} rounded-xl flex items-center justify-center mr-4 text-2xl`}>
                              {feature.icon}
                            </div>
                            <h3 className="text-lg font-semibold text-white">
                              {feature.title}
                            </h3>
                          </div>
                          <p className="text-gray-300 text-sm">
                            {feature.description}
                          </p>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                ))}
              </motion.div>
            </div>
          </div>
        </motion.div>

        {/* CTA */}
                 <motion.div
           initial={{ opacity: 0, y: 20 }}
           animate={{ opacity: 1, y: 0 }}
           transition={{ duration: 0.8, delay: 1.6 }}
           className="flex justify-center"
         >
          <Link href="/dashboard">
            <motion.button
              className="group relative px-6 md:px-10 py-3 md:py-4 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-blue-600/20 backdrop-blur-sm border border-blue-400/30 text-blue-300 font-semibold rounded-xl hover:bg-gradient-to-r hover:from-blue-500/30 hover:via-purple-500/30 hover:to-blue-600/30 hover:border-blue-400/50 hover:text-white transition-all duration-500 flex items-center gap-2 md:gap-3 shadow-lg hover:shadow-blue-500/25"
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
            >
              <span className="text-sm md:text-base">View Live Dashboard</span>
              <ArrowRight className="w-4 h-4 md:w-5 md:h-5 group-hover:translate-x-1 transition-transform duration-300" />
            </motion.button>
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
