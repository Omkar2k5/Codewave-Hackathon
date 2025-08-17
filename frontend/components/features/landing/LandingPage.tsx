"use client";

import { motion } from "framer-motion";
import {
  ArrowRight,
  Users,
  MapPin,
  Shield,
  AlertTriangle,
  Clock,
  Users2,
} from "lucide-react";
import Link from "next/link";
import { HorizontalSlides, SplitText } from "@/components/common";

export default function LandingPage() {
  const problemSlides = [
    {
      id: "scale",
      title: "Massive Scale Events",
      description:
        "India hosts the world's largest gatherings with millions of people. From Kumbh Mela attracting over 100 million devotees to religious festivals drawing massive crowds, the scale is unprecedented.",
      icon: <Users size={48} className="text-white" />,
      stats: [
        { label: "People at Kumbh Mela", value: "100M+" },
        { label: "Major Festivals Annually", value: "500+" },
        { label: "Average Crowd Size", value: "50K+" },
      ],
    },
    {
      id: "incidents",
      title: "Tragic Consequences",
      description:
        "Poor crowd management leads to devastating stampedes and casualties. Traditional manual monitoring fails to prevent disasters, resulting in loss of precious lives.",
      icon: <AlertTriangle size={48} className="text-white" />,
      stats: [
        { label: "Lives Lost Annually", value: "200+" },
        { label: "Major Stampedes (2000-2023)", value: "50+" },
        { label: "Average Response Time", value: "15min" },
      ],
    },
    {
      id: "challenges",
      title: "Current Limitations",
      description:
        "Manual observation, delayed responses, and lack of real-time data create dangerous situations. Security personnel cannot monitor vast areas effectively or predict crowd behavior patterns.",
      icon: <Clock size={48} className="text-white" />,
      stats: [
        { label: "Detection Delay", value: "10-15min" },
        { label: "Coverage Area per Guard", value: "Limited" },
        { label: "Prediction Accuracy", value: "Poor" },
      ],
    },
    {
      id: "solution",
      title: "AI-Powered Solution",
      description:
        "Drishti revolutionizes crowd management with real-time AI detection, instant alerts, and predictive analytics. Our system monitors vast areas simultaneously and prevents disasters before they occur.",
      icon: <Users2 size={48} className="text-white" />,
      stats: [
        { label: "Detection Time", value: "<1sec" },
        { label: "Coverage Area", value: "Unlimited" },
        { label: "Prediction Accuracy", value: "95%+" },
      ],
    },
  ];

  const features = [
    {
      icon: Users,
      title: "Mass Gatherings",
      description:
        "Managing crowds in festivals, religious events, and public spaces",
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
  ];

  return (
    <div className="relative min-h-screen pb-32">
      {/* Hero Section */}
      <div className="relative z-20 min-h-screen flex flex-col pt-20">
        <div className="flex-1 flex items-center justify-center">
          <div className="w-full max-w-7xl px-6 lg:px-8 text-center">
            <div className="mb-6">
              <SplitText
                text="Drishti"
                className="text-6xl md:text-8xl font-bold text-blue-400"
                splitType="chars"
                delay={150}
                duration={0.8}
                ease="power3.out"
                from={{ opacity: 0, y: 60 }}
                to={{ opacity: 1, y: 0 }}
                textAlign="center"
              />
            </div>
            <div className="max-w-3xl mx-auto mb-8">
              <SplitText
                text="Revolutionizing crowd management in India through real-time AI detection, intelligent analysis, and proactive safety measures"
                className="text-xl md:text-2xl text-gray-300 leading-relaxed"
                splitType="words"
                delay={80}
                duration={0.6}
                ease="power2.out"
                from={{ opacity: 0, y: 30 }}
                to={{ opacity: 1, y: 0 }}
                textAlign="center"
              />
            </div>
          </div>
        </div>

        {/* Dive Deeper Button - Positioned at bottom of hero */}
        <div className="flex justify-center pb-8 md:pb-12">
          <motion.button
            onClick={() => {
              const problemSection = document.querySelector(
                ".horizontal-slides-section"
              );
              problemSection?.scrollIntoView({ behavior: "smooth" });
            }}
            className="group relative px-6 md:px-10 py-3 md:py-4 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-blue-600/20 backdrop-blur-sm border border-blue-400/30 text-blue-300 font-semibold rounded-xl hover:bg-gradient-to-r hover:from-blue-500/30 hover:via-purple-500/30 hover:to-blue-600/30 hover:border-blue-400/50 hover:text-white transition-all duration-500 flex items-center gap-2 md:gap-3 shadow-lg hover:shadow-blue-500/25"
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 2, duration: 0.6 }}
          >
            <span className="text-sm md:text-base">Dive Deeper</span>
            <ArrowRight className="w-4 h-4 md:w-5 md:h-5 group-hover:translate-x-1 transition-transform duration-300" />
          </motion.button>
        </div>
      </div>

      {/* Problem Statement Horizontal Slides */}
      <div className="relative z-20 horizontal-slides-section">
        <HorizontalSlides slides={problemSlides} />
      </div>

      {/* Explore Working Button */}
      <div className="relative z-20 flex justify-center mt-6 md:mt-8 mb-4 md:mb-6">
        <Link href="/working">
          <motion.button
            className="group relative px-6 md:px-10 py-3 md:py-4 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-blue-600/20 backdrop-blur-sm border border-blue-400/30 text-blue-300 font-semibold rounded-xl hover:bg-gradient-to-r hover:from-blue-500/30 hover:via-purple-500/30 hover:to-blue-600/30 hover:border-blue-400/50 hover:text-white transition-all duration-500 flex items-center gap-2 md:gap-3 shadow-lg hover:shadow-blue-500/25"
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.5, duration: 0.6 }}
          >
            <span className="text-sm md:text-base">Explore Working</span>
            <ArrowRight className="w-4 h-4 md:w-5 md:h-5 group-hover:translate-x-1 transition-transform duration-300" />
          </motion.button>
        </Link>
      </div>
    </div>
  );
}
