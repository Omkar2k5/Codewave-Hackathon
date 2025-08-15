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
import HorizontalSlides from "./HorizontalSlides";
import SplitText from "./SplitText";

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
      <div className="relative z-20 min-h-screen flex items-center justify-center pt-20">
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

      {/* Problem Statement Horizontal Slides */}
      <div className="relative z-20">
        <HorizontalSlides slides={problemSlides} />
      </div>
    </div>
  );
}
