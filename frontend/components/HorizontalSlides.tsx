"use client";

import { useRef, useState, useEffect } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import Image from "next/image";

interface Slide {
  id: string;
  title: string;
  description: string;
  image?: string;
  icon?: React.ReactNode;
  stats?: { label: string; value: string }[];
}

interface HorizontalSlidesProps {
  slides: Slide[];
  className?: string;
}

export default function HorizontalSlides({
  slides,
  className = "",
}: HorizontalSlidesProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentSlide, setCurrentSlide] = useState(0);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"],
  });

  // Calculate which slide should be active based on scroll progress
  useEffect(() => {
    const unsubscribe = scrollYProgress.onChange((progress) => {
      const slideIndex = Math.floor(progress * slides.length);
      const clampedIndex = Math.max(0, Math.min(slides.length - 1, slideIndex));
      setCurrentSlide(clampedIndex);
    });

    return unsubscribe;
  }, [scrollYProgress, slides.length]);

  return (
    <div
      ref={containerRef}
      className={`relative ${className}`}
      style={{ height: `${slides.length * 100}vh` }}
    >
      <div className="sticky top-0 h-screen overflow-hidden">
        {slides.map((slide, index) => {
          const isActive = index === currentSlide;
          const isPrev = index < currentSlide;
          const isNext = index > currentSlide;

          return (
            <motion.div
              key={slide.id}
              className="absolute inset-0 w-full h-full flex items-center justify-center px-8"
              initial={{ clipPath: "inset(0 100% 0 0)" }}
              animate={{
                clipPath: isActive
                  ? "inset(0 0% 0 0%)"
                  : isPrev
                  ? "inset(0 100% 0 0%)"
                  : "inset(0 0% 0 100%)",
              }}
              transition={{
                duration: 0.8,
                ease: [0.25, 1, 0.3, 1],
              }}
            >
              <div className="max-w-6xl mx-auto text-center">
                <motion.div
                  initial={{ opacity: 0, y: 50 }}
                  animate={{
                    opacity: isActive ? 1 : 0,
                    y: isActive ? 0 : 50,
                  }}
                  transition={{ duration: 0.6, delay: isActive ? 0.3 : 0 }}
                >
                  {slide.image && (
                    <div className="mb-8 relative w-full max-w-4xl mx-auto">
                      <div className="aspect-video rounded-2xl overflow-hidden shadow-2xl bg-gray-800/50">
                        <Image
                          src={slide.image}
                          alt={slide.title}
                          fill
                          className="object-cover"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            const container = target.closest(".aspect-video");
                            if (container) {
                              (
                                container.parentElement as HTMLElement
                              ).style.display = "none";
                            }
                          }}
                        />
                      </div>
                    </div>
                  )}

                  {slide.icon && (
                    <div className="mb-8 flex justify-center">
                      <div className="w-24 h-24 bg-gradient-to-br from-red-500 to-orange-600 rounded-full flex items-center justify-center">
                        {slide.icon}
                      </div>
                    </div>
                  )}

                  <h2 className="text-4xl pb-4 md:text-6xl font-bold mb-6 bg-gradient-to-r from-red-400 via-orange-400 to-yellow-400 bg-clip-text text-transparent">
                    {slide.title}
                  </h2>

                  <p className="text-xl md:text-2xl text-gray-300 max-w-4xl mx-auto leading-relaxed mb-8">
                    {slide.description}
                  </p>

                  {slide.stats && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12">
                      {slide.stats.map((stat, statIndex) => (
                        <motion.div
                          key={statIndex}
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{
                            opacity: isActive ? 1 : 0,
                            scale: isActive ? 1 : 0.8,
                          }}
                          transition={{
                            duration: 0.6,
                            delay: isActive ? 0.5 + statIndex * 0.1 : 0,
                          }}
                          className="bg-red-900/20 backdrop-blur-sm rounded-xl p-6 border border-red-500/30"
                        >
                          <div className="text-3xl md:text-4xl font-bold text-red-400 mb-2">
                            {stat.value}
                          </div>
                          <div className="text-gray-300 text-lg">
                            {stat.label}
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </motion.div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
