"use client";

import { useRef, useState, useEffect } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { ArrowRight, ArrowDown } from "lucide-react";
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
  const [hasScrolledPast, setHasScrolledPast] = useState(false);

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

  // Check if user has scrolled near the end of the page
  useEffect(() => {
    const checkScrollPosition = () => {
      const scrollTop = window.scrollY;
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight;

      // Hide button when user is within 200px of the bottom of the page
      const distanceFromBottom = documentHeight - (scrollTop + windowHeight);
      setHasScrolledPast(distanceFromBottom < 200);
    };

    // Check on mount
    checkScrollPosition();

    // Check on scroll
    window.addEventListener("scroll", checkScrollPosition);
    window.addEventListener("resize", checkScrollPosition);

    return () => {
      window.removeEventListener("scroll", checkScrollPosition);
      window.removeEventListener("resize", checkScrollPosition);
    };
  }, []);

  // Function to scroll to next slide or to explore working button
  const handleNavigation = () => {
    if (currentSlide < slides.length - 1 && containerRef.current) {
      // Navigate to next slide
      const containerTop =
        containerRef.current.getBoundingClientRect().top + window.scrollY;
      const nextSlidePosition =
        containerTop + (currentSlide + 1) * window.innerHeight;

      window.scrollTo({
        top: nextSlidePosition,
        behavior: "smooth",
      });
    } else {
      // On last slide, scroll to the explore working button
      if (containerRef.current) {
        const containerBottom =
          containerRef.current.getBoundingClientRect().bottom + window.scrollY;
        // Scroll to show the explore working button (add some offset for better visibility)
        window.scrollTo({
          top: containerBottom - window.innerHeight * 0.3,
          behavior: "smooth",
        });
      }
    }
  };

  return (
    <div
      ref={containerRef}
      className={`relative ${className}`}
      style={{ height: `${slides.length * 100}vh` }}
    >
      <div className="sticky top-0 h-screen overflow-hidden">
        {/* Navigation Button - Hide when scrolled past slides */}
        {!hasScrolledPast && (
          <div className="absolute right-6 md:right-8 bottom-6 md:bottom-8 z-30">
            <motion.button
              onClick={handleNavigation}
              className="group relative p-3 md:p-4 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-blue-600/20 backdrop-blur-sm border border-blue-400/30 text-blue-300 font-semibold rounded-full hover:bg-gradient-to-r hover:from-blue-500/30 hover:via-purple-500/30 hover:to-blue-600/30 hover:border-blue-400/50 hover:text-white transition-all duration-500 shadow-lg hover:shadow-blue-500/25"
              whileHover={{
                scale: 1.1,
                x: currentSlide < slides.length - 1 ? 2 : 0,
                y: currentSlide < slides.length - 1 ? 0 : 2,
              }}
              whileTap={{ scale: 0.95 }}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ delay: 1, duration: 0.6 }}
            >
              {currentSlide < slides.length - 1 ? (
                <ArrowRight className="w-5 h-5 md:w-6 md:h-6" />
              ) : (
                <ArrowDown className="w-5 h-5 md:w-6 md:h-6" />
              )}
            </motion.button>
          </div>
        )}
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
