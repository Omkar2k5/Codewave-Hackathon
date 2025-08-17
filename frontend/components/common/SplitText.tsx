"use client";

import { useRef, useEffect } from "react";

interface SplitTextProps {
  text: string;
  className?: string;
  delay?: number;
  duration?: number;
  ease?: string;
  splitType?: "chars" | "words" | "lines";
  from?: { opacity: number; y: number };
  to?: { opacity: number; y: number };
  threshold?: number;
  rootMargin?: string;
  textAlign?: "left" | "right" | "center" | "justify" | "start" | "end";
  onLetterAnimationComplete?: () => void;
}

const SplitText: React.FC<SplitTextProps> = ({
  text,
  className = "",
  delay = 100,
  duration = 0.6,
  ease = "power3.out",
  splitType = "chars",
  from = { opacity: 0, y: 40 },
  to = { opacity: 1, y: 0 },
  threshold = 0.1,
  rootMargin = "-100px",
  textAlign = "center",
  onLetterAnimationComplete,
}) => {
  const ref = useRef<HTMLParagraphElement>(null);
  const animationCompletedRef = useRef(false);

  useEffect(() => {
    if (typeof window === "undefined" || !ref.current || !text) return;

    // Dynamically import GSAP only on client side
    const initGSAP = async () => {
      try {
        let gsap: any;
        let ScrollTrigger: any;
        let GSAPSplitText: any;
        
        // Check if GSAP is already available (for development hot reload)
        if (typeof window !== "undefined" && (window as any).gsap) {
          gsap = (window as any).gsap;
          ScrollTrigger = (window as any).ScrollTrigger;
          GSAPSplitText = (window as any).SplitText;
          
          if (gsap && ScrollTrigger && GSAPSplitText) {
            gsap.registerPlugin(ScrollTrigger, GSAPSplitText);
          } else {
            throw new Error("GSAP plugins not available");
          }
        } else {
          const gsapModule = await import("gsap");
          const scrollTriggerModule = await import("gsap/ScrollTrigger");
          const splitTextModule = await import("gsap/SplitText");
          
          gsap = gsapModule.default || gsapModule;
          ScrollTrigger = scrollTriggerModule.ScrollTrigger || scrollTriggerModule.default;
          GSAPSplitText = splitTextModule.SplitText || splitTextModule.default;
          
          gsap.registerPlugin(ScrollTrigger, GSAPSplitText);
          
          // Store in window for development hot reload
          if (typeof window !== "undefined") {
            (window as any).gsap = gsap;
            (window as any).ScrollTrigger = ScrollTrigger;
            (window as any).SplitText = GSAPSplitText;
          }
        }

        const el = ref.current;
        if (!el) return;

        animationCompletedRef.current = false;

        const absoluteLines = splitType === "lines";
        if (absoluteLines) el.style.position = "relative";

        let splitter;
        try {
          splitter = new GSAPSplitText(el, {
            type: splitType,
            absolute: absoluteLines,
            linesClass: "split-line",
          });
        } catch (error) {
          console.error("Failed to create SplitText:", error);
          return;
        }

        let targets;
        switch (splitType) {
          case "lines":
            targets = splitter.lines;
            break;
          case "words":
            targets = splitter.words;
            break;
          case "chars":
            targets = splitter.chars;
            break;
          default:
            targets = splitter.chars;
        }

        if (!targets || targets.length === 0) {
          console.warn("No targets found for SplitText animation");
          splitter.revert();
          return;
        }

        targets.forEach((t: any) => {
          if (t instanceof HTMLElement) {
            t.style.willChange = "transform, opacity";
          }
        });

        const startPct = (1 - threshold) * 100;
        const marginMatch = /^(-?\d+(?:\.\d+)?)(px|em|rem|%)?$/.exec(rootMargin);
        const marginValue = marginMatch ? parseFloat(marginMatch[1]) : 0;
        const marginUnit = marginMatch ? marginMatch[2] || "px" : "px";
        const sign =
          marginValue < 0
            ? `-=${Math.abs(marginValue)}${marginUnit}`
            : `+=${marginValue}${marginUnit}`;
        const start = `top ${startPct}%${sign}`;

        const tl = gsap.timeline({
          scrollTrigger: {
            trigger: el,
            start,
            toggleActions: "play none none none",
            once: true,
            onToggle: (self: any) => {
              // Store reference if needed
            },
          },
          smoothChildTiming: true,
          onComplete: () => {
            animationCompletedRef.current = true;
            gsap.set(targets, {
              ...to,
              clearProps: "willChange",
              immediateRender: true,
            });
            onLetterAnimationComplete?.();
          },
        });

        tl.set(targets, { ...from, immediateRender: false, force3D: true });
        tl.to(targets, {
          ...to,
          duration,
          ease,
          stagger: delay / 1000,
          force3D: true,
        });

        return () => {
          tl.kill();
          gsap.killTweensOf(targets);
          if (splitter) {
            splitter.revert();
          }
        };
      } catch (error) {
        console.error("Failed to load GSAP:", error);
        // Fallback: just show the text without animation
        return;
      }
    };

    initGSAP();
  }, [text, delay, duration, ease, splitType, from, to, threshold, rootMargin, onLetterAnimationComplete]);

  return (
    <p
      ref={ref}
      className={`split-parent overflow-hidden inline-block whitespace-normal ${className}`}
      style={{
        textAlign,
        wordWrap: "break-word",
      }}
    >
      {text}
    </p>
  );
};

export default SplitText;
