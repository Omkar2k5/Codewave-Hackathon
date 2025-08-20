"use client"

import { useRef, useEffect, useState } from 'react'
import { useScroll, useTransform, motion } from 'framer-motion'

interface SectionTransitionProps {
  children: React.ReactNode
  className?: string
  transitionType?: 'square-center' | 'wipe-left' | 'wipe-right'
  triggerOffset?: ["start end", "end start"] | ["start start", "end end"] | ["start center", "end center"]
}

export default function SectionTransition({ 
  children, 
  className = '',
  transitionType = 'square-center',
  triggerOffset = ["start end", "end start"] as const
}: SectionTransitionProps) {
  const ref = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: triggerOffset
  })

  const getClipPath = () => {
    switch (transitionType) {
      case 'square-center':
        return useTransform(
          scrollYProgress,
          [0, 0.3, 0.7, 1],
          [
            'inset(100% 100% 100% 100%)',
            'inset(0% 0% 0% 0%)',
            'inset(0% 0% 0% 0%)',
            'inset(100% 100% 100% 100%)'
          ]
        )
      case 'wipe-left':
        return useTransform(
          scrollYProgress,
          [0, 0.3, 0.7, 1],
          [
            'inset(0 100% 0 0)',
            'inset(0 0% 0 0)',
            'inset(0 0% 0 0)',
            'inset(0 100% 0 0)'
          ]
        )
      case 'wipe-right':
        return useTransform(
          scrollYProgress,
          [0, 0.3, 0.7, 1],
          [
            'inset(0 0 0 100%)',
            'inset(0 0 0 0%)',
            'inset(0 0 0 0%)',
            'inset(0 0 0 100%)'
          ]
        )
      default:
        return useTransform(scrollYProgress, [0, 1], ['inset(0)', 'inset(0)'])
    }
  }

  const clipPath = getClipPath()

  return (
    <motion.div
      ref={ref}
      className={`min-h-screen w-full ${className}`}
      style={{ clipPath }}
    >
      {children}
    </motion.div>
  )
}