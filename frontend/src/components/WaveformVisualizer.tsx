import { useEffect, useRef, useState } from 'react'

interface Props {
  analyser: AnalyserNode | null
  isActive: boolean
  height?: number
}

export default function WaveformVisualizer({ analyser, isActive, height = 80 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const rafRef = useRef<number>(0)
  const [canvasWidth, setCanvasWidth] = useState(600)

  // Dynamically track container width
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = Math.floor(entry.contentRect.width)
        if (w > 0) setCanvasWidth(w)
      }
    })
    ro.observe(container)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!

    cancelAnimationFrame(rafRef.current)

    if (!analyser || !isActive) {
      // Draw a beautiful idle "flatline" with subtle animated glow
      let idlePhase = 0
      const drawIdle = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        idlePhase += 0.03
        const midY = canvas.height / 2
        const glowAlpha = 0.12 + Math.sin(idlePhase) * 0.06

        // Subtle glow band
        const grd = ctx.createLinearGradient(0, midY - 12, 0, midY + 12)
        grd.addColorStop(0, `rgba(246, 74, 26, 0)`)
        grd.addColorStop(0.5, `rgba(246, 74, 26, ${glowAlpha})`)
        grd.addColorStop(1, `rgba(246, 74, 26, 0)`)
        ctx.fillStyle = grd
        ctx.fillRect(0, midY - 12, canvas.width, 24)

        // Flat centre line
        const lineGrd = ctx.createLinearGradient(0, 0, canvas.width, 0)
        lineGrd.addColorStop(0, 'rgba(246,74,26,0.1)')
        lineGrd.addColorStop(0.5, 'rgba(246,74,26,0.35)')
        lineGrd.addColorStop(1, 'rgba(246,74,26,0.1)')
        ctx.strokeStyle = lineGrd
        ctx.lineWidth = 1.5
        ctx.setLineDash([6, 8])
        ctx.beginPath()
        ctx.moveTo(0, midY)
        ctx.lineTo(canvas.width, midY)
        ctx.stroke()
        ctx.setLineDash([])
        rafRef.current = requestAnimationFrame(drawIdle)
      }
      drawIdle()
      return () => cancelAnimationFrame(rafRef.current)
    }

    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const draw = () => {
      rafRef.current = requestAnimationFrame(draw)
      analyser.getByteTimeDomainData(dataArray)

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const midY = canvas.height / 2

      // Glow layer (mirrored fill)
      const fillGrd = ctx.createLinearGradient(0, 0, 0, canvas.height)
      fillGrd.addColorStop(0, 'rgba(246,74,26,0.18)')
      fillGrd.addColorStop(0.5, 'rgba(246,74,26,0.06)')
      fillGrd.addColorStop(1, 'rgba(246,74,26,0.18)')
      ctx.fillStyle = fillGrd
      ctx.beginPath()
      const sliceW = canvas.width / bufferLength
      let x = 0
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0
        const y = (v * canvas.height) / 2
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
        x += sliceW
      }
      ctx.lineTo(canvas.width, midY)
      ctx.lineTo(0, midY)
      ctx.closePath()
      ctx.fill()

      // Main waveform line with gradient
      const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0)
      gradient.addColorStop(0, '#f64a1a')
      gradient.addColorStop(0.35, '#ff7c52')
      gradient.addColorStop(0.65, '#ff7c52')
      gradient.addColorStop(1, '#f64a1a')

      ctx.lineWidth = 2.5
      ctx.strokeStyle = gradient
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'
      ctx.beginPath()

      x = 0
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0
        const y = (v * canvas.height) / 2
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
        x += sliceW
      }
      ctx.lineTo(canvas.width, midY)
      ctx.stroke()

      // Second thinner highlight pass
      ctx.lineWidth = 1
      ctx.strokeStyle = 'rgba(255,255,255,0.35)'
      ctx.beginPath()
      x = 0
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0
        const y = (v * canvas.height) / 2
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
        x += sliceW
      }
      ctx.stroke()
    }

    draw()
    return () => cancelAnimationFrame(rafRef.current)
  }, [analyser, isActive, canvasWidth])

  return (
    <div
      ref={containerRef}
      className="waveform-container"
      style={{ width: '100%', position: 'relative' }}
    >
      <canvas
        ref={canvasRef}
        className="waveform"
        width={canvasWidth}
        height={height}
        style={{ display: 'block', width: '100%', height: `${height}px` }}
      />
    </div>
  )
}
