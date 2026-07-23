import { useRef, useState, useCallback } from 'react'
import axios from 'axios'

interface CameraViewProps {
  stagePos?: { x: number; y: number; z: number } | null
  onMoved?: () => void
}

export default function CameraView({ stagePos, onMoved }: CameraViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [clickPos, setClickPos] = useState<{ x: number; y: number } | null>(null)
  const [moving, setMoving] = useState(false)

  const handleClick = useCallback(
    async (e: React.MouseEvent<HTMLDivElement>) => {
      if (moving) return
      const rect = e.currentTarget.getBoundingClientRect()
      const relX = e.clientX - rect.left
      const relY = e.clientY - rect.top

      const STREAM_W = 1060
      const STREAM_H = 800
      const px = (relX / rect.width) * STREAM_W
      const py = (relY / rect.height) * STREAM_H

      setClickPos({ x: relX, y: relY })
      setMoving(true)

      try {
        await axios.post('/api/stage/move-pixel', {
          px,
          py,
          stream_width: STREAM_W,
          stream_height: STREAM_H,
        })
        onMoved?.()
      } catch (err) {
        console.error('Stage move failed:', err)
      } finally {
        setTimeout(() => {
          setClickPos(null)
          setMoving(false)
        }, 500)
      }
    },
    [moving, onMoved],
  )

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-700 select-none flex flex-col h-full">
      {/* 상단 상태 바 */}
      <div className="flex-shrink-0 flex items-center justify-between px-3 py-1.5 bg-gray-800 text-xs text-gray-400">
        <span>카메라 뷰 — 클릭 시 스테이지 이동</span>
        {stagePos && (
          <span className="font-mono text-gray-300">
            X {stagePos.x.toFixed(3)} &nbsp;Y {stagePos.y.toFixed(3)} &nbsp;Z {stagePos.z.toFixed(3)} mm
          </span>
        )}
      </div>

      {/* 뷰포트 */}
      <div
        ref={containerRef}
        className="relative flex-1 cursor-crosshair overflow-hidden"
        onClick={handleClick}
      >
        {/* 플레이스홀더 (카메라 미연결 시 표시) */}
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <span className="text-gray-600 text-sm pointer-events-none">카메라 미연결</span>
        </div>

        {/* MJPEG 스트림 */}
        <img
          src="/api/camera/stream"
          className="absolute inset-0 w-full h-full object-contain"
          onError={e => ((e.currentTarget as HTMLImageElement).style.display = 'none')}
          onLoad={e => ((e.currentTarget as HTMLImageElement).style.display = '')}
          alt=""
          draggable={false}
        />

        {/* 중앙 십자선 — 레이저 조사 위치. 뷰포트 정중앙 = 이미지 중심 = 레이저 조사점.
            pointer-events-none 이라 클릭(스테이지 이동)을 가로막지 않는다. */}
        <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
          {/* 중심 링 — 정확한 조사점 */}
          <div
            className="relative w-4 h-4 rounded-full border border-red-500/90 bg-red-500/10"
            style={{ filter: 'drop-shadow(0 0 1px rgba(0,0,0,0.9))' }}
          />
        </div>

        {/* 클릭 인디케이터 */}
        {clickPos && (
          <div
            className="absolute pointer-events-none"
            style={{ left: clickPos.x - 10, top: clickPos.y - 10, width: 20, height: 20 }}
          >
            <div className="w-full h-full border-2 border-yellow-400 rounded-full animate-ping opacity-75" />
          </div>
        )}

        {/* 이동 중 오버레이 */}
        {moving && (
          <div className="absolute inset-0 bg-black/20 flex items-center justify-center pointer-events-none">
            <span className="text-white text-xs bg-black/60 px-3 py-1 rounded">이동 중...</span>
          </div>
        )}
      </div>
    </div>
  )
}
