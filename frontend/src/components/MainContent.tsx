import { useState, useRef, useEffect, useCallback } from 'react'
import { Menu } from 'lucide-react'
import axios from 'axios'
import SearchBar from './SearchBar'
import IconButton from './IconButton'
import AFMDashboard from './afm/AFMDashboard'
import CameraView from './raman/CameraView'
import ParameterPanel, { SpectrumParams } from './raman/ParameterPanel'
import type { PageId } from '../App'

interface MainContentProps {
  onMenuClick: () => void
  sidebarOpen: boolean
  activePage: PageId
  onPageSelect: (id: PageId) => void
}

interface ChatMessage {
  role: 'user' | 'assistant'
  text: string
  isError?: boolean
}

const DEFAULT_PARAMS: SpectrumParams = {
  acqMode: 'single',
  exposureTime: 1.0,
  numAccumulations: 1,
  accCycleTime: 0,
  kineticCount: 1,
  kineticCycleTime: 0,
  readMode: 'fvb',
  preampGainIndex: 0,
  shutter: 'auto',
  laserPower: 20,
  targetTemp: -40,
  cosmicrayFilter: false,
  accumType: 'sum',
  rayleighCorr: 0,
  stageSpeedX: 5.0,
  stageSpeedY: 5.0,
}

export default function MainContent({
  onMenuClick,
  sidebarOpen,
  activePage,
}: MainContentProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [chatLoading, setChatLoading] = useState(false)
  const [params, setParams] = useState<SpectrumParams>(DEFAULT_PARAMS)
  const [stagePos, setStagePos] = useState<{ x: number; y: number; z: number } | null>(null)
  const [availableGains, setAvailableGains] = useState<number[]>([])
  const [isAcquiring, setIsAcquiring] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, chatLoading])

  const syncHardwareState = useCallback(async () => {
    try {
      const { data } = await axios.get('/api/hardware/state')

      if (data.ccd) {
        const c = data.ccd
        setParams(prev => ({
          ...prev,
          ...(c.exposure_time != null && { exposureTime: c.exposure_time }),
          ...(c.acq_mode      != null && { acqMode: c.acq_mode }),
          ...(c.num_acc       != null && { numAccumulations: c.num_acc }),
          ...(c.num_kin       != null && { kineticCount: c.num_kin }),
          ...(c.ro_mode       != null && { readMode: c.ro_mode }),
          ...(c.preamp_gain_i != null && { preampGainIndex: c.preamp_gain_i }),
        }))
        if (Array.isArray(c.preamp_gains) && c.preamp_gains.length > 0) {
          setAvailableGains(c.preamp_gains)
        }
      }

      if (data.laser?.power_pct != null) {
        setParams(prev => ({ ...prev, laserPower: data.laser.power_pct }))
      }

      if (data.stage) {
        const s = data.stage
        if (s.x != null) setStagePos({ x: s.x, y: s.y, z: s.z })
        if (s.velocity) {
          setParams(prev => ({
            ...prev,
            stageSpeedX: s.velocity.x,
            stageSpeedY: s.velocity.y,
          }))
        }
      }
    } catch {
      // 하드웨어 미연결 상태에서 무시
    }
  }, [])

  useEffect(() => {
    syncHardwareState()
  }, [syncHardwareState])

  const handleChat = useCallback(async (command: string) => {
    setMessages(prev => [...prev, { role: 'user', text: command }])
    setChatLoading(true)
    try {
      const { data } = await axios.post('/api/hardware-command', { command })
      setMessages(prev => [...prev, { role: 'assistant', text: data.message }])
      await syncHardwareState()
    } catch (err: any) {
      const errText = err.response?.data?.detail ?? '명령 전송 실패'
      setMessages(prev => [...prev, { role: 'assistant', text: errText, isError: true }])
    } finally {
      setChatLoading(false)
    }
  }, [syncHardwareState])

  const handleParamChange = useCallback((update: Partial<SpectrumParams>) => {
    setParams(prev => {
      const next = { ...prev, ...update }
      if (update.stageSpeedX != null || update.stageSpeedY != null) {
        axios.post('/api/stage/speed', { x: next.stageSpeedX, y: next.stageSpeedY, z: 5.0 })
          .catch(() => {})
      }
      return next
    })
  }, [])

  const handleAcquire = useCallback(async () => {
    setIsAcquiring(true)
    try {
      await axios.post('/api/spectrum/acquire', {
        exposure:          params.exposureTime,
        power:             params.laserPower,
        acq_mode:          params.acqMode,
        num_accumulations: params.numAccumulations,
        kinetic_count:     params.kineticCount,
        kinetic_cycle_time: params.kineticCycleTime || null,
        read_mode:         params.readMode,
      })
      await syncHardwareState()
    } catch {
      // 오류 무시 (추후 toast 추가 가능)
    } finally {
      setIsAcquiring(false)
    }
  }, [params, syncHardwareState])

  const handleHoming = useCallback(async () => {
    try {
      await axios.post('/api/hardware-command', { command: '스테이지 홈 위치로 이동해줘' })
      await syncHardwareState()
    } catch {}
  }, [syncHardwareState])

  const handleSilicon = useCallback(async () => {
    try {
      await axios.post('/api/hardware-command', { command: '실리콘 기준으로 라만 캘리브레이션해줘' })
    } catch {}
  }, [])

  const getAFMModule = (pageId: PageId): 'overview' | 'image-raman' | 'autofocus' | 'optimization' | 'hardware' | 'troubleshooting' => {
    switch (pageId) {
      case 'image-raman':
      case 'afm-image-raman':    return 'image-raman'
      case 'autofocus':
      case 'afm-autofocus':      return 'autofocus'
      case 'optimization':
      case 'afm-optimization':   return 'optimization'
      case 'hardware':
      case 'afm-hardware':       return 'hardware'
      case 'troubleshooting':
      case 'afm-troubleshooting':return 'troubleshooting'
      case 'afm':                return 'overview'
      default:                   return 'overview'
    }
  }

  const renderPage = () => {
    switch (activePage) {
      case 'image-raman':
      case 'autofocus':
      case 'optimization':
      case 'hardware':
      case 'troubleshooting':
      case 'afm':
      case 'afm-image-raman':
      case 'afm-autofocus':
      case 'afm-optimization':
      case 'afm-hardware':
      case 'afm-troubleshooting':
        return (
          <div className="flex-1 flex flex-col w-full">
            <AFMDashboard initialModule={getAFMModule(activePage)} />
          </div>
        )

      case 'home':
      default:
        return (
          <div className="flex-1 flex overflow-hidden min-h-0">

            {/* 좌열: 채팅 */}
            <div className="w-[45%] flex flex-col border-r border-gray-200 min-h-0">
              {/* 채팅 메시지 영역 */}
              <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
                {messages.length === 0 && (
                  <div className="flex items-center justify-center h-full">
                    <p className="text-sm text-gray-400">Raman-GPT에게 명령을 내려보세요</p>
                  </div>
                )}
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] px-3 py-2 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
                        msg.role === 'user'
                          ? 'bg-raman-500 text-white rounded-br-sm'
                          : msg.isError
                          ? 'bg-red-50 border border-red-200 text-red-700 rounded-bl-sm'
                          : 'bg-gray-100 text-gray-800 rounded-bl-sm'
                      }`}
                    >
                      {msg.text}
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 px-4 py-3 rounded-2xl rounded-bl-sm flex gap-1 items-center">
                      <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                )}
                <div ref={bottomRef} />
              </div>

              {/* 하단 입력창 */}
              <div className="px-3 py-3 border-t border-gray-100">
                <SearchBar onSubmit={handleChat} isLoading={chatLoading} />
              </div>
            </div>

            {/* 우열: 카메라 + 파라미터 */}
            <div className="w-[55%] flex flex-col min-h-0">
              {/* 상단: 카메라 뷰 */}
              <div className="flex-1 min-h-0 p-2 overflow-hidden">
                <CameraView stagePos={stagePos} onMoved={syncHardwareState} />
              </div>

              {/* 하단: 파라미터 패널 */}
              <div className="flex-shrink-0 overflow-y-auto p-2 border-t border-gray-200" style={{ maxHeight: '45%' }}>
                <ParameterPanel
                  params={params}
                  onParamChange={handleParamChange}
                  availablePreampGains={availableGains}
                  isAcquiring={isAcquiring}
                  onAcquire={handleAcquire}
                  onHoming={handleHoming}
                  onSilicon={handleSilicon}
                />
              </div>
            </div>

          </div>
        )
    }
  }

  const isAFMPage = activePage.startsWith('afm')

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <header className={`flex-shrink-0 sticky top-0 z-10 border-b transition-all ${
        isAFMPage
          ? 'bg-white/40 backdrop-blur-md border-gray-200/50'
          : 'bg-white/80 backdrop-blur-sm border-gray-200'
      }`}>
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            {!sidebarOpen && (
              <IconButton
                icon={Menu}
                onClick={onMenuClick}
                label="Open sidebar"
              />
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className={`text-sm hidden sm:inline ${
              isAFMPage ? 'text-gray-700' : 'text-gray-600'
            }`}>
              Olympus BX + RAONSpec
            </span>
            <div className="w-2 h-2 rounded-full bg-green-500" title="System online" />
          </div>
        </div>
      </header>

      {renderPage()}
    </div>
  )
}
