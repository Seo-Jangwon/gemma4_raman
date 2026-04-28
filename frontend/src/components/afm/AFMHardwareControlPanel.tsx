import { useState, useEffect, useCallback } from 'react'
import { Send, Loader2, CheckCircle, XCircle } from 'lucide-react'
import axios from 'axios'
import CameraView from '../raman/CameraView'
import ParameterPanel, { type SpectrumParams } from '../raman/ParameterPanel'

interface HistoryItem {
  command: string
  response: string
  success: boolean
  timestamp: Date
}

const DEFAULT_PARAMS: SpectrumParams = {
  acqMode: 'single',
  exposureTime: 0.2,
  numAccumulations: 1,
  accCycleTime: 0,
  kineticCount: 1,
  kineticCycleTime: 0,
  readMode: 'fvb',
  preampGainIndex: 0,
  shutter: 'auto',
  laserPower: 40,
  targetTemp: -40,
  cosmicrayFilter: false,
  accumType: 'sum',
  rayleighCorr: 532.191,
  stageSpeedX: 5.0,
  stageSpeedY: 5.0,
}

export default function AFMHardwareControlPanel() {
  const [command, setCommand]       = useState('')
  const [isExecuting, setIsExecuting] = useState(false)
  const [history, setHistory]       = useState<HistoryItem[]>([])
  const [params, setParams]         = useState<SpectrumParams>(DEFAULT_PARAMS)
  const [preampGains, setPreampGains] = useState<number[]>([])
  const [stagePos, setStagePos]     = useState<{ x: number; y: number; z: number } | null>(null)
  const [isAcquiring, setIsAcquiring] = useState(false)

  // ── 하드웨어 상태 폴링 (3초 간격) ──
  const fetchHardwareState = useCallback(async () => {
    try {
      const { data } = await axios.get('/api/hardware/state')

      if (data.ccd) {
        const c = data.ccd
        setParams(prev => ({
          ...prev,
          ...(c.exposure_time != null ? { exposureTime:      c.exposure_time }  : {}),
          ...(c.acq_mode      != null ? { acqMode:           c.acq_mode }       : {}),
          ...(c.num_acc       != null ? { numAccumulations:  c.num_acc }        : {}),
          ...(c.num_kin       != null ? { kineticCount:      c.num_kin }        : {}),
          ...(c.preamp_gain_i != null ? { preampGainIndex:   c.preamp_gain_i }  : {}),
          ...(c.temperature   != null ? { targetTemp:        c.temperature }    : {}),
        }))
        if (c.preamp_gains?.length) setPreampGains(c.preamp_gains)
      }

      if (data.laser?.power_pct != null) {
        setParams(prev => ({ ...prev, laserPower: data.laser.power_pct }))
      }

      if (data.stage) setStagePos(data.stage)
    } catch {
      // 하드웨어 미연결 시 무시
    }
  }, [])

  useEffect(() => {
    fetchHardwareState()
    const id = setInterval(fetchHardwareState, 3000)
    return () => clearInterval(id)
  }, [fetchHardwareState])

  // ── AI 채팅 실행 ──
  const handleExecute = async () => {
    if (!command.trim()) return
    setIsExecuting(true)
    const ts = new Date()
    try {
      const { data } = await axios.post('/api/hardware-command', { command })
      setHistory(prev => [
        { command, response: data.message || '실행 완료', success: true, timestamp: ts },
        ...prev,
      ])
      setCommand('')
      // AI가 파라미터를 바꿨을 수 있으므로 즉시 재동기화
      await fetchHardwareState()
    } catch (err: any) {
      setHistory(prev => [
        {
          command,
          response: err.response?.data?.detail || '실행 실패',
          success: false,
          timestamp: ts,
        },
        ...prev,
      ])
    } finally {
      setIsExecuting(false)
    }
  }

  // ── 수동 측정 ──
  const handleAcquire = async (label?: string) => {
    setIsAcquiring(true)
    const ts = new Date()
    const cmdLabel = label ?? `ACQUIRE (${params.acqMode}, ${params.exposureTime}s, ${params.laserPower}%)`
    try {
      const { data } = await axios.post('/api/spectrum/acquire', {
        exposure:            params.exposureTime,
        power:               params.laserPower,
        acq_mode:            params.acqMode,
        num_accumulations:   params.numAccumulations,
        kinetic_count:       params.kineticCount,
        kinetic_cycle_time:  params.kineticCycleTime > 0 ? params.kineticCycleTime : null,
        read_mode:           params.readMode,
        hbin:                1,
        trigger_mode:        'internal',
      })
      const summary =
        data.mode === 'kinetic'
          ? `mode=kinetic, frames=${data.num_frames}, px=${data.frames?.[0]?.length ?? 0}`
          : `mode=${data.mode}, px=${data.length}, max=${data.max_intensity?.toFixed(0) ?? '?'}`
      setHistory(prev => [
        { command: cmdLabel, response: summary, success: true, timestamp: ts },
        ...prev,
      ])
    } catch (err: any) {
      setHistory(prev => [
        {
          command:   cmdLabel,
          response:  err.response?.data?.detail || '측정 실패',
          success:   false,
          timestamp: ts,
        },
        ...prev,
      ])
    } finally {
      setIsAcquiring(false)
    }
  }

  // ── 원점 복귀 ──
  const handleHoming = async () => {
    const ts = new Date()
    try {
      const { data } = await axios.post('/api/hardware-command', {
        command: '스테이지를 원점(X=0, Y=0, Z=0)으로 이동해줘',
      })
      setHistory(prev => [
        { command: 'Homing', response: data.message || '원점 이동 완료', success: true, timestamp: ts },
        ...prev,
      ])
      await fetchHardwareState()
    } catch (err: any) {
      setHistory(prev => [
        {
          command:   'Homing',
          response:  err.response?.data?.detail || '원점 이동 실패',
          success:   false,
          timestamp: ts,
        },
        ...prev,
      ])
    }
  }

  // ── Silicon 레퍼런스 측정 ──
  const handleSilicon = () => handleAcquire('Silicon reference')

  const quickCmds = [
    '레이저 ON',
    '레이저 OFF',
    '현재 위치 확인',
    'CCD 정보 확인',
    '스펙트럼 취득',
  ]

  return (
    <div className="flex gap-4 p-4" style={{ height: 'calc(100vh - 160px)', minHeight: '700px' }}>

      {/* ═══════════════════════════════════════════
          좌측: AI 채팅 (50%)
      ═══════════════════════════════════════════ */}
      <div className="w-1/2 flex flex-col min-h-0 gap-3">
        <h2 className="text-base font-semibold text-gray-800 flex-shrink-0">AI 하드웨어 제어</h2>

        {/* 명령 입력 */}
        <div className="flex gap-2 flex-shrink-0">
          <input
            type="text"
            value={command}
            onChange={e => setCommand(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleExecute()
              }
            }}
            placeholder="자연어로 명령을 입력하세요..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
          />
          <button
            onClick={handleExecute}
            disabled={isExecuting || !command.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center"
          >
            {isExecuting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* 히스토리 (스크롤) */}
        <div className="flex-1 overflow-y-auto space-y-2 min-h-0 pr-1">
          {history.length === 0 ? (
            <div className="flex items-center justify-center h-full text-gray-400 text-sm">
              명령 히스토리가 없습니다
            </div>
          ) : (
            history.map((item, idx) => (
              <div
                key={idx}
                className={`rounded-lg p-3 border text-sm ${
                  item.success
                    ? 'bg-green-50 border-green-200'
                    : 'bg-red-50 border-red-200'
                }`}
              >
                <div className="flex items-start gap-2">
                  {item.success ? (
                    <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-900 break-words">$ {item.command}</p>
                    <p className={`text-xs mt-1 break-words ${item.success ? 'text-green-800' : 'text-red-800'}`}>
                      {item.response}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      {item.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* 빠른 명령 */}
        <div className="flex-shrink-0 pt-2 border-t border-gray-200">
          <div className="flex flex-wrap gap-1.5">
            {quickCmds.map(cmd => (
              <button
                key={cmd}
                onClick={() => setCommand(cmd)}
                className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded hover:bg-gray-200 border border-gray-200 transition-colors"
              >
                {cmd}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ═══════════════════════════════════════════
          우측: 카메라 + 파라미터 패널 (50%)
      ═══════════════════════════════════════════ */}
      <div className="w-1/2 flex flex-col min-h-0 gap-3 overflow-y-auto">
        <CameraView stagePos={stagePos} onMoved={fetchHardwareState} />
        <ParameterPanel
          params={params}
          onParamChange={update => setParams(prev => ({ ...prev, ...update }))}
          availablePreampGains={preampGains}
          isAcquiring={isAcquiring}
          onAcquire={() => handleAcquire()}
          onHoming={handleHoming}
          onSilicon={handleSilicon}
        />
      </div>
    </div>
  )
}
