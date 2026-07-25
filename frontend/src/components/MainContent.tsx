import { useState, useRef, useEffect, useCallback } from 'react'
import { Menu } from 'lucide-react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import SearchBar, { Attachment } from './SearchBar'
import IconButton from './IconButton'
import AFMDashboard from './afm/AFMDashboard'
import CameraView from './raman/CameraView'
import ParameterPanel, { SpectrumParams } from './raman/ParameterPanel'
import type { PageId } from '../App'
import type { Chat, ChatMessage } from '../chatStore'

// 어시스턴트 답변의 마크다운(**굵게**, *기울임*, 목록, 제목, 코드 등)을 실제 서식으로 렌더링한다.
// @tailwindcss/typography(prose) 플러그인이 없어 요소별 클래스를 직접 매핑한다.
// node 는 DOM 요소로 흘리면 React 경고가 나므로 구조분해로 제거한다.
const MD_COMPONENTS = {
  p:          ({ node, ...p }: any) => <p className="mb-2 last:mb-0" {...p} />,
  ul:         ({ node, ...p }: any) => <ul className="list-disc pl-5 mb-2 last:mb-0 space-y-0.5" {...p} />,
  ol:         ({ node, ...p }: any) => <ol className="list-decimal pl-5 mb-2 last:mb-0 space-y-0.5" {...p} />,
  li:         ({ node, ...p }: any) => <li className="leading-relaxed" {...p} />,
  strong:     ({ node, ...p }: any) => <strong className="font-semibold" {...p} />,
  em:         ({ node, ...p }: any) => <em className="italic" {...p} />,
  h1:         ({ node, ...p }: any) => <h1 className="text-base font-bold mt-1 mb-1.5" {...p} />,
  h2:         ({ node, ...p }: any) => <h2 className="text-sm font-bold mt-1 mb-1.5" {...p} />,
  h3:         ({ node, ...p }: any) => <h3 className="text-sm font-semibold mt-1 mb-1" {...p} />,
  a:          ({ node, ...p }: any) => <a className="text-raman-600 underline" target="_blank" rel="noreferrer" {...p} />,
  code:       ({ node, ...p }: any) => <code className="px-1 py-0.5 rounded bg-black/5 font-mono text-[0.85em]" {...p} />,
  pre:        ({ node, ...p }: any) => <pre className="p-2 rounded bg-black/5 font-mono text-xs overflow-x-auto mb-2" {...p} />,
  blockquote: ({ node, ...p }: any) => <blockquote className="border-l-2 border-gray-300 pl-2 text-gray-600" {...p} />,
  hr:         ({ node, ...p }: any) => <hr className="my-2 border-gray-200" {...p} />,
}

// 수식 렌더링 플러그인. remark-math 로 $…$ / $$…$$ 를 수식 노드로 파싱하고
// rehype-katex 로 KaTeX 렌더링한다(스타일은 위의 katex.min.css).
const MD_REMARK_PLUGINS = [remarkMath]
const MD_REHYPE_PLUGINS = [rehypeKatex]

// LaTeX 구분자 정규화 — remark-math 는 $…$ / $$…$$ 만 인식하므로, 모델이 흔히 쓰는
// \[ … \] → $$ … $$, \( … \) → $ … $ 로 바꿔 수식이 깨지지 않게 한다.
function normalizeMath(src: string): string {
  return src
    .replace(/\\\[([\s\S]+?)\\\]/g, (_m, e) => `$$${e}$$`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_m, e) => `$${e}$`)
}

interface MainContentProps {
  onMenuClick: () => void
  sidebarOpen: boolean
  activePage: PageId
  onPageSelect: (id: PageId) => void
  // 채팅 기록 연동 — App(useChats)이 활성 대화를 넘겨주고, 변경분을 다시 올려 영속화한다.
  // App에서 key={activeId}로 렌더하므로 대화 전환 시 이 컴포넌트가 새로 마운트되어
  // 아래 초기값이 그 대화의 메시지/세션으로 다시 잡힌다.
  initialChat?: Chat
  onPersist?: (messages: ChatMessage[], sessionId: string) => void
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

// Andor 내부 ro_mode 문자열(예: 'FULL_VERTICAL_BINNING')을 프론트 readMode 값으로.
// 알 수 없는 값이면 undefined를 반환해 라이브 동기화가 select를 깨진 값으로 만들지 않게 한다.
function mapReadMode(v: unknown): SpectrumParams['readMode'] | undefined {
  const m: Record<string, SpectrumParams['readMode']> = {
    full_vertical_binning: 'fvb', fvb: 'fvb',
    single_track: 'single_track',
    img: 'image', image: 'image',
  }
  return m[String(v).toLowerCase()]
}

// 하드웨어 상태 라이브 폴링 주기(ms)와, 사용자가 값을 바꾼 뒤 폴링이 그 값을 다시
// 덮어쓰지 않도록 두는 유예시간(ms). 대부분의 파라미터는 Acquire 시점에 적용되므로,
// 편집 직후 잠깐은 라이브 동기화를 멈춰 사용자가 스테이징한 값을 지켜준다.
const HW_POLL_MS = 1500
const PARAM_EDIT_GRACE_MS = 6000

export default function MainContent({
  onMenuClick,
  sidebarOpen,
  activePage,
  initialChat,
  onPersist,
}: MainContentProps) {
  // 활성 대화의 메시지/세션으로 초기화한다. App이 key={activeId}로 렌더하므로
  // 대화를 바꾸면 새 마운트에서 이 초기값이 그 대화 기준으로 다시 잡힌다.
  const [messages, setMessages] = useState<ChatMessage[]>(() => initialChat?.messages ?? [])
  const [chatLoading, setChatLoading] = useState(false)
  // clarification(되묻기) 대화를 이어가기 위한 세션 id.
  // 빈 문자열이면 새 실험, 값이 있으면 진행 중인 되묻기의 답변을 같은 세션에 이어붙인다.
  const [sessionId, setSessionId] = useState(() => initialChat?.sessionId ?? '')
  const [params, setParams] = useState<SpectrumParams>(DEFAULT_PARAMS)
  const [stagePos, setStagePos] = useState<{ x: number; y: number; z: number } | null>(null)
  const [availableGains, setAvailableGains] = useState<number[]>([])
  const [isAcquiring, setIsAcquiring] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  // 라이브 폴링이 사용자의 편집을 덮어쓰지 않게 하는 가드.
  //  · paramEditingRef  : 파라미터 패널의 어떤 입력에든 포커스가 있는 동안 true
  //  · lastParamEditRef : 마지막 로컬 편집 시각(ms). 이후 PARAM_EDIT_GRACE_MS 동안 동기화 보류
  const paramEditingRef = useRef(false)
  const lastParamEditRef = useRef(0)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, chatLoading])

  // 대화 변경을 채팅 기록에 영속화한다. 스트리밍 중 메시지가 초당 여러 번 바뀌므로
  // 800ms 디바운스로 묶어 localStorage 쓰기 폭주를 막는다(응답이 끝나면 안정되어 저장됨).
  useEffect(() => {
    if (!onPersist) return
    const t = window.setTimeout(() => onPersist(messages, sessionId), 800)
    return () => window.clearTimeout(t)
  }, [messages, sessionId, onPersist])

  const syncHardwareState = useCallback(async () => {
    try {
      const { data } = await axios.get('/api/hardware/state')

      // 편집 폼(params)을 라이브 값으로 덮어써도 되는지 — 포커스 중이거나 방금 편집했으면 보류.
      // 스테이지 좌표(표시 전용)와 프리앰프 게인 목록은 편집 대상이 아니라 항상 갱신한다.
      const holdParams =
        paramEditingRef.current || Date.now() - lastParamEditRef.current < PARAM_EDIT_GRACE_MS

      if (data.ccd) {
        const c = data.ccd
        if (Array.isArray(c.preamp_gains) && c.preamp_gains.length > 0) {
          setAvailableGains(c.preamp_gains)
        }
        if (!holdParams) {
          const rm = c.ro_mode != null ? mapReadMode(c.ro_mode) : undefined
          setParams(prev => ({
            ...prev,
            ...(c.exposure_time != null && { exposureTime: c.exposure_time }),
            ...(c.acq_mode      != null && { acqMode: c.acq_mode }),
            ...(c.num_acc       != null && { numAccumulations: c.num_acc }),
            ...(c.num_kin       != null && { kineticCount: c.num_kin }),
            ...(rm              != null && { readMode: rm }),
            ...(c.preamp_gain_i != null && { preampGainIndex: c.preamp_gain_i }),
            ...(c.shutter       != null && { shutter: c.shutter }),
            ...(c.temperature   != null && { targetTemp: c.temperature }),
          }))
        }
      }

      if (!holdParams && data.laser?.power_pct != null) {
        setParams(prev => ({ ...prev, laserPower: data.laser.power_pct }))
      }

      if (data.stage) {
        const s = data.stage
        if (s.x != null) setStagePos({ x: s.x, y: s.y, z: s.z })
        if (!holdParams && s.velocity) {
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

  // 마운트 시 1회 + 주기 폴링으로 카메라 아래 설정값을 라이브로 반영한다
  // (에이전트가 툴로 CCD/스테이지/레이저 설정을 바꾸면 패널이 곧 따라간다).
  useEffect(() => {
    syncHardwareState()
    const id = window.setInterval(syncHardwareState, HW_POLL_MS)
    return () => window.clearInterval(id)
  }, [syncHardwareState])

  // ── 멀티에이전트 실험 파이프라인 (SSE 스트리밍 + clarification) ──
  // 홈 채팅 입력은 이 핸들러로 간다. /api/experiment/stream 을 열고
  // 서버가 흘려보내는 SSE 이벤트(intent/clarification/node/done/error)를 파싱해
  // 진행상황을 실시간으로 채팅에 반영한다.
  const handleChat = useCallback(async (command: string, attachments: Attachment[] = []) => {
    // 첨부가 있으면 file_id 목록을 메시지 끝에 덧붙인다. 에이전트는 이 줄을 보고
    // list_uploaded_files / inspect_file 로 넘어간다.
    // 덧붙인 줄을 사용자 말풍선에도 그대로 보여준다 — 화면에 보이는 것과 모델에게
    // 실제로 전달된 것이 어긋나면 나중에 로그를 볼 때 원인 추적이 불가능해진다.
    const attachNote = attachments.length
      ? '\n\n[Attached files]\n' +
        attachments.map(a => `- ${a.filename} (file_id: ${a.fileId})`).join('\n')
      : ''
    const text = command || (attachments.length ? 'Analyze the attached file(s).' : '')
    if (!text) return
    const fullMessage = text + attachNote

    setMessages(prev => [...prev, { role: 'user', text: fullMessage }])
    setChatLoading(true)

    const streamId = Date.now()          // 이번 스트림의 진행상황 메시지를 식별하는 키
    let localSid = sessionId             // 서버가 발급/유지하는 세션 id

    // 노드 진행 로그 한 줄을 진행상황 메시지에 누적한다(없으면 새로 만든다).
    const appendStep = (line: string) => {
      setMessages(prev => {
        const idx = prev.findIndex(m => m.id === streamId)
        if (idx === -1) {
          return [...prev, { role: 'assistant', text: '', kind: 'progress', steps: [line], id: streamId }]
        }
        const copy = [...prev]
        copy[idx] = { ...copy[idx], steps: [...(copy[idx].steps || []), line] }
        return copy
      })
    }

    const handleEvent = (type: string, data: any) => {
      if (data?.session_id) localSid = data.session_id
      switch (type) {
        case 'intent':
          // 해석된 의도는 굳이 채팅에 노출하지 않는다(진행 로그로 충분). 필요 시 확장.
          break
        case 'chat':
          // 실험 요청이 아닌 잡담/메타 질문 — 실험 파이프라인 없이 바로 답변.
          // 세션은 유지한다(초기화하면 백엔드가 쌓아둔 대화 이력을 다음 턴에 못 씀 —
          // "내가 이전에 뭐라고 했지?" 같은 질문에 답하려면 같은 session_id를 계속 써야 한다).
          setSessionId(localSid)
          setMessages(prev => [...prev, {
            role: 'assistant', text: data.reply || '',
          }])
          break
        case 'node':
          if (data?.message) appendStep(data.message)
          break
        case 'spectrum':
          // 측정 스펙트럼 이미지 — 채팅에 인라인 렌더 + 다운로드 링크.
          setMessages(prev => [...prev, {
            role: 'assistant', text: data.title || '', kind: 'spectrum',
            imageUrl: data.image_url, csvUrl: data.csv_url, jsonUrl: data.json_url,
            zipUrl: data.zip_url,
          }])
          break
        case 'clarification':
          // 되묻기 — 세션을 유지하고, 다음 사용자 입력을 답변으로 이어붙인다.
          setSessionId(localSid)
          setMessages(prev => [...prev, {
            role: 'assistant', text: data.question || '추가 정보가 필요합니다.', kind: 'clarification',
          }])
          break
        case 'done': {
          // 실험 자체(계획/누적 버퍼)는 끝났지만 세션(대화 이력)은 유지한다 —
          // 백엔드가 다음 턴을 위해 결과를 history에 남겨두므로 같은 session_id로 이어간다.
          setSessionId(localSid)
          const report = data.final_report || data.abort_reason || '실험이 완료되었습니다.'
          setMessages(prev => [...prev, {
            role: 'assistant', text: report, kind: 'report', isError: !!data.abort_reason,
          }])
          syncHardwareState()
          break
        }
        case 'error':
          // 네트워크/서버 오류 — 세션은 유지해 재시도 시 대화 맥락이 끊기지 않게 한다.
          setSessionId(localSid)
          setMessages(prev => [...prev, {
            role: 'assistant', text: data.detail || '실험 실행 오류', isError: true,
          }])
          break
      }
    }

    try {
      const resp = await fetch('/api/experiment/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: fullMessage, session_id: sessionId }),
      })
      if (!resp.ok || !resp.body) throw new Error(`서버 응답 오류 (${resp.status})`)

      // SSE 파싱: "event: X\ndata: Y\n\n" 블록 단위로 끊어 처리.
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const blocks = buffer.split('\n\n')
        buffer = blocks.pop() ?? ''       // 마지막 조각은 미완성일 수 있어 보류
        for (const block of blocks) {
          if (!block.trim()) continue
          let etype = 'message'
          let dataStr = ''
          for (const line of block.split('\n')) {
            if (line.startsWith('event:')) etype = line.slice(6).trim()
            else if (line.startsWith('data:')) dataStr += line.slice(5).trim()
          }
          if (dataStr) {
            try { handleEvent(etype, JSON.parse(dataStr)) } catch { /* 파싱 실패 무시 */ }
          }
        }
      }
    } catch (err: any) {
      // fetch 자체가 실패한 경우(네트워크 등) — 세션은 유지(재시도 시 맥락 보존).
      setSessionId(localSid)
      setMessages(prev => [...prev, {
        role: 'assistant', text: err?.message ?? '실험 요청 실패', isError: true,
      }])
    } finally {
      setChatLoading(false)
    }
  }, [sessionId, syncHardwareState])

  const handleParamChange = useCallback((update: Partial<SpectrumParams>) => {
    // 사용자가 방금 값을 바꿨다 — 잠시 라이브 폴링이 이 값을 되돌리지 않게 유예 타이머를 찍는다.
    lastParamEditRef.current = Date.now()
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
                    key={msg.id ?? i}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] px-3 py-2 rounded-2xl text-sm leading-relaxed ${
                        msg.role === 'user'
                          ? 'bg-raman-500 text-white rounded-br-sm'
                          : msg.isError
                          ? 'bg-red-50 border border-red-200 text-red-700 rounded-bl-sm'
                          : msg.kind === 'clarification'
                          ? 'bg-amber-50 border border-amber-200 text-amber-900 rounded-bl-sm'
                          : msg.kind === 'progress'
                          ? 'bg-gray-50 border border-gray-200 text-gray-600 rounded-bl-sm font-mono text-xs'
                          : 'bg-gray-100 text-gray-800 rounded-bl-sm'
                      }`}
                    >
                      {msg.kind === 'progress'
                        ? (msg.steps || []).map((s, j) => <div key={j}>{s}</div>)
                        : msg.kind === 'spectrum'
                        ? (
                          <div className="space-y-1.5">
                            {msg.text && <div className="font-medium text-gray-700">{msg.text}</div>}
                            {msg.imageUrl && (
                              <a href={msg.imageUrl} target="_blank" rel="noreferrer">
                                <img
                                  src={msg.imageUrl}
                                  alt={msg.text || '스펙트럼'}
                                  className="rounded-lg border border-gray-200 max-w-full"
                                />
                              </a>
                            )}
                            <div className="flex gap-3 text-xs text-raman-600 flex-wrap">
                              {msg.csvUrl && <a href={msg.csvUrl} download className="hover:underline">CSV 다운로드</a>}
                              {msg.imageUrl && <a href={msg.imageUrl} download className="hover:underline">PNG 다운로드</a>}
                              {msg.zipUrl && <a href={msg.zipUrl} download className="hover:underline">전체 다운로드 (zip)</a>}
                            </div>
                          </div>
                        )
                        : msg.role === 'user'
                        ? <span className="whitespace-pre-wrap">{msg.text}</span>
                        : <ReactMarkdown
                            remarkPlugins={MD_REMARK_PLUGINS}
                            rehypePlugins={MD_REHYPE_PLUGINS}
                            components={MD_COMPONENTS}
                          >{normalizeMath(msg.text || '')}</ReactMarkdown>}
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

              {/* 하단: 파라미터 패널 — 입력에 포커스가 있는 동안은 라이브 폴링이 값을
                  덮어쓰지 않도록 편집 플래그를 세운다(캡처 단계로 자식 입력까지 포착). */}
              <div
                className="flex-shrink-0 overflow-y-auto p-2 border-t border-gray-200"
                style={{ maxHeight: '45%' }}
                onFocusCapture={() => { paramEditingRef.current = true }}
                onBlurCapture={() => { paramEditingRef.current = false }}
              >
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
