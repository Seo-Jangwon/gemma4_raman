import { useState, useRef, useEffect } from 'react'
import { Plus, Mic, Send, Loader2, X, FileSpreadsheet, AlertCircle } from 'lucide-react'
import axios from 'axios'
import ModelSelector from './ModelSelector'
import ToolsMenu from './ToolsMenu'

/** 업로드가 끝나 에이전트가 참조할 수 있게 된 첨부 파일. */
export interface Attachment {
  fileId: string
  filename: string
  bytes: number
}

/** 첨부 칩 하나의 표시 상태(업로드 중/완료/실패). */
interface PendingFile {
  key: number
  filename: string
  status: 'uploading' | 'done' | 'error'
  fileId?: string
  bytes?: number
  error?: string
}

interface SearchBarProps {
  onSubmit?: (command: string, attachments: Attachment[]) => void
  isLoading?: boolean
}

// 표 형태 데이터만. 논문·프로토콜 문서(pdf/md)는 지식베이스 업로드 담당이라 제외한다.
// backend/upload_store.py 의 ALLOWED_SUFFIXES 와 일치해야 한다.
const ACCEPT = '.csv,.tsv,.txt,.dat,.xlsx,.xls'

function formatBytes(n?: number): string {
  if (!n) return ''
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

export default function SearchBar({ onSubmit, isLoading = false }: SearchBarProps) {
  const [value, setValue] = useState('')
  const [focused, setFocused] = useState(false)
  const [pending, setPending] = useState<PendingFile[]>([])
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '/' && document.activeElement !== inputRef.current) {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // 파일은 '고른 즉시' 업로드한다 — 전송 버튼을 누를 때 올리면 큰 파일에서 채팅이
  // 몇 초씩 멈춘 것처럼 보인다. 업로드가 끝난 것만 file_id 를 얻어 전송에 포함된다.
  const handleFiles = async (fileList: FileList | null) => {
    if (!fileList?.length) return
    const picked = Array.from(fileList)
    const base = Date.now()

    setPending(prev => [
      ...prev,
      ...picked.map((f, i) => ({
        key: base + i, filename: f.name, status: 'uploading' as const,
      })),
    ])

    await Promise.all(picked.map(async (f, i) => {
      const key = base + i
      const form = new FormData()
      form.append('file', f)
      try {
        const { data } = await axios.post('/api/files/upload', form)
        setPending(prev => prev.map(p => p.key === key
          ? { ...p, status: 'done', fileId: data.file_id, bytes: data.bytes }
          : p))
      } catch (err: any) {
        setPending(prev => prev.map(p => p.key === key
          ? { ...p, status: 'error', error: err?.response?.data?.detail ?? '업로드 실패' }
          : p))
      }
    }))
  }

  const removePending = (key: number) =>
    setPending(prev => prev.filter(p => p.key !== key))

  const uploading = pending.some(p => p.status === 'uploading')
  const ready = pending.filter(p => p.status === 'done')
  const canSubmit = (!!value.trim() || ready.length > 0) && !isLoading && !uploading

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!canSubmit) return
    const command = value.trim()
    const attachments: Attachment[] = ready.map(p => ({
      fileId: p.fileId!, filename: p.filename, bytes: p.bytes ?? 0,
    }))
    setValue('')
    setPending([])
    onSubmit?.(command, attachments)
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div
        className={`
          relative flex flex-col gap-2 p-4
          bg-white border-2 rounded-3xl
          transition-all duration-200
          ${focused
            ? 'border-raman-500 shadow-lg shadow-raman-100'
            : 'border-gray-300 hover:border-gray-400 shadow-sm'
          }
        `}
      >
        {/* 첨부 칩 — 업로드 중/완료/실패를 한눈에 */}
        {pending.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {pending.map(p => (
              <span
                key={p.key}
                className={`inline-flex items-center gap-1.5 pl-2 pr-1 py-1 rounded-lg text-xs border ${
                  p.status === 'error'
                    ? 'bg-red-50 border-red-200 text-red-700'
                    : 'bg-gray-50 border-gray-200 text-gray-700'
                }`}
                title={p.status === 'error' ? p.error : p.fileId}
              >
                {p.status === 'uploading'
                  ? <Loader2 className="w-3.5 h-3.5 animate-spin text-gray-400" />
                  : p.status === 'error'
                  ? <AlertCircle className="w-3.5 h-3.5" />
                  : <FileSpreadsheet className="w-3.5 h-3.5 text-raman-500" />}
                <span className="max-w-[180px] truncate">{p.filename}</span>
                {p.status === 'done' && (
                  <span className="text-gray-400">{formatBytes(p.bytes)}</span>
                )}
                <button
                  type="button"
                  onClick={() => removePending(p.key)}
                  className="p-0.5 hover:bg-gray-200 rounded"
                  aria-label={`${p.filename} 첨부 취소`}
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))}
          </div>
        )}

        <div className="flex items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPT}
            multiple
            className="hidden"
            onChange={e => { handleFiles(e.target.files); e.target.value = '' }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="p-2 hover:bg-gray-100 rounded-full flex-shrink-0"
            aria-label="Attach file"
            title="데이터 파일 첨부 (csv, excel, txt)"
          >
            <Plus className="w-5 h-5 text-gray-600" />
          </button>

          <textarea
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder="Ask Raman-GPT anything..."
            className="flex-1 resize-none outline-none text-gray-900 placeholder-gray-500 min-h-[24px] max-h-[200px]"
            rows={1}
            autoComplete="off"
            autoCorrect="off"
            spellCheck={false}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSubmit(e as unknown as React.FormEvent)
              }
            }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement
              target.style.height = 'auto'
              target.style.height = target.scrollHeight + 'px'
            }}
          />

          <div className="flex items-center gap-2 flex-shrink-0">
            <ToolsMenu />
            <ModelSelector />

            {value.trim() || ready.length > 0 || isLoading ? (
              <button
                type="submit"
                disabled={!canSubmit}
                className="p-2 bg-raman-500 hover:bg-raman-600 disabled:opacity-60 text-white rounded-full"
                aria-label="Send message"
                title={uploading ? '파일 업로드 중…' : undefined}
              >
                {isLoading || uploading
                  ? <Loader2 className="w-5 h-5 animate-spin" />
                  : <Send className="w-5 h-5" />
                }
              </button>
            ) : (
              <button
                type="button"
                className="p-2 hover:bg-gray-100 rounded-full"
                aria-label="Voice input"
              >
                <Mic className="w-5 h-5 text-gray-600" />
              </button>
            )}
          </div>
        </div>
      </div>
    </form>
  )
}
