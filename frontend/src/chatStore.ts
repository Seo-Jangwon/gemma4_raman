import { useCallback, useEffect, useState } from 'react'

// 채팅 말풍선 하나. (이전엔 MainContent에 있던 타입을 스토어로 옮겨 공유한다 —
// 스펙트럼 이미지는 base64가 아니라 서버 URL(imageUrl)이라 대화 하나의 크기가 작다.)
export interface ChatMessage {
  role: 'user' | 'assistant'
  text: string
  isError?: boolean
  // 스트리밍 실험용 확장 필드
  id?: number
  kind?: 'progress' | 'clarification' | 'report' | 'spectrum'
  steps?: string[]
  imageUrl?: string
  csvUrl?: string
  jsonUrl?: string
  zipUrl?: string
}

// 저장되는 대화 하나. sessionId 를 함께 들고 있어야 그 대화를 다시 열었을 때
// 백엔드 히스토리(_SESSIONS)에 이어서 되묻기까지 계속할 수 있다.
export interface Chat {
  id: string
  title: string
  sessionId: string
  messages: ChatMessage[]
  updatedAt: number
}

const STORAGE_KEY = 'ramangpt.chats.v1'
// 최대 보존 대화 수 — 오래된 것부터 버린다. localStorage(≈5MB)에 수백 개 텍스트
// 대화는 들어가지만, 폭주를 막는 상한이다. 초과분은 자동 삭제된다.
const MAX_CHATS = 200

function genId(): string {
  // 브라우저 런타임이라 Date.now()/Math.random() 사용 가능(워크플로 스크립트 제약과 무관).
  return `c_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`
}

export function chatTitle(messages: ChatMessage[]): string {
  const firstUser = messages.find(m => m.role === 'user')?.text?.trim()
  if (!firstUser) return '새 대화'
  const oneLine = firstUser.split('\n')[0].trim()
  return oneLine.length > 40 ? oneLine.slice(0, 40) + '…' : oneLine || '새 대화'
}

export function loadChats(): Chat[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    const arr = raw ? JSON.parse(raw) : []
    if (!Array.isArray(arr)) return []
    // 최신순 정렬 보장
    return (arr as Chat[]).sort((a, b) => b.updatedAt - a.updatedAt)
  } catch {
    return []
  }
}

// quota-safe 저장 — 용량 초과(QuotaExceededError)면 가장 오래된 것부터 하나씩 버리며 재시도.
function persist(chats: Chat[]): Chat[] {
  let list = [...chats].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_CHATS)
  while (list.length) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(list))
      return list
    } catch {
      list = list.slice(0, list.length - 1) // 끝(가장 오래된) 하나 버리고 재시도
    }
  }
  try { localStorage.removeItem(STORAGE_KEY) } catch { /* ignore */ }
  return []
}

export interface UseChats {
  chats: Chat[]
  activeId: string
  activeChat: Chat | undefined
  newChat: () => void
  selectChat: (id: string) => void
  deleteChat: (id: string) => void
  /** 활성 대화의 메시지/세션을 갱신(제목은 첫 사용자 메시지로 자동). 빈 대화는 저장하지 않는다. */
  updateActive: (messages: ChatMessage[], sessionId: string) => void
}

/**
 * 채팅 목록 + 활성 대화 상태를 관리하고 localStorage에 영속화한다.
 * App 한 곳에서만 호출해 Sidebar(목록/선택/새채팅)와 MainContent(활성 대화)로 흘려보낸다.
 */
export function useChats(): UseChats {
  const [chats, setChats] = useState<Chat[]>(() => {
    const loaded = loadChats()
    if (loaded.length > 0) return loaded
    // 첫 실행: 빈 대화 하나로 시작
    return [{ id: genId(), title: '새 대화', sessionId: '', messages: [], updatedAt: Date.now() }]
  })
  const [activeId, setActiveId] = useState<string>(() => {
    const loaded = loadChats()
    return loaded[0]?.id ?? ''
  })

  // activeId 가 비어 있으면(로드 결과가 없던 경우) 첫 대화로 맞춘다.
  useEffect(() => {
    if (!activeId && chats[0]) setActiveId(chats[0].id)
  }, [activeId, chats])

  useEffect(() => { persist(chats) }, [chats])

  const newChat = useCallback(() => {
    setChats(prev => {
      // 이미 비어 있는 '새 대화'가 활성/최상단에 있으면 중복 생성하지 않고 그걸 연다.
      const emptyExisting = prev.find(c => c.messages.length === 0)
      if (emptyExisting) {
        setActiveId(emptyExisting.id)
        return prev
      }
      const c: Chat = { id: genId(), title: '새 대화', sessionId: '', messages: [], updatedAt: Date.now() }
      setActiveId(c.id)
      return [c, ...prev]
    })
  }, [])

  const selectChat = useCallback((id: string) => setActiveId(id), [])

  const deleteChat = useCallback((id: string) => {
    setChats(prev => {
      const next = prev.filter(c => c.id !== id)
      if (next.length === 0) {
        const c: Chat = { id: genId(), title: '새 대화', sessionId: '', messages: [], updatedAt: Date.now() }
        setActiveId(c.id)
        return [c]
      }
      setActiveId(cur => (cur === id ? next[0].id : cur))
      return next
    })
  }, [])

  const updateActive = useCallback((messages: ChatMessage[], sessionId: string) => {
    if (messages.length === 0) return // 빈 대화는 목록을 어지럽히지 않게 저장 생략
    setChats(prev => prev.map(c =>
      c.id === activeId
        ? { ...c, messages, sessionId, title: chatTitle(messages), updatedAt: Date.now() }
        : c,
    ))
  }, [activeId])

  const activeChat = chats.find(c => c.id === activeId)
  return { chats, activeId, activeChat, newChat, selectChat, deleteChat, updateActive }
}
