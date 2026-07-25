import { Clock, Settings, X, Plus, MessageSquare, Trash2, Sparkles } from 'lucide-react'
import type { PageId } from '../App'
import type { Chat } from '../chatStore'
import logoImage from '../logo/logo.png'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  onPageSelect: (id: PageId) => void
  chats: Chat[]
  activeChatId: string
  onNewChat: () => void
  onSelectChat: (id: string) => void
  onDeleteChat: (id: string) => void
}

export default function Sidebar({
  isOpen,
  onClose,
  chats,
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
}: SidebarProps) {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-50
          w-64 bg-white/60 backdrop-blur-md border-r border-gray-200/50
          transform transition-transform duration-200 ease-in-out
          flex flex-col shadow-sm
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
        aria-label="Navigation sidebar"
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img
              src={logoImage}
              alt="Raman-GPT Logo"
              className="w-6 h-6 object-contain"
            />
            <h1 className="text-xl font-semibold text-gray-900">Raman-GPT</h1>
          </div>
          <button
            onClick={onClose}
            className="lg:hidden p-2 hover:bg-gray-200 rounded-lg"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Raman Agent section — 새 채팅 시작 */}
        <div className="p-3 border-b border-gray-200">
          <div className="flex items-center justify-between px-4 py-2">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Raman Agent
            </h2>
            <button
              onClick={() => { onNewChat(); onClose() }}
              className="p-1 text-raman-500 hover:bg-raman-50 rounded focus:ring-2 focus:ring-raman-500"
              aria-label="New chat"
              title="새 채팅"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
          <button
            className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm text-raman-500 hover:bg-raman-50 rounded-lg focus:ring-2 focus:ring-raman-500 font-medium"
            onClick={() => { onNewChat(); onClose() }}
          >
            <Sparkles className="w-4 h-4 text-raman-500" />
            <span>새 채팅</span>
          </button>
        </div>

        {/* Chat history section (was Agents) */}
        <div className="flex-1 overflow-y-auto p-3">
          <h2 className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            채팅 기록
          </h2>
          <nav className="space-y-1" aria-label="채팅 기록">
            {chats.length === 0 && (
              <p className="px-4 py-2 text-xs text-gray-400">아직 대화가 없습니다.</p>
            )}
            {chats.map(chat => {
              const active = chat.id === activeChatId
              return (
                <div
                  key={chat.id}
                  className={`group flex items-center gap-2 pr-1 rounded-lg ${
                    active ? 'bg-raman-50' : 'hover:bg-gray-200'
                  }`}
                >
                  <button
                    className={`flex-1 min-w-0 flex items-center gap-3 px-4 py-2.5 text-left text-sm focus:ring-2 focus:ring-raman-500 rounded-lg ${
                      active ? 'text-raman-600 font-medium' : 'text-gray-800'
                    }`}
                    onClick={() => { onSelectChat(chat.id); onClose() }}
                    title={chat.title}
                  >
                    <MessageSquare className="w-4 h-4 flex-shrink-0 text-gray-400" />
                    <span className="truncate">{chat.title || '새 대화'}</span>
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); onDeleteChat(chat.id) }}
                    className="p-1.5 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 focus:opacity-100 rounded"
                    aria-label="대화 삭제"
                    title="대화 삭제"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              )
            })}
          </nav>
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-gray-200 space-y-1">
          <button
            className="w-full flex items-center gap-3 px-4 py-3 text-left text-sm text-gray-700 hover:bg-gray-200 rounded-lg focus:ring-2 focus:ring-raman-500"
            aria-label="View activity"
          >
            <Clock className="w-5 h-5" />
            Activity
          </button>

          <button
            className="w-full flex items-center gap-3 px-4 py-3 text-left text-sm text-gray-700 hover:bg-gray-200 rounded-lg focus:ring-2 focus:ring-raman-500"
            aria-label="Open settings"
          >
            <Settings className="w-5 h-5" />
            Settings
          </button>
        </div>
      </aside>
    </>
  )
}
