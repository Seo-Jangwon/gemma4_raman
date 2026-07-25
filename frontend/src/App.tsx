import { useState, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import MainContent from './components/MainContent'
import Banner from './components/Banner'
import SystemInitModal from './components/SystemInitModal'
import CCDStatusBar from './components/CCDStatusBar'
import { useChats } from './chatStore'

export type PageId =
  | 'home'
  | 'image-raman'
  | 'autofocus'
  | 'optimization'
  | 'hardware'
  | 'troubleshooting'
  | 'afm'
  | 'afm-image-raman'
  | 'afm-autofocus'
  | 'afm-optimization'
  | 'afm-hardware'
  | 'afm-troubleshooting'

function App() {
  const [initialized, setInitialized] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [bannerVisible, setBannerVisible] = useState(true)
  const [activePage, setActivePage] = useState<PageId>('home')

  // 채팅 기록 상태(localStorage 영속) — 사이드바 목록/새채팅과 MainContent 활성 대화의 단일 출처.
  const { chats, activeId, activeChat, newChat, selectChat, deleteChat, updateActive } = useChats()

  // 새 채팅/기록 선택 시 홈(채팅) 화면으로 전환한다.
  const handleNewChat = useCallback(() => { newChat(); setActivePage('home') }, [newChat])
  const handleSelectChat = useCallback((id: string) => { selectChat(id); setActivePage('home') }, [selectChat])

  // Keep sidebar visible but make it more subtle for AFM pages
  const isAFMPage = activePage.startsWith('afm')
  const shouldShowSidebar = sidebarOpen

  if (!initialized) {
    return <SystemInitModal onComplete={() => setInitialized(true)} />
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* CCD 실시간 온도 상태 바 — 항상 최상단 */}
      <CCDStatusBar />

      {/* 메인 레이아웃 */}
      <div className="flex flex-1 overflow-hidden min-h-0">
        {/* Skip to content link for accessibility */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-raman-500 focus:text-white focus:rounded"
        >
          Skip to content
        </a>

        {/* Sidebar */}
        <Sidebar
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          onPageSelect={(id) => setActivePage(id)}
          chats={chats}
          activeChatId={activeId}
          onNewChat={handleNewChat}
          onSelectChat={handleSelectChat}
          onDeleteChat={deleteChat}
        />

        {/* Main content area */}
        <main
          id="main-content"
          className="flex-1 flex flex-col overflow-hidden"
        >
          {/* key={activeId}: 대화를 바꾸면 MainContent가 새로 마운트되어 그 대화의
              메시지/세션으로 초기화된다(로드 로직 없이 깔끔히 전환). */}
          <MainContent
            key={activeId}
            onMenuClick={() => setSidebarOpen(!sidebarOpen)}
            sidebarOpen={shouldShowSidebar}
            activePage={activePage}
            onPageSelect={setActivePage}
            initialChat={activeChat}
            onPersist={updateActive}
          />
        </main>

        {/* Bottom banner - Hidden for AFM pages */}
        {bannerVisible && !isAFMPage && (
          <Banner onDismiss={() => setBannerVisible(false)} />
        )}
      </div>
    </div>
  )
}

export default App

