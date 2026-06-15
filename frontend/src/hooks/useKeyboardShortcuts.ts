import { useEffect, useCallback } from 'react'

export interface Shortcut {
  /** 按键组合，如 'ctrl+enter', 'escape', 'ctrl+k' */
  keys: string
  /** 快捷键描述 */
  description: string
  /** 回调函数 */
  handler: () => void
  /** 是否仅在未聚焦输入框时生效 */
  requireNoInput?: boolean
}

/**
 * 键盘快捷键 Hook
 * 注册全局快捷键，支持 Ctrl/Meta 组合键
 */
export function useKeyboardShortcuts(shortcuts: Shortcut[]) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const isInputFocused =
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable

      for (const shortcut of shortcuts) {
        if (shortcut.requireNoInput && isInputFocused) continue

        const parts = shortcut.keys.toLowerCase().split('+')
        const key = parts[parts.length - 1]
        const ctrl = parts.includes('ctrl')
        const meta = parts.includes('meta')
        const shift = parts.includes('shift')
        const alt = parts.includes('alt')

        const ctrlMatch = (ctrl || meta) && (e.ctrlKey || e.metaKey)
        const shiftMatch = shift ? e.shiftKey : !e.shiftKey
        const altMatch = alt ? e.altKey : !e.altKey
        const keyMatch = e.key.toLowerCase() === key

        if (ctrlMatch && shiftMatch && altMatch && keyMatch) {
          e.preventDefault()
          shortcut.handler()
          return
        }
      }
    },
    [shortcuts],
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])
}