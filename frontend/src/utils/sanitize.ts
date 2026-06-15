/**
 * 输入安全过滤工具
 * 防止XSS和注入攻击
 */

/** HTML 实体转义映射 */
const ENTITY_MAP: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#x27;',
  '/': '&#x2F;',
}

/**
 * 转义HTML特殊字符，防止XSS
 * @param str 原始字符串
 * @returns 转义后的安全字符串
 */
export function escapeHtml(str: string): string {
  return str.replace(/[&<>"'/]/g, (char) => ENTITY_MAP[char] || char)
}

/**
 * 安全截断字符串，防止过长输入
 * @param str 原始字符串
 * @param maxLength 最大长度，默认500
 * @returns 截断后的字符串
 */
export function truncateSafe(str: string, maxLength: number = 500): string {
  if (str.length <= maxLength) return str
  return str.slice(0, maxLength) + '...'
}

/**
 * 移除控制字符和零宽字符
 * @param str 原始字符串
 * @returns 清洗后的字符串
 */
export function sanitizeInput(str: string): string {
  return str
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '') // 移除控制字符
    .replace(/[\u200B-\u200D\uFEFF]/g, '') // 移除零宽字符
    .replace(/[\u0009\u000A\u000D]/g, ' ') // 将制表符、换行符替换为空格
    .trim()
}

/**
 * 验证用户输入是否合法
 * @param input 用户输入
 * @returns 是否合法
 */
export function isValidInput(input: string): boolean {
  if (!input || !input.trim()) return false
  if (input.length > 500) return false
  // 检测明显的注入尝试
  const injectionPatterns = [
    /<script\b/i,
    /javascript:/i,
    /on\w+\s*=\s*["']/i,
    /<iframe\b/i,
    /data:text\/html/i,
    /\bexec\s*\(/i,
    /\beval\s*\(/i,
  ]
  return !injectionPatterns.some((p) => p.test(input))
}