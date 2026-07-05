/**
 * Proper Markdown renderer using `marked` + DOMPurify.
 *
 * Usage:
 *   import { renderMarkdown } from '../lib/markdown'
 *   <div dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }} />
 */
import { marked } from 'marked'
import DOMPurify from 'dompurify'

// Configure marked — synchronous, secure defaults
marked.setOptions({
  gfm: true,     // GitHub Flavoured Markdown (tables, strikethrough, task lists)
  breaks: true,  // \n inside paragraphs becomes <br>
})

/**
 * Convert raw Markdown text to sanitized HTML.
 * Returns an empty string for null / undefined / empty input.
 */
export function renderMarkdown(text: string | undefined | null): string {
  if (!text) return ''
  const raw = marked.parse(text) as string
  return DOMPurify.sanitize(raw, {
    // Allow basic formatting tags; block scripts and event handlers
    ALLOWED_TAGS: [
      'h1','h2','h3','h4','h5','h6',
      'p','br','hr',
      'strong','b','em','i','u','s','del','mark',
      'ul','ol','li',
      'blockquote','pre','code',
      'table','thead','tbody','tr','th','td',
      'a','img',
    ],
    ALLOWED_ATTR: ['href','src','alt','title','class','target','rel'],
  })
}
