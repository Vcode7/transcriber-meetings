export interface PromptRule {
  id: string;
  text: string;
  level: number;        // 0=top-level, 1=sub-rule, 2=sub-sub-rule
  enabled: boolean;
  isHeader: boolean;    // Section headers like "CORE EXTRACTION RULES:" or "1. CONCISE, TOPIC-BASED EXTRACTION"
  isStructural: boolean; // JSON templates between ``` markers, variables, blank lines
  children: PromptRule[];
}

export function generateRuleId(): string {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
}

export function createRule(text: string, level: number = 0): PromptRule {
  return {
    id: generateRuleId(),
    text,
    level,
    enabled: true,
    isHeader: false,
    isStructural: false,
    children: []
  };
}

export function createSubRule(text: string, parentLevel: number = 0): PromptRule {
  return createRule(text, parentLevel + 1);
}

/**
 * Parses raw prompt text into a hierarchical rule tree.
 * - Detects section headers (ALL-CAPS ending with colon, numbered section titles).
 * - Detects sub-rules (bullet points, indented items) under their respective headers.
 * - Preserves structural elements (code blocks, variables like {window_text}).
 * - For prompts without hierarchical headers, parses all lines as a flat single-level list.
 */
export function parsePromptToRules(promptText: string): PromptRule[] {
  if (!promptText || !promptText.trim()) return [];

  const rawLines = promptText.split('\n');
  const rules: PromptRule[] = [];

  let inCodeBlock = false;
  let codeBlockLines: string[] = [];

  // Check if the prompt has ANY hierarchical headers at all
  let hasAnyHeaders = false;
  for (const l of rawLines) {
    const trimmed = l.trim();
    if (trimmed.startsWith('```') || (trimmed.includes('{') && trimmed.includes('}'))) continue;
    if (/^[A-Z0-9\s_\-&/]{3,}:$/.test(trimmed) || /^\d+\.\s+[A-Z]/.test(trimmed)) {
      hasAnyHeaders = true;
      break;
    }
  }

  // Active header tracking
  let currentHeader: PromptRule | null = null;

  for (let i = 0; i < rawLines.length; i++) {
    const line = rawLines[i];
    const trimmed = line.trim();

    // 1. Code block handling (e.g. ```json ... ```)
    if (trimmed.startsWith('```')) {
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeBlockLines = [line];
      } else {
        inCodeBlock = false;
        codeBlockLines.push(line);
        const rule = createRule(codeBlockLines.join('\n'), 0);
        rule.isStructural = true;
        rules.push(rule);
        currentHeader = null;
      }
      continue;
    }

    if (inCodeBlock) {
      codeBlockLines.push(line);
      continue;
    }

    // 2. Blank line handling (structural spacer)
    if (trimmed === '') {
      const rule = createRule('', 0);
      rule.isStructural = true;
      rules.push(rule);
      continue;
    }

    // 3. Structural variable placeholders (e.g. "{window_text}")
    if (trimmed.includes('{') && trimmed.includes('}')) {
      const rule = createRule(line, 0);
      rule.isStructural = true;
      rules.push(rule);
      currentHeader = null;
      continue;
    }

    // 4. Label directly preceding a variable (e.g. "TRANSCRIPT WINDOW:" preceding "{window_text}")
    if (trimmed.endsWith(':') && i + 1 < rawLines.length) {
      const nextTrimmed = rawLines[i + 1].trim();
      const nextNextTrimmed = i + 2 < rawLines.length ? rawLines[i + 2].trim() : '';
      if (
        (nextTrimmed.includes('{') && nextTrimmed.includes('}')) ||
        (nextTrimmed === '' && nextNextTrimmed.includes('{') && nextNextTrimmed.includes('}'))
      ) {
        const rule = createRule(line, 0);
        rule.isStructural = true;
        rules.push(rule);
        currentHeader = null;
        continue;
      }
    }

    // If the prompt has NO hierarchical headers at all:
    // Display all rules as a flat single-level list!
    if (!hasAnyHeaders) {
      const rule = createRule(line, 0);
      rules.push(rule);
      continue;
    }

    // 5. Check for ALL-CAPS header (e.g. "WRITING STYLE:", "CORE EXTRACTION RULES:")
    const isAllCapsHeader = /^[A-Z0-9\s_\-&/]{3,}:$/.test(trimmed);
    if (isAllCapsHeader) {
      const rule = createRule(line, 0);
      rule.isHeader = true;
      rules.push(rule);
      currentHeader = rule;
      continue;
    }

    // 6. Check for Numbered Header (e.g. "1. CONCISE, TOPIC-BASED EXTRACTION")
    const isNumberedHeader = /^\d+\.\s+[A-Z]/.test(trimmed);
    if (isNumberedHeader) {
      const rule = createRule(line, 0);
      rule.isHeader = true;
      rules.push(rule);
      currentHeader = rule;
      continue;
    }

    // 7. Check for Sub-rule under an active header (bullet lines: *, -, •, or indented by 2+ spaces)
    const isBulletOrIndent = /^(\s*[-*•]|\s{2,})/.test(line);
    if (isBulletOrIndent && currentHeader) {
      const childRule = createSubRule(line, currentHeader.level);
      currentHeader.children.push(childRule);
      continue;
    }

    // 8. Regular top-level line (e.g. introductory sentences, instructions)
    const rule = createRule(line, 0);
    rules.push(rule);
    currentHeader = null;
  }

  return rules;
}

/**
 * Assembles the final prompt text from the rule tree:
 * - Completely EXCLUDES any disabled rule (`enabled === false`).
 * - If a header is disabled, completely EXCLUDES the entire section (header + all sub-rules).
 * - If a sub-rule is disabled, completely EXCLUDES that sub-rule.
 * - Preserves exact original ordering of all enabled rules and sub-rules.
 * - Collapses excessive blank lines.
 */
export function assemblePromptFromRules(rules: PromptRule[]): string {
  const result: string[] = [];

  function processRule(rule: PromptRule) {
    // Structural lines (code blocks, variables, blank spacers)
    if (rule.isStructural) {
      result.push(rule.text);
      return;
    }

    // 1. If rule is disabled, completely exclude it and all its children!
    if (!rule.enabled) {
      return;
    }

    // 2. Include enabled rule
    result.push(rule.text);

    // 3. Process enabled children in original order
    if (rule.children && rule.children.length > 0) {
      for (const child of rule.children) {
        processRule(child);
      }
    }
  }

  for (const rule of rules) {
    processRule(rule);
  }

  // Join lines preserving original order
  const fullText = result.join('\n');

  // Collapse 3+ consecutive newlines into 2 (clean single empty line)
  const cleaned = fullText.replace(/\n{3,}/g, '\n\n');
  return cleaned.trim();
}
