export interface PromptRule {
  id: string;
  text: string;
  level: number;        // 0=top-level, 1=sub-rule, 2=sub-sub-rule
  enabled: boolean;
  isHeader: boolean;    // Section headers like "CORE EXTRACTION RULES:" or "1. CONCISE, TOPIC-BASED EXTRACTION"
  isStructural: boolean; // JSON templates between ```json markers, variable placeholders
  children: PromptRule[];
}

export function generateRuleId(): string {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : Math.random().toString(36).substring(2, 15);
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

export function parsePromptToRules(promptText: string): PromptRule[] {
  const lines = promptText.split('\n');
  const rules: PromptRule[] = [];
  
  let inCodeBlock = false;
  let codeBlockLines: string[] = [];
  
  let currentL0: PromptRule | null = null;
  let currentL1: PromptRule | null = null;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // Code block check
    if (line.trim().startsWith('```')) {
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeBlockLines = [line];
      } else {
        inCodeBlock = false;
        codeBlockLines.push(line);
        const rule = createRule(codeBlockLines.join('\n'), 0);
        rule.isStructural = true;
        rules.push(rule);
        currentL0 = null;
        currentL1 = null;
      }
      continue;
    }
    
    if (inCodeBlock) {
      codeBlockLines.push(line);
      continue;
    }
    
    // Blank line
    if (line.trim() === '') {
      const rule = createRule('', 0);
      rule.isStructural = true;
      rules.push(rule);
      continue;
    }
    
    // Structural: variables
    if (line.includes('{') && line.includes('}')) {
      const rule = createRule(line, 0);
      rule.isStructural = true;
      rules.push(rule);
      continue;
    }
    
    // Structural: labels before variables (e.g. "previous_context:")
    if (line.trim().endsWith(':') && i + 1 < lines.length && lines[i+1].includes('{') && lines[i+1].includes('}')) {
      const rule = createRule(line, 0);
      rule.isStructural = true;
      rules.push(rule);
      continue;
    }
    
    // ALL CAPS header
    const isAllCapsHeader = /^[A-Z0-9\s_]+:$/.test(line.trim());
    if (isAllCapsHeader) {
      const rule = createRule(line, 0);
      rule.isHeader = true;
      rules.push(rule);
      currentL0 = rule;
      currentL1 = null;
      continue;
    }
    
    // Numbered header (Level 1)
    if (/^\d+\.\s/.test(line.trim())) {
      const rule = createRule(line, 1);
      rule.isHeader = true;
      if (currentL0) {
        currentL0.children.push(rule);
      } else {
        rules.push(rule);
      }
      currentL1 = rule;
      continue;
    }
    
    // Sub-rule (Level 2)
    if (/^\s{3,}[-*]?\s/.test(line) || /^\s{3,}/.test(line)) {
      const rule = createRule(line, 2);
      if (currentL1) {
        currentL1.children.push(rule);
      } else if (currentL0) {
        currentL0.children.push(rule);
      } else {
        rules.push(rule);
      }
      continue;
    }
    
    // Default top-level
    const rule = createRule(line, 0);
    rules.push(rule);
  }
  
  return rules;
}

export function assemblePromptFromRules(rules: PromptRule[]): string {
  const result: string[] = [];
  
  function hasEnabledDescendant(rule: PromptRule): boolean {
    if (rule.enabled) return true;
    return rule.children.some(c => hasEnabledDescendant(c));
  }
  
  function processRule(rule: PromptRule) {
    if (rule.isStructural) {
      result.push(rule.text);
      return;
    }
    
    let shouldInclude = false;
    if (rule.isHeader) {
      shouldInclude = rule.enabled || rule.children.some(c => hasEnabledDescendant(c));
    } else {
      shouldInclude = rule.enabled;
    }
    
    if (shouldInclude) {
      result.push(rule.text);
      for (const child of rule.children) {
        processRule(child);
      }
    }
  }
  
  for (const rule of rules) {
    processRule(rule);
  }
  
  return result.join('\n');
}
