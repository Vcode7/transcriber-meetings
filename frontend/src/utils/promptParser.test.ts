import { describe, it, expect } from 'vitest';
import {
  parsePromptToRules,
  assemblePromptFromRules,
  createRule,
  PromptRule,
} from './promptParser';

describe('promptParser', () => {
  it('Issue 1: Disabled rules must be completely excluded from the assembled prompt', () => {
    const rawPrompt = `CORE EXTRACTION RULES:
- Point 1
- Point 2
- Point 3`;

    const rules = parsePromptToRules(rawPrompt);
    expect(rules.length).toBeGreaterThan(0);

    // Disable Point 2
    const header = rules.find((r) => r.isHeader);
    expect(header).toBeDefined();
    expect(header!.children.length).toBe(3);

    header!.children[1].enabled = false; // Disable Point 2

    const assembled = assemblePromptFromRules(rules);
    expect(assembled).toContain('Point 1');
    expect(assembled).not.toContain('Point 2');
    expect(assembled).toContain('Point 3');
  });

  it('Issue 2: Disabling a header and its children completely removes the section', () => {
    const rawPrompt = `You are an expert analyst.

CORE EXTRACTION RULES:
- Extract facts
- Be concise

FINAL INSTRUCTIONS:
- Return valid JSON`;

    const rules = parsePromptToRules(rawPrompt);
    const coreHeader = rules.find((r) => r.text === 'CORE EXTRACTION RULES:');
    expect(coreHeader).toBeDefined();

    // Disable header and all its sub-points
    coreHeader!.enabled = false;
    coreHeader!.children.forEach((c) => (c.enabled = false));

    const assembled = assemblePromptFromRules(rules);
    expect(assembled).not.toContain('CORE EXTRACTION RULES:');
    expect(assembled).not.toContain('Extract facts');
    expect(assembled).not.toContain('Be concise');
    expect(assembled).toContain('You are an expert analyst.');
    expect(assembled).toContain('FINAL INSTRUCTIONS:');
    expect(assembled).toContain('Return valid JSON');
  });

  it('Issue 3: Flat prompts without headers are parsed as single-level list', () => {
    const flatPrompt = `First instruction line.
Second instruction line.
Third instruction line.`;

    const rules = parsePromptToRules(flatPrompt);
    expect(rules.length).toBe(3);
    expect(rules.every((r) => !r.isHeader && r.level === 0)).toBe(true);

    // Can disable any line in the flat list
    rules[1].enabled = false;
    const assembled = assemblePromptFromRules(rules);
    expect(assembled).toContain('First instruction line.');
    expect(assembled).not.toContain('Second instruction line.');
    expect(assembled).toContain('Third instruction line.');
  });

  it('Preserves exact original ordering of enabled rules', () => {
    const r1 = createRule('Rule 1', 0);
    const r2 = createRule('Rule 2', 0);
    const r3 = createRule('Rule 3', 0);
    r2.enabled = false;

    const assembled = assemblePromptFromRules([r1, r2, r3]);
    expect(assembled).toBe('Rule 1\nRule 3');
  });
});
