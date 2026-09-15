import React, { useState, useEffect, useMemo } from 'react'
import {
  FlaskConical, Save, Play, Plus, Loader, ChevronDown, ChevronRight, Check,
  AlertTriangle, History, Trash2, Edit2, X, RefreshCw, Layers, Sparkles,
  Info, CornerDownRight, Tag, ArrowDownToLine, Copy
} from 'lucide-react'
import api from '../api/client'
import { toast } from 'sonner'
import {
  PromptRule, parsePromptToRules, assemblePromptFromRules,
  createRule, createSubRule
} from '../utils/promptParser'

interface ArenaHistoryEntry {
  id: string
  history_id?: string
  prompt_key: string
  meeting_id: string
  recording_name: string
  model_name: string
  window_index?: number | null
  created_at: string
  data: any
}

export default function ModelArena() {
  const [ollamaModels, setOllamaModels] = useState<string[]>([])
  const [runningModels, setRunningModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  
  const [promptGroups, setPromptGroups] = useState<any[]>([])
  const [selectedPromptKey, setSelectedPromptKey] = useState('')
  
  const [meetings, setMeetings] = useState<any[]>([])
  const [selectedMeetingId, setSelectedMeetingId] = useState('')
  
  const [rules, setRules] = useState<PromptRule[]>([])
  const [originalTemplate, setOriginalTemplate] = useState('')
  
  const [windows, setWindows] = useState<any[]>([])
  const [selectedWindowIndex, setSelectedWindowIndex] = useState<number | null>(null)
  
  const [stageData, setStageData] = useState<any>(null)
  const [testResult, setTestResult] = useState<any>(null)
  const [testing, setTesting] = useState(false)
  const [loadingData, setLoadingData] = useState(true)
  
  const [drafts, setDrafts] = useState<any[]>([])
  
  // Generation History state
  const [historyEntries, setHistoryEntries] = useState<ArenaHistoryEntry[]>([])
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null)
  const [historyFilter, setHistoryFilter] = useState<'current' | 'all'>('current')
  const [copiedOutput, setCopiedOutput] = useState(false)
  
  // Add Rule state & Insertion positioning
  const [addRuleMode, setAddRuleMode] = useState<'closed' | 'direct' | 'ai'>('closed')
  const [insertionIndex, setInsertionIndex] = useState<number>(0)
  const [targetParentId, setTargetParentId] = useState<string | null>(null)
  const [newRuleType, setNewRuleType] = useState<'rule' | 'subrule' | 'example'>('rule')
  const [newRuleText, setNewRuleText] = useState('')
  const [aiDescription, setAiDescription] = useState('')
  const [generatedRule, setGeneratedRule] = useState('')
  const [generatingRule, setGeneratingRule] = useState(false)
  
  // Draft Save modal state
  const [savingDraft, setSavingDraft] = useState(false)
  const [savingProd, setSavingProd] = useState(false)
  const [showDraftModal, setShowDraftModal] = useState(false)
  const [draftName, setDraftName] = useState('')

  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    'Model Used': true,
    'Meeting / Window': true,
    'Final Prompt Sent': false,
    'Input Context': false,
    'Generated Output': true,
    'Timing': true,
    'Warnings': true
  })

  useEffect(() => {
    const fetchData = async () => {
      setLoadingData(true)
      try {
        const [modelsRes, promptsRes, meetingsRes, draftsRes, historyRes] = await Promise.all([
          api.get('/arena/ollama-models').catch(() => ({ data: { available_models: [], running_models: [] } })),
          api.get('/arena/prompt-groups').catch(() => ({ data: [] })),
          api.get('/arena/meetings').catch(() => ({ data: [] })),
          api.get('/arena/drafts').catch(() => ({ data: [] })),
          api.get('/arena/history').catch(() => ({ data: [] }))
        ])
        
        const allModels = modelsRes.data.available_models || modelsRes.data.models || []
        setOllamaModels(allModels)
        setRunningModels(modelsRes.data.running_models || modelsRes.data.running || [])
        if (allModels.length > 0) setSelectedModel(allModels[0])
        
        const groups = promptsRes.data || []
        setPromptGroups(groups)
        if (groups.length > 0 && groups[0].keys?.length > 0) {
          setSelectedPromptKey(groups[0].keys[0].key)
        }
        
        const meetingList = meetingsRes.data || []
        setMeetings(meetingList)
        if (meetingList.length > 0) setSelectedMeetingId(meetingList[0].id)
        
        setDrafts(draftsRes.data || [])
        setHistoryEntries(historyRes.data || [])
      } catch (err) {
        toast.error('Failed to load Model Arena data')
      } finally {
        setLoadingData(false)
      }
    }
    fetchData()
  }, [])

  useEffect(() => {
    const fetchPrompt = async () => {
      if (!selectedPromptKey) {
        setRules([])
        setOriginalTemplate('')
        return
      }
      try {
        const res = await api.get(`/prompt-templates/${selectedPromptKey}`)
        const template = res.data.template || ''
        setOriginalTemplate(template)
        setRules(parsePromptToRules(template))
      } catch (err) {
        toast.error('Failed to load prompt template')
      }
    }
    fetchPrompt()
  }, [selectedPromptKey])

  const isWindowSpecific = useMemo(() => {
    return [
      'rom_discussion_embedded', 'rom_discussion', 'rom_discussion_no_actions',
      'rom_action_extraction', 'rom_enhance_window', 'mom_action_regen'
    ].includes(selectedPromptKey)
  }, [selectedPromptKey])

  useEffect(() => {
    const fetchMeetingDetails = async () => {
      if (!selectedMeetingId) return
      try {
        if (isWindowSpecific) {
          const wRes = await api.get(`/arena/meetings/${selectedMeetingId}/windows`).catch(() => ({ data: [] }))
          setWindows(wRes.data || [])
          if (wRes.data?.length > 0) setSelectedWindowIndex(0)
          else setSelectedWindowIndex(null)
        }
        const sRes = await api.get(`/arena/meetings/${selectedMeetingId}/stage-data`).catch(() => ({ data: null }))
        setStageData(sRes.data)
      } catch (err) {
        console.error(err)
      }
    }
    fetchMeetingDetails()
  }, [selectedMeetingId, selectedPromptKey, isWindowSpecific])

  // Dependency pre-validation
  const dependencyWarning = useMemo(() => {
    if (!selectedPromptKey || !stageData) return null
    const stage1Keys = ["rom_discussion_embedded", "rom_discussion", "rom_discussion_no_actions", "rom_action_extraction", "stage1_json_repair"]
    const stage2Keys = ["rom_enhance_window", "rom_enhance_all_together", "rom_deduplicate", "rom_polish"]
    const stage3Keys = ["rom_agenda", "rom_agenda_assign_batch", "rom_mom_expansion", "rom_agenda_doc_points"]
    const finalRomKeys = ["rom_version_short", "rom_version_medium", "rom_ai_edit_points", "rom_ai_chat"]
    const momKeys = ["mom", "mom_extract_actions", "mom_merge", "mom_action_dedup", "mom_action_regen"]

    if (stage1Keys.includes(selectedPromptKey)) {
      if (!stageData.has_transcript) {
        return {
          missing: "Meeting Transcript",
          requiredStage: "Stage 0 (Transcription)",
          reason: "Stage 1 discussion extraction requires transcript segments from the meeting to execute."
        }
      }
    } else if (stage2Keys.includes(selectedPromptKey)) {
      if (!stageData.has_stage1_points && !stageData.has_transcript) {
        return {
          missing: "Stage 1 Discussion Points",
          requiredStage: "Stage 1 (Discussion Points Extraction)",
          reason: "Stage 2 point enhancement requires discussion points generated from Stage 1."
        }
      }
    } else if (stage3Keys.includes(selectedPromptKey)) {
      if (!stageData.has_stage2_points && !stageData.has_stage1_points && !stageData.has_transcript) {
        return {
          missing: "Discussion Points (or Transcript)",
          requiredStage: "Stage 2 (Point Enhancement)",
          reason: "Stage 3 agenda generation and mapping requires discussion points from prior stages."
        }
      }
    } else if (finalRomKeys.includes(selectedPromptKey)) {
      if (!stageData.has_stage3_agendas && !stageData.has_final_rom && !stageData.has_stage2_points && !stageData.has_stage1_points && !stageData.has_transcript) {
        return {
          missing: "Discussion Points or Agendas",
          requiredStage: "Stage 1, 2, or 3",
          reason: "Final ROM short/medium rewrites require structured agendas or discussion points."
        }
      }
    } else if (momKeys.includes(selectedPromptKey)) {
      if (!stageData.has_stage2_points && !stageData.has_stage1_points && !stageData.has_transcript) {
        return {
          missing: "Discussion Points or Transcript",
          requiredStage: "Stage 2 or Stage 1",
          reason: "MoM generation requires discussion points or transcript data."
        }
      }
    }
    return null
  }, [selectedPromptKey, stageData])

  // Entries for the currently selected prompt + meeting
  const currentRuns = useMemo(() => {
    return historyEntries.filter(h => h.prompt_key === selectedPromptKey && h.meeting_id === selectedMeetingId)
  }, [historyEntries, selectedPromptKey, selectedMeetingId])

  // Displayed history depending on filter toggle
  const displayedHistory = useMemo(() => {
    return historyFilter === 'current' ? currentRuns : historyEntries
  }, [historyFilter, currentRuns, historyEntries])

  // Map each entry to its sequential run number within its prompt+meeting group (oldest to newest: 1, 2, 3...)
  const runNumbersMap = useMemo(() => {
    const map: Record<string, number> = {}
    const groups: Record<string, ArenaHistoryEntry[]> = {}
    for (const h of historyEntries) {
      const k = `${h.prompt_key}::${h.meeting_id}`
      if (!groups[k]) groups[k] = []
      groups[k].push(h)
    }
    for (const k in groups) {
      const sorted = [...groups[k]].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
      sorted.forEach((item, idx) => {
        map[item.id] = idx + 1
      })
    }
    return map
  }, [historyEntries])

  // Determine active history entry and result to display
  const activeHistoryEntry = useMemo(() => {
    if (selectedHistoryId) {
      const found = historyEntries.find(h => h.id === selectedHistoryId)
      if (found) return found
    }
    if (currentRuns.length > 0) return currentRuns[0]
    return null
  }, [selectedHistoryId, historyEntries, currentRuns])

  const activeResult = useMemo(() => {
    if (activeHistoryEntry) return activeHistoryEntry.data
    return testResult
  }, [activeHistoryEntry, testResult])

  const handleDeleteHistory = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    try {
      await api.delete(`/arena/history/${id}`)
      setHistoryEntries(prev => prev.filter(h => h.id !== id))
      if (selectedHistoryId === id) {
        setSelectedHistoryId(null)
      }
      toast.success('History entry deleted')
    } catch (err) {
      toast.error('Failed to delete history entry')
    }
  }

  const handleClearHistory = async () => {
    if (!window.confirm('Clear all generation history for this prompt & meeting?')) return
    try {
      await api.delete(`/arena/history?prompt_key=${selectedPromptKey}&meeting_id=${selectedMeetingId}`)
      setHistoryEntries(prev => prev.filter(h => !(h.prompt_key === selectedPromptKey && h.meeting_id === selectedMeetingId)))
      setSelectedHistoryId(null)
      toast.success('History cleared for this prompt and meeting')
    } catch (err) {
      toast.error('Failed to clear history')
    }
  }

  const handleTestPrompt = async () => {
    if (!selectedModel) {
      toast.error('Please select an Ollama model')
      return
    }
    if (!selectedPromptKey) {
      toast.error('Please select a prompt template')
      return
    }
    if (!selectedMeetingId) {
      toast.error('Please select a test meeting')
      return
    }
    
    if (dependencyWarning) {
      toast.error(`Cannot test: Missing ${dependencyWarning.missing}. ${dependencyWarning.reason}`)
      return
    }

    // Verify assembled prompt only has enabled rules
    const assembled = assemblePromptFromRules(rules)
    if (!assembled.trim()) {
      toast.error('Cannot test: All rules are disabled or prompt is empty')
      return
    }
    
    setTesting(true)
    
    try {
      const payload = {
        model_name: selectedModel,
        model: selectedModel,
        prompt_key: selectedPromptKey,
        recording_id: selectedMeetingId,
        meeting_id: selectedMeetingId,
        window_index: isWindowSpecific ? selectedWindowIndex : null,
        prompt_text: assembled
      }
      
      const res = await api.post('/arena/test', payload, { timeout: 1800000 })
      if (res.data.success === false) {
        if (res.data.error === 'missing_dependency') {
          toast.error(`Missing Dependency: ${res.data.missing_dependency} (${res.data.reason})`)
        } else {
          toast.error(res.data.error || 'Test failed')
        }
        setTestResult(res.data)
      } else {
        const meetingName = meetings.find(m => m.id === selectedMeetingId)?.filename || res.data.recording_name || 'Meeting'
        const newEntry: ArenaHistoryEntry = {
          id: res.data.id || res.data.history_id || String(Date.now()),
          history_id: res.data.id || res.data.history_id,
          prompt_key: selectedPromptKey,
          meeting_id: selectedMeetingId,
          recording_name: meetingName,
          model_name: selectedModel,
          window_index: isWindowSpecific ? selectedWindowIndex : null,
          created_at: res.data.created_at || new Date().toISOString(),
          data: res.data
        }
        setHistoryEntries(prev => [newEntry, ...prev.filter(h => h.id !== newEntry.id)])
        setSelectedHistoryId(newEntry.id)
        setTestResult(res.data)
        toast.success('Prompt test completed successfully')
      }
    } catch (err: any) {
      toast.error(err.response?.data?.detail || err.message || 'Test execution failed')
    } finally {
      setTesting(false)
    }
  }
  
  const handleGenerateRule = async () => {
    if (!aiDescription.trim()) {
      toast.error('Please describe the rule to generate')
      return
    }
    setGeneratingRule(true)
    try {
      // Strictly uses default model saved in Settings (main pipeline model), ignoring Model Arena selection
      const res = await api.post('/arena/generate-rule', { 
        description: aiDescription,
        context: assemblePromptFromRules(rules)
      }, { timeout: 1800000 })
      setGeneratedRule(res.data.rule || '')
      if (res.data.model_used) {
        toast.info(`Generated using Settings default model: ${res.data.model_used}`)
      }
    } catch (err) {
      toast.error('Failed to generate rule with AI')
    } finally {
      setGeneratingRule(false)
    }
  }

  const handleOpenAddModal = (
    parentId: string | null = null, 
    type: 'rule' | 'subrule' | 'example' = 'rule',
    insertAtIdx: number | null = null
  ) => {
    setTargetParentId(parentId)
    setNewRuleType(type)
    if (insertAtIdx !== null) {
      setInsertionIndex(insertAtIdx)
    } else if (parentId === null) {
      setInsertionIndex(rules.length)
    }
    setNewRuleText('')
    setAiDescription('')
    setGeneratedRule('')
    setAddRuleMode('direct')
  }

  const handleAddDirectRule = () => {
    if (!newRuleText.trim()) return
    let textToAdd = newRuleText.trim()
    if (newRuleType === 'subrule' && !textToAdd.startsWith('-') && !textToAdd.startsWith('*')) {
      textToAdd = `   - ${textToAdd}`
    } else if (newRuleType === 'example') {
      textToAdd = `[Example]\n${textToAdd}`
    }

    if (targetParentId) {
      // Append as sub-rule under the specific parent header
      const appendChild = (nodes: PromptRule[]): PromptRule[] => {
        return nodes.map(node => {
          if (node.id === targetParentId) {
            const childRule = createSubRule(textToAdd, node.level)
            return { ...node, children: [...node.children, childRule] }
          }
          if (node.children.length > 0) {
            return { ...node, children: appendChild(node.children) }
          }
          return node
        })
      }
      setRules(appendChild(rules))
    } else {
      // Insert at the chosen top-level insertion position!
      const rule = createRule(textToAdd, 0)
      const updated = [...rules]
      const pos = Math.min(Math.max(0, insertionIndex), updated.length)
      updated.splice(pos, 0, rule)
      setRules(updated)
    }
    
    setNewRuleText('')
    setAddRuleMode('closed')
    setTargetParentId(null)
    toast.success('Rule added at selected position')
  }
  
  const handleAcceptAiRule = () => {
    if (!generatedRule.trim()) return
    let textToAdd = generatedRule.trim()
    if (newRuleType === 'subrule' && !textToAdd.startsWith('-') && !textToAdd.startsWith('*')) {
      textToAdd = `   - ${textToAdd}`
    }

    if (targetParentId) {
      // Append as sub-rule under the specific parent header
      const appendChild = (nodes: PromptRule[]): PromptRule[] => {
        return nodes.map(node => {
          if (node.id === targetParentId) {
            const childRule = createSubRule(textToAdd, node.level)
            return { ...node, children: [...node.children, childRule] }
          }
          if (node.children.length > 0) {
            return { ...node, children: appendChild(node.children) }
          }
          return node
        })
      }
      setRules(appendChild(rules))
    } else {
      // Insert at the chosen top-level insertion position!
      const rule = createRule(textToAdd, 0)
      const updated = [...rules]
      const pos = Math.min(Math.max(0, insertionIndex), updated.length)
      updated.splice(pos, 0, rule)
      setRules(updated)
    }

    setGeneratedRule('')
    setAiDescription('')
    setAddRuleMode('closed')
    setTargetParentId(null)
    toast.success('AI rule added at selected position')
  }

  const handleSaveDraft = async () => {
    if (!draftName.trim()) return
    setSavingDraft(true)
    try {
      const payload = {
        draft_name: draftName.trim(),
        name: draftName.trim(),
        prompt_key: selectedPromptKey,
        template: assemblePromptFromRules(rules),
        rule_states: {}
      }
      await api.post('/arena/drafts', payload)
      const res = await api.get('/arena/drafts')
      setDrafts(res.data || [])
      toast.success('Draft saved successfully')
      setShowDraftModal(false)
      setDraftName('')
    } catch (err) {
      toast.error('Failed to save draft')
    } finally {
      setSavingDraft(false)
    }
  }

  const handleDeleteDraft = async (e: React.MouseEvent, draftId: string) => {
    e.stopPropagation()
    if (!confirm('Are you sure you want to delete this draft?')) return
    try {
      await api.delete(`/arena/drafts/${draftId}`)
      setDrafts(prev => prev.filter(d => d.id !== draftId))
      toast.success('Draft deleted')
    } catch {
      toast.error('Failed to delete draft')
    }
  }
  
  const handleSaveToProduction = async () => {
    if (!confirm(`Save changes to production for prompt "${selectedPromptKey}"? This will immediately affect all future pipeline runs.`)) return
    setSavingProd(true)
    try {
      const assembled = assemblePromptFromRules(rules)
      await api.put(`/prompt-templates/${selectedPromptKey}`, { template: assembled })
      toast.success('Prompt saved to production successfully')
    } catch (err) {
      toast.error('Failed to save prompt to production')
    } finally {
      setSavingProd(false)
    }
  }
  
  const toggleSection = (name: string) => {
    setExpandedSections(prev => ({ ...prev, [name]: !prev[name] }))
  }

  const updateRuleText = (id: string, newText: string) => {
    const updateInTree = (nodes: PromptRule[]): PromptRule[] => {
      return nodes.map(node => {
        if (node.id === id) return { ...node, text: newText }
        if (node.children.length > 0) return { ...node, children: updateInTree(node.children) }
        return node
      })
    }
    setRules(updateInTree(rules))
  }
  
  /**
   * Toggles rule enable state.
   * Requirement: When a header is unchecked, automatically uncheck all sub-points/sub-rules under that header.
   */
  const toggleRuleEnabled = (id: string) => {
    const setAllDescendants = (children: PromptRule[], state: boolean): PromptRule[] => {
      return children.map(c => ({
        ...c,
        enabled: state,
        children: setAllDescendants(c.children, state)
      }))
    }

    const updateInTree = (nodes: PromptRule[]): PromptRule[] => {
      return nodes.map(node => {
        if (node.id === id) {
          const nextState = !node.enabled
          return {
            ...node,
            enabled: nextState,
            // Header uncheck cascades to all sub-rules; header check cascades re-enable
            children: node.children.length > 0 ? setAllDescendants(node.children, nextState) : node.children
          }
        }
        if (node.children.length > 0) {
          return { ...node, children: updateInTree(node.children) }
        }
        return node
      })
    }
    setRules(updateInTree(rules))
  }

  const deleteRule = (id: string) => {
    const filterTree = (nodes: PromptRule[]): PromptRule[] => {
      return nodes
        .filter(node => node.id !== id)
        .map(node => ({
          ...node,
          children: node.children.length > 0 ? filterTree(node.children) : []
        }))
    }
    setRules(filterTree(rules))
  }

  const renderRule = (rule: PromptRule, isChild = false) => {
    if (rule.isStructural && !rule.text.trim()) {
      return <div key={rule.id} style={{ height: '10px' }} />
    }
    
    return (
      <div key={rule.id} style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginBottom: '6px' }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'flex-start',
          gap: '10px',
          paddingLeft: `${Math.min(rule.level * 24, 72)}px`,
          opacity: rule.enabled ? 1 : 0.4,
          transition: 'opacity 0.2s'
        }}>
          {!rule.isStructural ? (
            <input 
              type="checkbox"
              checked={rule.enabled}
              onChange={() => toggleRuleEnabled(rule.id)}
              style={{ marginTop: '6px', cursor: 'pointer', accentColor: 'hsl(var(--accent))' }}
              title={rule.enabled ? 'Uncheck to exclude from prompt' : 'Check to include in prompt'}
            />
          ) : (
            <Tag size={13} style={{ marginTop: '6px', color: 'hsl(var(--pencil))', flexShrink: 0 }} />
          )}
          
          {rule.isStructural ? (
            <pre style={{ 
              margin: 0, 
              padding: '6px 10px', 
              background: 'hsl(var(--ink) / .05)', 
              borderRadius: '6px',
              fontFamily: 'monospace',
              fontSize: '0.82rem',
              width: '100%',
              whiteSpace: 'pre-wrap',
              color: 'hsl(var(--ink))',
              border: '1px solid hsl(var(--border) / .5)'
            }}>
              {rule.text}
            </pre>
          ) : (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '3px' }}>
              <textarea
                value={rule.text}
                onChange={(e) => updateRuleText(rule.id, e.target.value)}
                style={{
                  width: '100%',
                  border: '1px solid transparent',
                  background: 'transparent',
                  resize: 'vertical',
                  minHeight: rule.isHeader ? '28px' : '36px',
                  fontFamily: 'Inter, sans-serif',
                  fontSize: rule.isHeader ? '0.92rem' : '0.86rem',
                  fontWeight: rule.isHeader ? 700 : 400,
                  color: 'hsl(var(--foreground))',
                  padding: '3px 6px',
                  borderRadius: '4px',
                  outline: 'none',
                  textDecoration: rule.enabled ? 'none' : 'line-through'
                }}
                onFocus={(e) => e.target.style.border = '1px solid hsl(var(--border))'}
                onBlur={(e) => e.target.style.border = '1px solid transparent'}
              />
              
              {/* Context Actions: Sub-rule & Example creation under headers */}
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                {rule.isHeader && (
                  <button
                    onClick={() => handleOpenAddModal(rule.id, 'subrule')}
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      fontSize: '0.74rem', color: 'hsl(var(--accent))',
                      display: 'flex', alignItems: 'center', gap: '3px', padding: 0
                    }}
                    title="Add sub-rule under this section"
                  >
                    <CornerDownRight size={12} /> + Add Sub-rule
                  </button>
                )}
                <button
                  onClick={() => handleOpenAddModal(rule.id, 'example')}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    fontSize: '0.74rem', color: 'hsl(var(--pencil))',
                    display: 'flex', alignItems: 'center', gap: '3px', padding: 0
                  }}
                  title="Add example under this rule"
                >
                  <Plus size={11} /> + Add Example
                </button>
                <button
                  onClick={() => deleteRule(rule.id)}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    fontSize: '0.74rem', color: 'hsl(var(--destructive) / .7)',
                    display: 'flex', alignItems: 'center', gap: '2px', padding: 0,
                    marginLeft: 'auto'
                  }}
                  title="Delete rule"
                >
                  <Trash2 size={11} /> Remove
                </button>
              </div>
            </div>
          )}
        </div>
        
        {/* Child sub-rules: NO insertion positions are shown here per instructions */}
        {rule.children.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {rule.children.map(child => renderRule(child, true))}
          </div>
        )}
      </div>
    )
  }

  if (loadingData) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '400px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
          <Loader className="spin" size={32} style={{ color: 'hsl(var(--accent))' }} />
          <span style={{ fontSize: '0.9rem', color: 'hsl(var(--pencil))' }}>Loading Model Arena…</span>
        </div>
      </div>
    )
  }

  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%', minHeight: 0, overflow: 'hidden' }}>
      <div style={{ padding: '2rem', flex: 1, overflowY: 'auto' }}>
        
        {/* Header */}
        <div style={{ marginBottom: '1.75rem', display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ 
            width: 48, height: 48, borderRadius: '12px', 
            background: 'hsl(var(--accent) / 0.12)', color: 'hsl(var(--accent))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 2px 8px hsl(var(--accent) / 0.2)'
          }}>
            <FlaskConical size={24} />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>Model Arena</h1>
            <p style={{ margin: '2px 0 0 0', fontSize: '0.88rem', color: 'hsl(var(--pencil))' }}>
              Experiment with different Ollama models, modify prompt rules selectively, and test with real pipeline inputs
            </p>
          </div>
        </div>

        {/* Top 3 Configuration Selectors */}
        <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          
          {/* 1. Ollama Model Selector */}
          <div style={{ flex: 1, minWidth: '240px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px', fontSize: '0.85rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
              Ollama Model (for Testing)
              {runningModels.length > 0 && (
                <span style={{ fontSize: '0.72rem', color: 'hsl(140, 60%, 45%)', fontWeight: 500 }}>
                  ({runningModels.length} active)
                </span>
              )}
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                width: '100%', padding: '10px 12px', borderRadius: '8px',
                border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                color: 'hsl(var(--foreground))', fontFamily: 'Inter, sans-serif', fontSize: '0.9rem'
              }}
            >
              {ollamaModels.length === 0 && <option value="">No Ollama models found</option>}
              {ollamaModels.map(m => (
                <option key={m} value={m}>
                  {m} {runningModels.includes(m) ? '🟢 (Running)' : '⚪ (Installed)'}
                </option>
              ))}
            </select>
          </div>
          
          {/* 2. Prompt Selector (Grouped by pipeline stage) */}
          <div style={{ flex: 1, minWidth: '240px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '0.85rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
              Prompt Template (by Stage)
            </label>
            <select
              value={selectedPromptKey}
              onChange={(e) => setSelectedPromptKey(e.target.value)}
              style={{
                width: '100%', padding: '10px 12px', borderRadius: '8px',
                border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                color: 'hsl(var(--foreground))', fontFamily: 'Inter, sans-serif', fontSize: '0.9rem'
              }}
            >
              {promptGroups.map(group => (
                <optgroup key={group.group || group.category} label={group.group || group.category}>
                  {(group.keys || group.prompts || []).map((p: any) => (
                    <option key={p.key} value={p.key}>
                      {p.name ? `${p.name} (${p.key})` : p.key}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>

          {/* 3. Sample Meeting Selector */}
          <div style={{ flex: 1, minWidth: '240px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '0.85rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
              Sample Meeting
            </label>
            <select
              value={selectedMeetingId}
              onChange={(e) => setSelectedMeetingId(e.target.value)}
              style={{
                width: '100%', padding: '10px 12px', borderRadius: '8px',
                border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                color: 'hsl(var(--foreground))', fontFamily: 'Inter, sans-serif', fontSize: '0.9rem'
              }}
            >
              {meetings.length === 0 && <option value="">No completed meetings available</option>}
              {meetings.map(m => (
                <option key={m.id} value={m.id}>
                  {m.filename} {m.has_rom_data ? '✓ (ROM ready)' : m.has_transcript ? '• (Transcript ready)' : ''}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Window Selector (Conditional: only shown when prompt is window-specific) */}
        {isWindowSpecific && (
          <div style={{ marginBottom: '1.5rem', padding: '1rem', background: 'hsl(var(--card))', borderRadius: '12px', border: '1px solid hsl(var(--border))' }}>
            <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.88rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
              <span>Test Window Selector <span style={{ fontWeight: 400, color: 'hsl(var(--pencil))' }}>(Prompt operates on 2-minute sliding windows)</span></span>
              {windows.length > 0 && <span style={{ fontSize: '0.8rem', color: 'hsl(var(--accent))' }}>{windows.length} window{windows.length !== 1 ? 's' : ''} available</span>}
            </label>
            <select
              value={selectedWindowIndex ?? ''}
              onChange={(e) => setSelectedWindowIndex(e.target.value !== '' ? Number(e.target.value) : null)}
              style={{
                width: '100%', padding: '10px 12px', borderRadius: '8px',
                border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                color: 'hsl(var(--foreground))', fontFamily: 'Inter, sans-serif', fontSize: '0.88rem'
              }}
            >
              {windows.length === 0 && <option value="">No transcript windows available</option>}
              {windows.map((w, i) => (
                <option key={w.index ?? i} value={w.index ?? i}>
                  Window {(w.index ?? i) + 1}: {w.start_time !== undefined ? w.start_time.toFixed(1) : (w.start || 0)}s – {w.end_time !== undefined ? w.end_time.toFixed(1) : (w.end || 0)}s ({w.segment_count ?? w.segments?.length ?? 0} segments)
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Dependency Pre-Validation Warning Banner */}
        {dependencyWarning && (
          <div style={{
            marginBottom: '1.5rem', padding: '1rem 1.25rem',
            background: 'hsl(45 90% 50% / .12)', borderRadius: '12px',
            border: '1.5px solid hsl(45 90% 50% / .45)',
            display: 'flex', gap: '14px', alignItems: 'flex-start'
          }}>
            <AlertTriangle size={22} style={{ color: 'hsl(45 90% 40%)', flexShrink: 0, marginTop: '2px' }} />
            <div>
              <h4 style={{ margin: '0 0 4px 0', color: 'hsl(45 90% 30%)', fontSize: '0.95rem', fontWeight: 700 }}>
                Required Dependency Missing: {dependencyWarning.missing}
              </h4>
              <p style={{ margin: '0 0 6px 0', fontSize: '0.85rem', color: 'hsl(var(--ink))' }}>
                <strong>Required Pipeline Output:</strong> {dependencyWarning.requiredStage}
              </p>
              <p style={{ margin: 0, fontSize: '0.82rem', color: 'hsl(var(--pencil))' }}>
                {dependencyWarning.reason} Run this stage in the meeting first before testing this prompt in the Arena.
              </p>
            </div>
          </div>
        )}

        {/* Main Workspace Layout (Editor + Results / Drafts) */}
        <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
          
          {/* Left Column: Prompt Editor */}
          <div style={{ flex: '1.6', minWidth: '420px' }}>
            <div style={{ padding: '1.5rem', background: 'hsl(var(--card))', borderRadius: '12px', border: '1px solid hsl(var(--border))', marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
                <div>
                  <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600, color: 'hsl(var(--foreground))' }}>
                    Prompt Rules Editor
                  </h3>
                  <span style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))' }}>
                    Uncheck headers to remove entire sections, uncheck lines to exclude them completely
                  </span>
                </div>
                <button
                  onClick={() => handleOpenAddModal(null, 'rule', rules.length)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    padding: '6px 12px', borderRadius: '6px',
                    background: 'hsl(var(--accent) / 0.1)', color: 'hsl(var(--accent))',
                    border: '1px solid hsl(var(--accent) / 0.25)', cursor: 'pointer',
                    fontSize: '0.85rem', fontWeight: 600
                  }}
                >
                  <Plus size={16} /> Add Rule
                </button>
              </div>

              {/* Add Rule Panel (Direct / AI Modal with Insertion Position) */}
              {addRuleMode !== 'closed' && (
                <div style={{ marginBottom: '1.5rem', padding: '1.25rem', background: 'hsl(var(--background))', borderRadius: '10px', border: '1.5px solid hsl(var(--accent) / 0.3)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', borderBottom: '1px solid hsl(var(--border))', paddingBottom: '0.5rem' }}>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                      <button
                        onClick={() => setAddRuleMode('direct')}
                        style={{
                          background: 'none', border: 'none', cursor: 'pointer',
                          fontSize: '0.88rem', fontWeight: 600,
                          color: addRuleMode === 'direct' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                          borderBottom: addRuleMode === 'direct' ? '2px solid hsl(var(--accent))' : 'none',
                          paddingBottom: '4px'
                        }}
                      >
                        1. Direct Input
                      </button>
                      <button
                        onClick={() => setAddRuleMode('ai')}
                        style={{
                          background: 'none', border: 'none', cursor: 'pointer',
                          fontSize: '0.88rem', fontWeight: 600,
                          color: addRuleMode === 'ai' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                          borderBottom: addRuleMode === 'ai' ? '2px solid hsl(var(--accent))' : 'none',
                          paddingBottom: '4px', display: 'flex', alignItems: 'center', gap: '4px'
                        }}
                      >
                        <Sparkles size={14} /> 2. Using AI
                      </button>
                    </div>
                    <button
                      onClick={() => { setAddRuleMode('closed'); setTargetParentId(null) }}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))' }}
                    >
                      <X size={16} />
                    </button>
                  </div>

                  {/* Insertion Position Selector (Only for top-level rules) */}
                  {!targetParentId && (
                    <div style={{ marginBottom: '12px', background: 'hsl(var(--card))', padding: '8px 12px', borderRadius: '6px', border: '1px solid hsl(var(--border))' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px', fontSize: '0.8rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
                        <ArrowDownToLine size={13} /> Insertion Position (between individual rules/headers):
                      </label>
                      <select
                        value={insertionIndex}
                        onChange={(e) => setInsertionIndex(Number(e.target.value))}
                        style={{
                          width: '100%', padding: '6px 8px', borderRadius: '4px',
                          border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                          fontSize: '0.82rem', color: 'hsl(var(--foreground))'
                        }}
                      >
                        <option value={0}>At the very beginning (Top)</option>
                        {rules.map((r, idx) => {
                          if (r.isStructural && !r.text.trim()) return null
                          const preview = r.text.length > 50 ? r.text.substring(0, 48) + '…' : r.text
                          return (
                            <option key={r.id} value={idx + 1}>
                              After {r.isHeader ? 'Header' : 'Rule'} #{idx + 1}: {preview || '(Structural)'}
                            </option>
                          )
                        })}
                      </select>
                    </div>
                  )}

                  {addRuleMode === 'direct' ? (
                    <div>
                      <textarea
                        value={newRuleText}
                        onChange={(e) => setNewRuleText(e.target.value)}
                        placeholder={
                          newRuleType === 'subrule' 
                            ? 'Enter sub-point text...' 
                            : newRuleType === 'example' 
                            ? 'Enter example output/rules...' 
                            : 'Write your instruction rule manually here...'
                        }
                        style={{
                          width: '100%', minHeight: '85px', padding: '10px 12px', borderRadius: '6px',
                          border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                          fontFamily: 'Inter, sans-serif', fontSize: '0.9rem', marginBottom: '12px',
                          color: 'hsl(var(--foreground))'
                        }}
                        autoFocus
                      />
                      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                        <button onClick={() => { setAddRuleMode('closed'); setTargetParentId(null) }} style={{ padding: '6px 14px', borderRadius: '6px', border: 'none', background: 'transparent', cursor: 'pointer', color: 'hsl(var(--ink))', fontSize: '0.85rem' }}>Cancel</button>
                        <button onClick={handleAddDirectRule} style={{ padding: '6px 14px', borderRadius: '6px', border: 'none', background: 'hsl(var(--accent))', color: '#fff', cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem' }}>Add Rule</button>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <div style={{ marginBottom: '8px', fontSize: '0.78rem', color: 'hsl(var(--pencil))', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Info size={13} color="hsl(var(--accent))" />
                        <span>AI generation strictly uses the default pipeline model saved in Settings.</span>
                      </div>
                      <textarea
                        value={aiDescription}
                        onChange={(e) => setAiDescription(e.target.value)}
                        placeholder="Describe what rule you want in plain language (e.g. 'Ensure all action items must name an explicit owner and deadline')..."
                        style={{
                          width: '100%', minHeight: '80px', padding: '10px 12px', borderRadius: '6px',
                          border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                          fontFamily: 'Inter, sans-serif', fontSize: '0.9rem', marginBottom: '10px',
                          color: 'hsl(var(--foreground))'
                        }}
                        autoFocus
                      />
                      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginBottom: generatedRule ? '12px' : '0' }}>
                        <button 
                          onClick={handleGenerateRule} 
                          disabled={generatingRule || !aiDescription.trim()} 
                          style={{ 
                            display: 'flex', alignItems: 'center', gap: '6px', 
                            padding: '6px 14px', borderRadius: '6px', border: 'none', 
                            background: 'hsl(var(--accent))', color: '#fff', 
                            cursor: (!generatingRule && aiDescription.trim()) ? 'pointer' : 'not-allowed', 
                            fontWeight: 600, fontSize: '0.85rem' 
                          }}
                        >
                          {generatingRule ? <Loader size={15} className="spin" /> : <Sparkles size={15} />}
                          Generate with AI
                        </button>
                      </div>
                      
                      {generatedRule && (
                        <div style={{ padding: '1rem', background: 'hsl(var(--accent) / 0.05)', borderRadius: '8px', border: '1px solid hsl(var(--accent) / 0.25)', marginTop: '10px' }}>
                          <h4 style={{ margin: '0 0 6px 0', fontSize: '0.82rem', color: 'hsl(var(--accent))', fontWeight: 700 }}>Generated Rule Preview (Editable):</h4>
                          <textarea
                            value={generatedRule}
                            onChange={(e) => setGeneratedRule(e.target.value)}
                            style={{
                              width: '100%', minHeight: '70px', padding: '8px 10px', borderRadius: '6px',
                              border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                              fontFamily: 'Inter, sans-serif', fontSize: '0.88rem', marginBottom: '10px',
                              color: 'hsl(var(--foreground))'
                            }}
                          />
                          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                            <button onClick={() => setGeneratedRule('')} style={{ padding: '4px 12px', borderRadius: '6px', border: 'none', background: 'transparent', cursor: 'pointer', color: 'hsl(var(--pencil))', fontSize: '0.82rem' }}>Discard</button>
                            <button onClick={handleAcceptAiRule} style={{ padding: '4px 14px', borderRadius: '6px', border: 'none', background: 'hsl(var(--accent))', color: '#fff', cursor: 'pointer', fontSize: '0.82rem', fontWeight: 600 }}>Accept &amp; Insert</button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Parsed Rules List with Insertion Dividers */}
              {rules.length === 0 ? (
                <div style={{ padding: '2.5rem', textAlign: 'center', color: 'hsl(var(--pencil))', fontSize: '0.9rem' }}>
                  Select a prompt template to view and edit its structured rules
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  
                  {/* Top-level insertion divider at Position 0 */}
                  <div style={{ display: 'flex', justifyContent: 'center', margin: '4px 0 8px 0' }}>
                    <button
                      onClick={() => handleOpenAddModal(null, 'rule', 0)}
                      style={{
                        background: 'transparent', border: '1px dashed hsl(var(--border))',
                        color: 'hsl(var(--pencil))', borderRadius: '16px', padding: '2px 10px',
                        fontSize: '0.72rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px',
                        opacity: 0.6, transition: 'all 0.15s'
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.borderColor = 'hsl(var(--accent))'; e.currentTarget.style.color = 'hsl(var(--accent))'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.6'; e.currentTarget.style.borderColor = 'hsl(var(--border))'; e.currentTarget.style.color = 'hsl(var(--pencil))'; }}
                      title="Insert rule at the very beginning"
                    >
                      <Plus size={11} /> Insert rule at top
                    </button>
                  </div>

                  {rules.map((rule, idx) => (
                    <React.Fragment key={rule.id}>
                      {renderRule(rule)}
                      
                      {/* Insertion divider between individual top-level rules/headers.
                          NEVER shown inside child sub-rules per user requirements */}
                      {!rule.isStructural && (
                        <div style={{ display: 'flex', justifyContent: 'center', margin: '6px 0' }}>
                          <button
                            onClick={() => handleOpenAddModal(null, 'rule', idx + 1)}
                            style={{
                              background: 'transparent', border: '1px dashed hsl(var(--border))',
                              color: 'hsl(var(--pencil))', borderRadius: '16px', padding: '2px 10px',
                              fontSize: '0.72rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px',
                              opacity: 0.5, transition: 'all 0.15s'
                            }}
                            onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.borderColor = 'hsl(var(--accent))'; e.currentTarget.style.color = 'hsl(var(--accent))'; }}
                            onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.5'; e.currentTarget.style.borderColor = 'hsl(var(--border))'; e.currentTarget.style.color = 'hsl(var(--pencil))'; }}
                            title={`Insert a rule after this section (position ${idx + 2})`}
                          >
                            <Plus size={11} /> Insert rule here
                          </button>
                        </div>
                      )}
                    </React.Fragment>
                  ))}
                </div>
              )}
            </div>

            {/* Test & Save Actions Bar */}
            <div style={{ 
              display: 'flex', alignItems: 'center', justifyContent: 'space-between', 
              padding: '1.25rem', background: 'hsl(var(--card))', borderRadius: '12px', 
              border: '1px solid hsl(var(--border))', gap: '1rem', flexWrap: 'wrap'
            }}>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  onClick={() => setShowDraftModal(true)}
                  disabled={rules.length === 0}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    padding: '9px 16px', borderRadius: '8px',
                    background: 'hsl(var(--ink) / 0.06)', color: 'hsl(var(--ink))',
                    border: '1px solid hsl(var(--border))', 
                    cursor: rules.length ? 'pointer' : 'not-allowed',
                    fontSize: '0.88rem', fontWeight: 600
                  }}
                  title="Save prompt modifications as a draft"
                >
                  <Save size={16} /> Save as Draft
                </button>
                <button
                  onClick={handleSaveToProduction}
                  disabled={rules.length === 0 || savingProd}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    padding: '9px 16px', borderRadius: '8px',
                    background: 'hsl(140, 60%, 42%)', color: '#fff',
                    border: 'none', cursor: rules.length ? 'pointer' : 'not-allowed',
                    fontSize: '0.88rem', fontWeight: 600,
                    boxShadow: '0 2px 6px hsl(140, 60%, 42% / .25)'
                  }}
                  title="Update this prompt in the live pipeline"
                >
                  {savingProd ? <Loader size={16} className="spin" /> : <Check size={16} />} Save to Production
                </button>
              </div>

              {/* Prominent Test Button */}
              <button
                onClick={handleTestPrompt}
                disabled={testing || rules.length === 0 || !!dependencyWarning}
                style={{
                  display: 'flex', alignItems: 'center', gap: '8px',
                  padding: '12px 28px', borderRadius: '8px',
                  background: dependencyWarning ? 'hsl(var(--pencil) / .3)' : 'hsl(var(--accent))',
                  color: '#fff',
                  border: 'none', cursor: (rules.length && !dependencyWarning && !testing) ? 'pointer' : 'not-allowed',
                  fontSize: '0.98rem', fontWeight: 700,
                  boxShadow: dependencyWarning ? 'none' : '0 4px 14px hsl(var(--accent) / 0.35)',
                  transition: 'all 0.2s'
                }}
              >
                {testing ? <Loader size={18} className="spin" /> : <Play size={18} style={{ fill: 'currentColor' }} />}
                {testing ? 'Testing Prompt…' : 'Test Prompt'}
              </button>
            </div>
          </div>

          {/* Right Column: History & Saved Drafts */}
          <div style={{ flex: '1', minWidth: '340px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* Generation History */}
            <div style={{ padding: '1.25rem', background: 'hsl(var(--card))', borderRadius: '12px', border: '1px solid hsl(var(--border))' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <h3 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <History size={16} color="hsl(var(--accent))" /> Generation History
                </h3>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '0.74rem', padding: '1px 6px', borderRadius: '10px', background: 'hsl(var(--accent) / 0.12)', color: 'hsl(var(--accent))', fontWeight: 600 }}>
                    {displayedHistory.length}
                  </span>
                  {currentRuns.length > 0 && historyFilter === 'current' && (
                    <button
                      type="button"
                      onClick={handleClearHistory}
                      style={{ background: 'none', border: 'none', color: 'hsl(var(--destructive) / .7)', cursor: 'pointer', padding: '2px', fontSize: '0.7rem' }}
                      title="Clear history for this prompt & meeting"
                    >
                      Clear
                    </button>
                  )}
                </div>
              </div>

              {/* Filter Tabs: Current Prompt+Meeting vs All */}
              <div style={{ display: 'flex', background: 'hsl(var(--background))', borderRadius: '6px', padding: '2px', marginBottom: '0.75rem', border: '1px solid hsl(var(--border))' }}>
                <button
                  type="button"
                  onClick={() => setHistoryFilter('current')}
                  style={{
                    flex: 1, padding: '4px 6px', borderRadius: '4px', border: 'none',
                    background: historyFilter === 'current' ? 'hsl(var(--card))' : 'transparent',
                    color: historyFilter === 'current' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                    fontSize: '0.74rem', fontWeight: 600, cursor: 'pointer',
                    boxShadow: historyFilter === 'current' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none'
                  }}
                >
                  This Prompt ({currentRuns.length})
                </button>
                <button
                  type="button"
                  onClick={() => setHistoryFilter('all')}
                  style={{
                    flex: 1, padding: '4px 6px', borderRadius: '4px', border: 'none',
                    background: historyFilter === 'all' ? 'hsl(var(--card))' : 'transparent',
                    color: historyFilter === 'all' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                    fontSize: '0.74rem', fontWeight: 600, cursor: 'pointer',
                    boxShadow: historyFilter === 'all' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none'
                  }}
                >
                  All ({historyEntries.length})
                </button>
              </div>

              {displayedHistory.length === 0 ? (
                <div style={{ fontSize: '0.82rem', color: 'hsl(var(--pencil))', fontStyle: 'italic', padding: '6px 0' }}>
                  {historyFilter === 'current' 
                    ? 'No generations yet for this prompt & meeting. Click "Test Prompt" to generate.' 
                    : 'No test generations recorded yet.'}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '240px', overflowY: 'auto' }}>
                  {displayedHistory.map((entry) => {
                    const isSelected = activeHistoryEntry?.id === entry.id
                    const runNum = runNumbersMap[entry.id] || 1
                    const isLatestForGroup = currentRuns[0]?.id === entry.id
                    const timeStr = new Date(entry.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                    const dateStr = new Date(entry.created_at).toLocaleDateString([], { month: 'short', day: 'numeric' })
                    
                    return (
                      <div
                        key={entry.id}
                        onClick={() => setSelectedHistoryId(entry.id)}
                        style={{
                          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                          padding: '8px 10px', borderRadius: '8px', cursor: 'pointer',
                          background: isSelected ? 'hsl(var(--accent) / 0.08)' : 'hsl(var(--background))',
                          border: isSelected ? '1.5px solid hsl(var(--accent))' : '1px solid hsl(var(--border))',
                          transition: 'all 0.15s ease'
                        }}
                      >
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <span style={{ fontSize: '0.84rem', fontWeight: 700, color: isSelected ? 'hsl(var(--accent))' : 'hsl(var(--foreground))' }}>
                              Run #{runNum}
                            </span>
                            {isLatestForGroup && (
                              <span style={{ fontSize: '0.65rem', padding: '1px 5px', borderRadius: '4px', background: 'hsl(140, 60%, 42% / 0.15)', color: 'hsl(140, 60%, 42%)', fontWeight: 700 }}>
                                Latest
                              </span>
                            )}
                            <span style={{ fontSize: '0.72rem', color: 'hsl(var(--pencil))', marginLeft: 'auto' }}>
                              {dateStr} {timeStr}
                            </span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '2px', fontSize: '0.72rem', color: 'hsl(var(--pencil))' }}>
                            <span style={{ fontFamily: 'monospace', fontWeight: 600, color: 'hsl(var(--ink))' }}>
                              {entry.model_name || entry.data?.model || 'model'}
                            </span>
                            <span>•</span>
                            <span>{entry.data?.duration_ms || 0}ms</span>
                            <span>•</span>
                            <span>{entry.data?.tokens_eval || 0} tokens</span>
                          </div>
                          {historyFilter === 'all' && (
                            <div style={{ fontSize: '0.68rem', color: 'hsl(var(--pencil))', marginTop: '2px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {entry.prompt_key} • {entry.recording_name}
                            </div>
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={(e) => handleDeleteHistory(e, entry.id)}
                          style={{ background: 'none', border: 'none', color: 'hsl(var(--destructive) / .6)', cursor: 'pointer', padding: '4px', marginLeft: '6px' }}
                          title="Delete history entry"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {/* Saved Drafts */}
            <div style={{ padding: '1.25rem', background: 'hsl(var(--card))', borderRadius: '12px', border: '1px solid hsl(var(--border))' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <h3 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Save size={16} color="hsl(var(--pencil))" /> Saved Drafts
                </h3>
                <span style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))' }}>{drafts.length}</span>
              </div>

              {drafts.length === 0 ? (
                <div style={{ fontSize: '0.82rem', color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>
                  No draft prompts saved yet. Click "Save as Draft" after modifying rules.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '180px', overflowY: 'auto' }}>
                  {drafts.map((d: any) => (
                    <div key={d.id} style={{ 
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between', 
                      padding: '7px 10px', background: 'hsl(var(--background))', 
                      borderRadius: '6px', border: '1px solid hsl(var(--border))' 
                    }}>
                      <div 
                        style={{ cursor: 'pointer', flex: 1, minWidth: 0 }} 
                        onClick={() => {
                          setSelectedPromptKey(d.prompt_key)
                          setTimeout(() => setRules(parsePromptToRules(d.template)), 300)
                          toast.info(`Loaded draft: ${d.draft_name || d.name}`)
                        }}
                      >
                        <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'hsl(var(--foreground))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {d.draft_name || d.name || 'Untitled Draft'}
                        </div>
                        <div style={{ fontSize: '0.72rem', color: 'hsl(var(--pencil))' }}>
                          {d.prompt_key} • {new Date(d.created_at).toLocaleDateString()}
                        </div>
                      </div>
                      <button 
                        onClick={(e) => handleDeleteDraft(e, d.id)}
                        style={{ background: 'none', border: 'none', color: 'hsl(var(--destructive) / .7)', cursor: 'pointer', padding: '4px' }}
                        title="Delete draft"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* FULL-WIDTH SECTION: Generated Output & Diagnostics */}
        <div style={{
          width: '100%', background: 'hsl(var(--card))', borderRadius: '14px',
          border: activeResult ? '1.5px solid hsl(var(--accent) / 0.35)' : '1px solid hsl(var(--border))',
          boxShadow: '0 4px 20px rgba(0,0,0,0.06)', overflow: 'hidden', marginBottom: '2.5rem'
        }}>
          
          {/* Header Bar */}
          <div style={{
            padding: '1rem 1.5rem', background: 'hsl(var(--ink) / 0.02)',
            borderBottom: '1px solid hsl(var(--border))', display: 'flex',
            alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FlaskConical size={18} style={{ color: 'hsl(var(--accent))' }} />
                <h2 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                  Generated Output &amp; Diagnostics
                </h2>
              </div>

              {activeHistoryEntry && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                  <span style={{
                    fontSize: '0.78rem', fontWeight: 700, padding: '2px 8px', borderRadius: '6px',
                    background: 'hsl(var(--accent) / 0.12)', color: 'hsl(var(--accent))',
                    border: '1px solid hsl(var(--accent) / 0.25)'
                  }}>
                    Run #{runNumbersMap[activeHistoryEntry.id] || 1}
                  </span>
                  <span style={{
                    fontSize: '0.75rem', fontWeight: 600, padding: '2px 8px', borderRadius: '6px',
                    background: activeHistoryEntry.id === currentRuns[0]?.id ? 'hsl(140, 60%, 42% / 0.15)' : 'hsl(var(--muted) / 0.8)',
                    color: activeHistoryEntry.id === currentRuns[0]?.id ? 'hsl(140, 60%, 42%)' : 'hsl(var(--pencil))',
                    border: '1px solid hsl(var(--border))'
                  }}>
                    {activeHistoryEntry.id === currentRuns[0]?.id ? '🟢 Latest Generation' : '📜 Historical Entry'}
                  </span>
                  <span style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))' }}>
                    Generated: {new Date(activeHistoryEntry.created_at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' })}
                  </span>
                </div>
              )}
            </div>

            {/* Quick Action Controls */}
            {activeResult && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <button
                  type="button"
                  onClick={() => {
                    const textToCopy = activeResult.output || ''
                    navigator.clipboard.writeText(textToCopy)
                    setCopiedOutput(true)
                    toast.success('Output copied to clipboard')
                    setTimeout(() => setCopiedOutput(false), 2000)
                  }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '5px',
                    padding: '6px 12px', borderRadius: '6px',
                    background: 'hsl(var(--background))', border: '1px solid hsl(var(--border))',
                    color: 'hsl(var(--ink))', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600
                  }}
                  title="Copy generated output text"
                >
                  {copiedOutput ? <Check size={14} style={{ color: 'hsl(140, 60%, 42%)' }} /> : <Copy size={14} />}
                  {copiedOutput ? 'Copied' : 'Copy Output'}
                </button>

                <button
                  type="button"
                  onClick={() => {
                    const allOpen = Object.values(expandedSections).every(Boolean)
                    const nextState: Record<string, boolean> = {}
                    Object.keys(expandedSections).forEach(k => {
                      nextState[k] = !allOpen
                    })
                    setExpandedSections(nextState)
                  }}
                  style={{
                    padding: '6px 12px', borderRadius: '6px',
                    background: 'hsl(var(--background))', border: '1px solid hsl(var(--border))',
                    color: 'hsl(var(--pencil))', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 500
                  }}
                >
                  {Object.values(expandedSections).every(Boolean) ? 'Collapse All' : 'Expand All'}
                </button>
              </div>
            )}
          </div>

          {/* Section Body */}
          <div style={{ padding: '1.5rem' }}>
            {testing ? (
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                padding: '4rem 2rem', gap: '16px', background: 'hsl(var(--background))', borderRadius: '10px'
              }}>
                <Loader className="spin" size={36} style={{ color: 'hsl(var(--accent))' }} />
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                    Generating Output with {selectedModel}…
                  </div>
                  <div style={{ fontSize: '0.82rem', color: 'hsl(var(--pencil))', marginTop: '4px' }}>
                    Testing prompt '{selectedPromptKey}' on meeting {meetings.find(m => m.id === selectedMeetingId)?.filename || 'selected meeting'}
                  </div>
                </div>
              </div>
            ) : !activeResult ? (
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                padding: '4rem 2rem', gap: '12px', textAlign: 'center'
              }}>
                <FlaskConical size={42} style={{ color: 'hsl(var(--pencil))', opacity: 0.4 }} />
                <div style={{ fontSize: '1rem', fontWeight: 600, color: 'hsl(var(--foreground))' }}>
                  No output generated yet
                </div>
                <div style={{ fontSize: '0.84rem', color: 'hsl(var(--pencil))', maxWidth: '440px' }}>
                  Select an Ollama model, prompt template, and meeting above, configure the rules, then click <strong>Test Prompt</strong>.
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {[
                  {
                    name: 'Generated Output',
                    content: activeResult.output || 'No output generated',
                    pre: true,
                    isMainOutput: true
                  },
                  {
                    name: 'Model Used',
                    content: `${activeResult.model || 'Unknown'} (${activeResult.server_url || 'Ollama Server'})`
                  },
                  {
                    name: 'Meeting / Window',
                    content: `${activeResult.recording_name || 'Meeting'}${activeResult.window_info ? ` • Window ${activeResult.window_info.index + 1} (${activeResult.window_info.start?.toFixed(1)}s – ${activeResult.window_info.end?.toFixed(1)}s)` : ''}`
                  },
                  {
                    name: 'Final Prompt Sent',
                    content: activeResult.final_prompt || activeResult.prompt || '',
                    pre: true
                  },
                  {
                    name: 'Input Context',
                    content: activeResult.input_context || activeResult.context ? JSON.stringify(activeResult.input_context || activeResult.context, null, 2) : 'None',
                    pre: true
                  },
                  {
                    name: 'Timing',
                    content: `${activeResult.duration_ms || 0}ms total duration • ${activeResult.tokens_eval || activeResult.timing?.eval_count || 0} output tokens generated`
                  },
                  {
                    name: 'Warnings',
                    content: activeResult.warnings?.length ? activeResult.warnings.join('\n') : 'No warnings recorded'
                  }
                ].map(section => (
                  <div
                    key={section.name}
                    style={{
                      border: section.isMainOutput ? '1.5px solid hsl(var(--accent) / 0.4)' : '1px solid hsl(var(--border))',
                      borderRadius: '10px', overflow: 'hidden', background: 'hsl(var(--background))'
                    }}
                  >
                    <button
                      type="button"
                      onClick={() => toggleSection(section.name)}
                      style={{
                        width: '100%', padding: '10px 14px', background: section.isMainOutput ? 'hsl(var(--accent) / 0.05)' : 'hsl(var(--background))',
                        border: 'none', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        cursor: 'pointer', fontSize: '0.9rem', fontWeight: 600, color: 'hsl(var(--ink))'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span>{section.name}</span>
                        {section.isMainOutput && (
                          <span style={{ fontSize: '0.7rem', padding: '1px 7px', borderRadius: '4px', background: 'hsl(var(--accent))', color: '#fff', fontWeight: 700 }}>
                            Result
                          </span>
                        )}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {expandedSections[section.name] ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                      </div>
                    </button>

                    {expandedSections[section.name] && (
                      <div style={{ padding: '14px', background: 'hsl(var(--card))', borderTop: '1px solid hsl(var(--border))' }}>
                        {section.pre ? (
                          <pre style={{
                            margin: 0, padding: '14px', background: 'hsl(var(--ink) / 0.04)',
                            borderRadius: '8px', fontSize: section.isMainOutput ? '0.86rem' : '0.8rem',
                            fontFamily: 'JetBrains Mono, Menlo, monospace', whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word', minHeight: section.isMainOutput ? '260px' : 'auto',
                            maxHeight: section.isMainOutput ? '650px' : '380px', overflowY: 'auto',
                            lineHeight: 1.55, color: 'hsl(var(--ink))'
                          }}>
                            {section.content}
                          </pre>
                        ) : (
                          <div style={{ fontSize: '0.88rem', color: 'hsl(var(--foreground))', lineHeight: 1.45 }}>
                            {section.content}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Save Draft Modal */}
      {showDraftModal && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          zIndex: 1000, backdropFilter: 'blur(3px)'
        }}>
          <div style={{
            background: 'hsl(var(--card))', borderRadius: '12px', padding: '1.5rem',
            width: '420px', border: '1px solid hsl(var(--border))',
            boxShadow: '0 8px 32px rgba(0,0,0,0.25)'
          }}>
            <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.1rem', fontWeight: 700 }}>Save Prompt Draft</h3>
            <p style={{ margin: '0 0 1rem 0', fontSize: '0.82rem', color: 'hsl(var(--pencil))' }}>
              Saved drafts can be reloaded at any time for iterative testing.
            </p>
            <input
              type="text"
              value={draftName}
              onChange={(e) => setDraftName(e.target.value)}
              placeholder="e.g. Concise Stage 1 with strict action owners"
              style={{
                width: '100%', padding: '10px 12px', borderRadius: '8px',
                border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                marginBottom: '1.25rem', fontFamily: 'Inter, sans-serif', fontSize: '0.9rem',
                color: 'hsl(var(--foreground))'
              }}
              autoFocus
            />
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button
                onClick={() => setShowDraftModal(false)}
                style={{ padding: '8px 14px', borderRadius: '8px', border: 'none', background: 'transparent', cursor: 'pointer', fontWeight: 500, fontSize: '0.85rem' }}
              >
                Cancel
              </button>
              <button
                onClick={handleSaveDraft}
                disabled={!draftName.trim() || savingDraft}
                style={{
                  padding: '8px 16px', borderRadius: '8px', border: 'none',
                  background: 'hsl(var(--accent))', color: '#fff',
                  cursor: draftName.trim() ? 'pointer' : 'not-allowed',
                  fontWeight: 600, fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '6px'
                }}
              >
                {savingDraft ? <Loader size={15} className="spin" /> : <Save size={15} />} Save Draft
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
