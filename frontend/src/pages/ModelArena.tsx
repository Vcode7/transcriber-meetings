import React, { useState, useEffect } from 'react'
import {
  FlaskConical, Save, Play, Plus, Loader, ChevronDown, ChevronRight, Check,
  AlertTriangle, History, Trash2, Edit2, X, RefreshCw, Layers
} from 'lucide-react'
import api from '../api/client'
import { toast } from 'sonner'
import {
  PromptRule, parsePromptToRules, assemblePromptFromRules,
  createRule, createSubRule
} from '../utils/promptParser'

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
  
  const [addRuleMode, setAddRuleMode] = useState<'closed' | 'direct' | 'ai'>('closed')
  const [newRuleText, setNewRuleText] = useState('')
  const [aiDescription, setAiDescription] = useState('')
  const [generatedRule, setGeneratedRule] = useState('')
  const [generatingRule, setGeneratingRule] = useState(false)
  
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
        const [modelsRes, promptsRes, meetingsRes, draftsRes] = await Promise.all([
          api.get('/arena/ollama-models').catch(() => ({ data: { models: [], running: [] } })),
          api.get('/arena/prompt-groups').catch(() => ({ data: [] })),
          api.get('/arena/meetings').catch(() => ({ data: [] })),
          api.get('/arena/drafts').catch(() => ({ data: [] }))
        ])
        setOllamaModels(modelsRes.data.models || [])
        setRunningModels(modelsRes.data.running || [])
        if (modelsRes.data.models?.length > 0) setSelectedModel(modelsRes.data.models[0])
        
        setPromptGroups(promptsRes.data || [])
        setMeetings(meetingsRes.data || [])
        if (meetingsRes.data?.length > 0) setSelectedMeetingId(meetingsRes.data[0].id)
        
        setDrafts(draftsRes.data || [])
      } catch (err) {
        toast.error('Failed to load arena data')
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

  const isWindowSpecific = ['rom_discussion_embedded', 'rom_discussion', 'rom_discussion_no_actions', 'rom_action_extraction', 'rom_enhance_window'].includes(selectedPromptKey)

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

  const handleTestPrompt = async () => {
    if (!selectedModel || !selectedPromptKey || !selectedMeetingId) {
      toast.error('Please select model, prompt, and meeting')
      return
    }
    
    setTesting(true)
    setTestResult(null)
    
    try {
      const assembled = assemblePromptFromRules(rules)
      const payload = {
        model: selectedModel,
        prompt_key: selectedPromptKey,
        meeting_id: selectedMeetingId,
        window_index: isWindowSpecific ? selectedWindowIndex : null,
        prompt_text: assembled
      }
      
      const res = await api.post('/arena/test', payload)
      setTestResult(res.data)
      toast.success('Test completed')
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Test failed')
    } finally {
      setTesting(false)
    }
  }
  
  const handleGenerateRule = async () => {
    if (!aiDescription.trim()) return
    setGeneratingRule(true)
    try {
      const res = await api.post('/arena/generate-rule', { description: aiDescription })
      setGeneratedRule(res.data.rule || '')
    } catch (err) {
      toast.error('Failed to generate rule')
    } finally {
      setGeneratingRule(false)
    }
  }

  const handleAddDirectRule = () => {
    if (!newRuleText.trim()) return
    const rule = createRule(newRuleText, 0)
    setRules([...rules, rule])
    setNewRuleText('')
    setAddRuleMode('closed')
  }
  
  const handleAcceptAiRule = () => {
    if (!generatedRule.trim()) return
    const rule = createRule(generatedRule, 0)
    setRules([...rules, rule])
    setGeneratedRule('')
    setAiDescription('')
    setAddRuleMode('closed')
  }

  const handleSaveDraft = async () => {
    if (!draftName.trim()) return
    setSavingDraft(true)
    try {
      const payload = {
        name: draftName,
        prompt_key: selectedPromptKey,
        template: assemblePromptFromRules(rules)
      }
      await api.post('/arena/drafts', payload)
      const res = await api.get('/arena/drafts')
      setDrafts(res.data)
      toast.success('Draft saved')
      setShowDraftModal(false)
      setDraftName('')
    } catch (err) {
      toast.error('Failed to save draft')
    } finally {
      setSavingDraft(false)
    }
  }
  
  const handleSaveToProduction = async () => {
    if (!confirm('Are you sure you want to save this prompt to production?')) return
    setSavingProd(true)
    try {
      const assembled = assemblePromptFromRules(rules)
      await api.put(`/prompt-templates/${selectedPromptKey}`, { template: assembled })
      toast.success('Saved to production')
    } catch (err) {
      toast.error('Failed to save to production')
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
  
  const toggleRuleEnabled = (id: string) => {
    const updateInTree = (nodes: PromptRule[]): PromptRule[] => {
      return nodes.map(node => {
        if (node.id === id) return { ...node, enabled: !node.enabled }
        if (node.children.length > 0) return { ...node, children: updateInTree(node.children) }
        return node
      })
    }
    setRules(updateInTree(rules))
  }

  const renderRule = (rule: PromptRule) => {
    if (rule.isStructural && !rule.text.trim()) {
      return <div key={rule.id} style={{ height: '16px' }} />
    }
    
    return (
      <div key={rule.id} style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginBottom: '8px' }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'flex-start',
          gap: '12px',
          paddingLeft: `${rule.level * 24}px`,
          opacity: rule.enabled ? 1 : 0.5
        }}>
          {!rule.isStructural && (
            <input 
              type="checkbox"
              checked={rule.enabled}
              onChange={() => toggleRuleEnabled(rule.id)}
              style={{ marginTop: '4px', cursor: 'pointer' }}
            />
          )}
          
          {rule.isStructural ? (
            <pre style={{ 
              margin: 0, 
              padding: '8px 12px', 
              background: 'hsl(var(--ink) / .05)', 
              borderRadius: '6px',
              fontFamily: 'monospace',
              fontSize: '0.85rem',
              width: '100%',
              whiteSpace: 'pre-wrap',
              color: 'hsl(var(--ink))'
            }}>
              {rule.text}
            </pre>
          ) : (
            <textarea
              value={rule.text}
              onChange={(e) => updateRuleText(rule.id, e.target.value)}
              style={{
                flex: 1,
                border: '1px solid transparent',
                background: 'transparent',
                resize: 'vertical',
                minHeight: rule.isHeader ? '30px' : '40px',
                fontFamily: 'Inter, sans-serif',
                fontSize: rule.isHeader ? '0.95rem' : '0.9rem',
                fontWeight: rule.isHeader ? 700 : 400,
                color: 'hsl(var(--foreground))',
                padding: '4px 8px',
                borderRadius: '4px',
                outline: 'none',
              }}
              onFocus={(e) => e.target.style.border = '1px solid hsl(var(--border))'}
              onBlur={(e) => e.target.style.border = '1px solid transparent'}
            />
          )}
        </div>
        
        {rule.children.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {rule.children.map(child => renderRule(child))}
          </div>
        )}
      </div>
    )
  }

  if (loadingData) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <Loader className="spin" size={32} style={{ color: 'hsl(var(--accent))' }} />
      </div>
    )
  }

  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%', minHeight: 0, overflow: 'hidden' }}>
      <div style={{ padding: '2rem', flex: 1, overflowY: 'auto' }}>
        
        {/* Header */}
        <div style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ 
            width: 48, height: 48, borderRadius: '12px', 
            background: 'hsl(var(--accent) / 0.1)', color: 'hsl(var(--accent))',
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <FlaskConical size={24} />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>Model Arena</h1>
            <p style={{ margin: 0, fontSize: '0.9rem', color: 'hsl(var(--pencil))' }}>
              Experiment with different Ollama models and selectively modify prompt rules
            </p>
          </div>
        </div>

        {/* Configuration Row */}
        <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: '250px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '0.85rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
              Ollama Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                width: '100%', padding: '10px 12px', borderRadius: '8px',
                border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                color: 'hsl(var(--foreground))', fontFamily: 'Inter, sans-serif'
              }}
            >
              <option value="">Select model...</option>
              {ollamaModels.map(m => (
                <option key={m} value={m}>{m} {runningModels.includes(m) ? '(Running)' : ''}</option>
              ))}
            </select>
          </div>
          
          <div style={{ flex: 1, minWidth: '250px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '0.85rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
              Prompt Template
            </label>
            <select
              value={selectedPromptKey}
              onChange={(e) => setSelectedPromptKey(e.target.value)}
              style={{
                width: '100%', padding: '10px 12px', borderRadius: '8px',
                border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                color: 'hsl(var(--foreground))', fontFamily: 'Inter, sans-serif'
              }}
            >
              <option value="">Select prompt...</option>
              {promptGroups.map(group => (
                <optgroup key={group.category} label={group.category}>
                  {group.prompts.map((p: any) => (
                    <option key={p.key} value={p.key}>{p.key}</option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>

          <div style={{ flex: 1, minWidth: '250px' }}>
            <label style={{ display: 'block', marginBottom: '6px', fontSize: '0.85rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
              Test Meeting
            </label>
            <select
              value={selectedMeetingId}
              onChange={(e) => setSelectedMeetingId(e.target.value)}
              style={{
                width: '100%', padding: '10px 12px', borderRadius: '8px',
                border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                color: 'hsl(var(--foreground))', fontFamily: 'Inter, sans-serif'
              }}
            >
              <option value="">Select meeting...</option>
              {meetings.map(m => (
                <option key={m.id} value={m.id}>{m.filename}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Window Selector (Conditional) */}
        {isWindowSpecific && (
          <div style={{ marginBottom: '2rem', padding: '1rem', background: 'hsl(var(--card))', borderRadius: '12px', border: '1px solid hsl(var(--border))' }}>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
              Select Window for Test
            </label>
            <select
              value={selectedWindowIndex ?? ''}
              onChange={(e) => setSelectedWindowIndex(e.target.value ? Number(e.target.value) : null)}
              style={{
                width: '100%', padding: '10px 12px', borderRadius: '8px',
                border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                color: 'hsl(var(--foreground))', fontFamily: 'Inter, sans-serif'
              }}
            >
              <option value="">Select window...</option>
              {windows.map((w, i) => (
                <option key={i} value={i}>
                  Window {i + 1}: {w.start}s – {w.end}s ({w.segments?.length || 0} segments)
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Dependency Warnings */}
        {stageData?.warnings && stageData.warnings.length > 0 && (
          <div style={{ marginBottom: '2rem', padding: '1rem', background: 'hsl(var(--destructive) / 0.1)', borderRadius: '12px', border: '1px solid hsl(var(--destructive) / 0.3)', display: 'flex', gap: '12px' }}>
            <AlertTriangle style={{ color: 'hsl(var(--destructive))', flexShrink: 0 }} />
            <div>
              <h4 style={{ margin: '0 0 4px 0', color: 'hsl(var(--destructive))', fontSize: '0.95rem' }}>Missing Dependencies</h4>
              <ul style={{ margin: 0, paddingLeft: '1.2rem', color: 'hsl(var(--destructive))', fontSize: '0.85rem' }}>
                {stageData.warnings.map((w: string, i: number) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          </div>
        )}

        <div style={{ display: 'flex', gap: '2rem' }}>
          {/* Main Prompt Editor */}
          <div style={{ flex: '2', minWidth: '400px' }}>
            <div style={{ padding: '1.5rem', background: 'hsl(var(--card))', borderRadius: '12px', border: '1px solid hsl(var(--border))', marginBottom: '2rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
                <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600, color: 'hsl(var(--foreground))' }}>Prompt Rules</h3>
                <button
                  onClick={() => setAddRuleMode(addRuleMode === 'closed' ? 'direct' : 'closed')}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    padding: '6px 12px', borderRadius: '6px',
                    background: 'hsl(var(--ink) / 0.05)', color: 'hsl(var(--ink))',
                    border: 'none', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 500
                  }}
                >
                  <Plus size={16} /> Add Rule
                </button>
              </div>

              {/* Add Rule Panel */}
              {addRuleMode !== 'closed' && (
                <div style={{ marginBottom: '1.5rem', padding: '1rem', background: 'hsl(var(--background))', borderRadius: '8px', border: '1px solid hsl(var(--border))' }}>
                  <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', borderBottom: '1px solid hsl(var(--border))', paddingBottom: '0.5rem' }}>
                    <button
                      onClick={() => setAddRuleMode('direct')}
                      style={{
                        background: 'none', border: 'none', cursor: 'pointer',
                        fontSize: '0.9rem', fontWeight: 600,
                        color: addRuleMode === 'direct' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))'
                      }}
                    >
                      Direct Edit
                    </button>
                    <button
                      onClick={() => setAddRuleMode('ai')}
                      style={{
                        background: 'none', border: 'none', cursor: 'pointer',
                        fontSize: '0.9rem', fontWeight: 600,
                        color: addRuleMode === 'ai' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))'
                      }}
                    >
                      Using AI
                    </button>
                  </div>

                  {addRuleMode === 'direct' ? (
                    <div>
                      <textarea
                        value={newRuleText}
                        onChange={(e) => setNewRuleText(e.target.value)}
                        placeholder="Enter rule text here..."
                        style={{
                          width: '100%', minHeight: '80px', padding: '8px 12px', borderRadius: '6px',
                          border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                          fontFamily: 'Inter, sans-serif', fontSize: '0.9rem', marginBottom: '12px'
                        }}
                      />
                      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                        <button onClick={() => setAddRuleMode('closed')} style={{ padding: '6px 12px', borderRadius: '6px', border: 'none', background: 'transparent', cursor: 'pointer', color: 'hsl(var(--ink))' }}>Cancel</button>
                        <button onClick={handleAddDirectRule} style={{ padding: '6px 12px', borderRadius: '6px', border: 'none', background: 'hsl(var(--accent))', color: '#fff', cursor: 'pointer', fontWeight: 500 }}>Add</button>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <textarea
                        value={aiDescription}
                        onChange={(e) => setAiDescription(e.target.value)}
                        placeholder="Describe the rule you want to add..."
                        style={{
                          width: '100%', minHeight: '80px', padding: '8px 12px', borderRadius: '6px',
                          border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                          fontFamily: 'Inter, sans-serif', fontSize: '0.9rem', marginBottom: '12px'
                        }}
                      />
                      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginBottom: generatedRule ? '12px' : '0' }}>
                        <button onClick={handleGenerateRule} disabled={generatingRule} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', borderRadius: '6px', border: 'none', background: 'hsl(var(--accent))', color: '#fff', cursor: 'pointer', fontWeight: 500 }}>
                          {generatingRule ? <Loader size={16} className="spin" /> : <Layers size={16} />}
                          Generate
                        </button>
                      </div>
                      
                      {generatedRule && (
                        <div style={{ padding: '1rem', background: 'hsl(var(--accent) / 0.05)', borderRadius: '6px', border: '1px solid hsl(var(--accent) / 0.2)' }}>
                          <h4 style={{ margin: '0 0 8px 0', fontSize: '0.85rem', color: 'hsl(var(--accent))' }}>Generated Preview</h4>
                          <textarea
                            value={generatedRule}
                            onChange={(e) => setGeneratedRule(e.target.value)}
                            style={{
                              width: '100%', minHeight: '60px', padding: '8px', borderRadius: '4px',
                              border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                              fontFamily: 'Inter, sans-serif', fontSize: '0.9rem', marginBottom: '12px'
                            }}
                          />
                          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                            <button onClick={() => setGeneratedRule('')} style={{ padding: '4px 10px', borderRadius: '6px', border: 'none', background: 'transparent', cursor: 'pointer', color: 'hsl(var(--ink))', fontSize: '0.85rem' }}>Discard</button>
                            <button onClick={handleAcceptAiRule} style={{ padding: '4px 10px', borderRadius: '6px', border: 'none', background: 'hsl(var(--accent))', color: '#fff', cursor: 'pointer', fontSize: '0.85rem' }}>Accept</button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Rules List */}
              {rules.length === 0 ? (
                <div style={{ padding: '2rem', textAlign: 'center', color: 'hsl(var(--pencil))', fontSize: '0.9rem' }}>
                  Select a prompt template to edit rules
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  {rules.map(rule => renderRule(rule))}
                </div>
              )}
            </div>

            {/* Test Action Area */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1rem', background: 'hsl(var(--card))', borderRadius: '12px', border: '1px solid hsl(var(--border))' }}>
              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  onClick={() => setShowDraftModal(true)}
                  disabled={rules.length === 0}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '8px',
                    padding: '10px 16px', borderRadius: '8px',
                    background: 'hsl(var(--ink) / 0.05)', color: 'hsl(var(--ink))',
                    border: 'none', cursor: rules.length ? 'pointer' : 'not-allowed',
                    fontSize: '0.9rem', fontWeight: 600
                  }}
                >
                  <Save size={18} /> Save Draft
                </button>
                <button
                  onClick={handleSaveToProduction}
                  disabled={rules.length === 0 || savingProd}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '8px',
                    padding: '10px 16px', borderRadius: '8px',
                    background: 'hsl(140, 60%, 45%)', color: '#fff',
                    border: 'none', cursor: rules.length ? 'pointer' : 'not-allowed',
                    fontSize: '0.9rem', fontWeight: 600
                  }}
                >
                  {savingProd ? <Loader size={18} className="spin" /> : <Check size={18} />} Save to Prod
                </button>
              </div>
              <button
                onClick={handleTestPrompt}
                disabled={testing || rules.length === 0}
                style={{
                  display: 'flex', alignItems: 'center', gap: '8px',
                  padding: '12px 24px', borderRadius: '8px',
                  background: 'hsl(var(--accent))', color: '#fff',
                  border: 'none', cursor: rules.length ? 'pointer' : 'not-allowed',
                  fontSize: '1rem', fontWeight: 700,
                  boxShadow: '0 4px 12px hsl(var(--accent) / 0.3)'
                }}
              >
                {testing ? <Loader size={20} className="spin" /> : <Play size={20} style={{ fill: 'currentColor' }} />}
                Test Prompt
              </button>
            </div>
          </div>

          {/* Sidebar / Results Area */}
          <div style={{ flex: '1', minWidth: '350px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* Drafts Section */}
            <div style={{ padding: '1rem', background: 'hsl(var(--card))', borderRadius: '12px', border: '1px solid hsl(var(--border))' }}>
              <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <History size={18} color="hsl(var(--pencil))" /> Saved Drafts
              </h3>
              {drafts.length === 0 ? (
                <div style={{ fontSize: '0.85rem', color: 'hsl(var(--pencil))' }}>No drafts saved yet.</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {drafts.map((d: any) => (
                    <div key={d.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', background: 'hsl(var(--background))', borderRadius: '6px', border: '1px solid hsl(var(--border))' }}>
                      <div style={{ cursor: 'pointer', flex: 1 }} onClick={() => {
                        setSelectedPromptKey(d.prompt_key)
                        setTimeout(() => setRules(parsePromptToRules(d.template)), 500)
                      }}>
                        <div style={{ fontSize: '0.9rem', fontWeight: 500, color: 'hsl(var(--foreground))' }}>{d.name}</div>
                        <div style={{ fontSize: '0.75rem', color: 'hsl(var(--pencil))' }}>{d.prompt_key} • {new Date(d.created_at).toLocaleDateString()}</div>
                      </div>
                      <button style={{ background: 'none', border: 'none', color: 'hsl(var(--destructive))', cursor: 'pointer', padding: '4px' }}>
                        <Trash2 size={16} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Test Results */}
            {testResult && (
              <div style={{ padding: '1rem', background: 'hsl(var(--card))', borderRadius: '12px', border: '1px solid hsl(var(--border))', flex: 1, overflowY: 'auto' }}>
                <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem', fontWeight: 600, color: 'hsl(var(--accent))' }}>Test Results</h3>
                
                {[
                  { name: 'Generated Output', key: 'output', content: testResult.output, pre: true },
                  { name: 'Model Used', key: 'model', content: `${testResult.model} (${testResult.server_url})` },
                  { name: 'Final Prompt Sent', key: 'prompt', content: testResult.prompt, pre: true },
                  { name: 'Input Context', key: 'context', content: testResult.context ? JSON.stringify(testResult.context, null, 2) : 'None', pre: true },
                  { name: 'Timing', key: 'timing', content: `${testResult.duration_ms}ms • ${testResult.tokens_eval} eval tokens` }
                ].map(section => (
                  <div key={section.name} style={{ marginBottom: '12px', border: '1px solid hsl(var(--border))', borderRadius: '8px', overflow: 'hidden' }}>
                    <button
                      onClick={() => toggleSection(section.name)}
                      style={{
                        width: '100%', padding: '10px 12px', background: 'hsl(var(--background))', border: 'none',
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        cursor: 'pointer', fontSize: '0.9rem', fontWeight: 600, color: 'hsl(var(--ink))'
                      }}
                    >
                      {section.name}
                      {expandedSections[section.name] ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                    </button>
                    {expandedSections[section.name] && (
                      <div style={{ padding: '12px', background: 'hsl(var(--card))', borderTop: '1px solid hsl(var(--border))' }}>
                        {section.pre ? (
                          <pre style={{ margin: 0, padding: '12px', background: 'hsl(var(--ink) / 0.05)', borderRadius: '6px', fontSize: '0.8rem', fontFamily: 'monospace', whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: '300px', overflowY: 'auto' }}>
                            {section.content}
                          </pre>
                        ) : (
                          <div style={{ fontSize: '0.9rem', color: 'hsl(var(--foreground))' }}>{section.content}</div>
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

      {/* Draft Save Modal */}
      {showDraftModal && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          zIndex: 1000, backdropFilter: 'blur(4px)'
        }}>
          <div style={{
            background: 'hsl(var(--card))', borderRadius: '12px', padding: '1.5rem',
            width: '400px', border: '1px solid hsl(var(--border))',
            boxShadow: '0 8px 32px rgba(0,0,0,0.2)'
          }}>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem', fontWeight: 600 }}>Save Draft</h3>
            <input
              type="text"
              value={draftName}
              onChange={(e) => setDraftName(e.target.value)}
              placeholder="e.g. Experimental concise extraction"
              style={{
                width: '100%', padding: '10px 12px', borderRadius: '8px',
                border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                marginBottom: '1.5rem', fontFamily: 'Inter, sans-serif', fontSize: '0.95rem'
              }}
              autoFocus
            />
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button
                onClick={() => setShowDraftModal(false)}
                style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: 'transparent', cursor: 'pointer', fontWeight: 500 }}
              >
                Cancel
              </button>
              <button
                onClick={handleSaveDraft}
                disabled={!draftName.trim() || savingDraft}
                style={{
                  padding: '8px 16px', borderRadius: '8px', border: 'none',
                  background: 'hsl(var(--accent))', color: '#fff', cursor: draftName.trim() ? 'pointer' : 'not-allowed',
                  fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px'
                }}
              >
                {savingDraft ? <Loader size={16} className="spin" /> : <Save size={16} />} Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
