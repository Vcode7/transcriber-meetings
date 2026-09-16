import React, { useState, useEffect } from 'react';
import {
  Brain, Sparkles, Layers, Cpu, CheckCircle2,
  FileText, ArrowRight, ShieldCheck, Database
} from 'lucide-react';
import BaseModelSelector from './components/BaseModelSelector';
import DocumentUploader from './components/DocumentUploader';
import TextExtractionView from './components/TextExtractionView';
import TrainingProgressView from './components/TrainingProgressView';
import SaveVariantCard from './components/SaveVariantCard';
import VariantManagementTab from './components/VariantManagementTab';
import {
  BaseModelInfo,
  ExtractionResult,
  TrainingJobStatus,
  embeddingTrainingApi
} from '../../services/embeddingTrainingApi';

type TopTab = 'adaptation' | 'variants';
type WizardStep = 1 | 2 | 3 | 4 | 5;

export default function EmbeddingTrainingLayout() {
  const [activeTab, setActiveTab] = useState<TopTab>('adaptation');
  const [currentStep, setCurrentStep] = useState<WizardStep>(1);

  // Shared State across Steps
  const [baseModels, setBaseModels] = useState<BaseModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [extractionResult, setExtractionResult] = useState<ExtractionResult | null>(null);
  const [completedJob, setCompletedJob] = useState<TrainingJobStatus | null>(null);

  // Fetch base models on mount
  useEffect(() => {
    const loadModels = async () => {
      setModelsLoading(true);
      try {
        const models = await embeddingTrainingApi.getBaseModels();
        setBaseModels(models);
        const defaultModel = models.find((m) => m.is_default) || models[0];
        if (defaultModel) {
          setSelectedModelId(defaultModel.id);
        }
      } catch (err) {
        console.error('Failed to discover base models:', err);
      } finally {
        setModelsLoading(false);
      }
    };
    loadModels();
  }, []);

  const handleVariantSaved = (variantId: string, activated: boolean) => {
    setActiveTab('variants');
    // Reset wizard state for future training
    setCurrentStep(1);
    setUploadedFiles([]);
    setExtractionResult(null);
    setCompletedJob(null);
  };

  const stepsList = [
    { num: 1, label: 'Base Model' },
    { num: 2, label: 'Upload Documents' },
    { num: 3, label: 'Extract Text' },
    { num: 4, label: 'Embedding Training' },
    { num: 5, label: 'Save Variant' },
  ];

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      minHeight: 0,
      width: '100%',
      background: 'hsl(var(--background))',
      color: 'hsl(var(--foreground))',
      fontFamily: 'Inter, sans-serif',
      overflow: 'hidden',
    }}>
      {/* Top Header */}
      <div style={{
        background: 'linear-gradient(135deg, hsl(var(--card)) 0%, hsl(var(--background)) 100%)',
        borderBottom: '1px solid hsl(var(--border))',
        padding: '1.5rem 2rem 0',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: 42, height: 42, borderRadius: '12px',
              background: 'linear-gradient(135deg, hsl(215 90% 55%) 0%, hsl(var(--accent)) 100%)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 0 20px hsl(var(--accent) / 0.35)', flexShrink: 0, color: 'white',
            }}>
              <Brain size={22} />
            </div>
            <div>
              <h1 style={{
                margin: 0, fontSize: '1.4rem', fontWeight: 800,
                letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: 8,
              }}>
                Embedding Training
                <Sparkles size={16} style={{ color: 'hsl(var(--accent))' }} />
              </h1>
              <p style={{ margin: 0, fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))', marginTop: 2 }}>
                Adapt embedding models to the ADA domain using raw documents without modifying base models
              </p>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div style={{ display: 'flex', gap: '4px' }}>
          <button
            onClick={() => setActiveTab('adaptation')}
            style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              padding: '0.6rem 1.4rem',
              background: activeTab === 'adaptation' ? 'hsl(var(--accent) / 0.12)' : 'transparent',
              border: 'none',
              borderBottom: activeTab === 'adaptation' ? '2px solid hsl(var(--accent))' : '2px solid transparent',
              color: activeTab === 'adaptation' ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))',
              cursor: 'pointer',
              borderRadius: '8px 8px 0 0',
              fontSize: '0.875rem',
              fontWeight: activeTab === 'adaptation' ? 700 : 500,
              transition: 'all 0.15s ease',
              fontFamily: 'Inter, sans-serif',
            }}
          >
            <Sparkles size={15} />
            Domain Adaptation
          </button>

          <button
            onClick={() => setActiveTab('variants')}
            style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              padding: '0.6rem 1.4rem',
              background: activeTab === 'variants' ? 'hsl(var(--accent) / 0.12)' : 'transparent',
              border: 'none',
              borderBottom: activeTab === 'variants' ? '2px solid hsl(var(--accent))' : '2px solid transparent',
              color: activeTab === 'variants' ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))',
              cursor: 'pointer',
              borderRadius: '8px 8px 0 0',
              fontSize: '0.875rem',
              fontWeight: activeTab === 'variants' ? 700 : 500,
              transition: 'all 0.15s ease',
              fontFamily: 'Inter, sans-serif',
            }}
          >
            <Layers size={15} />
            Variant Management
          </button>
        </div>
      </div>

      {/* Main Body */}
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '1.75rem 2rem' }}>
        {activeTab === 'variants' ? (
          <VariantManagementTab
            onTrainNewClick={() => {
              setActiveTab('adaptation');
              setCurrentStep(1);
            }}
          />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem', maxWidth: '1080px', margin: '0 auto' }}>
            {/* Stepper Progress Header */}
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              background: 'hsl(var(--card))', borderRadius: '14px',
              border: '1px solid hsl(var(--border))', padding: '0.85rem 1.25rem',
              overflowX: 'auto',
            }}>
              {stepsList.map((st, i) => {
                const isCurrent = currentStep === st.num;
                const isPassed = currentStep > st.num;
                const isClickable = st.num < currentStep;

                return (
                  <React.Fragment key={st.num}>
                    <div
                      onClick={() => isClickable && setCurrentStep(st.num as WizardStep)}
                      style={{
                        display: 'flex', alignItems: 'center', gap: '8px',
                        cursor: isClickable ? 'pointer' : 'default',
                        opacity: isCurrent || isPassed ? 1 : 0.45,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      <div style={{
                        width: 26, height: 26, borderRadius: '50%',
                        background: isPassed
                          ? 'hsl(150 75% 45%)'
                          : isCurrent
                          ? 'hsl(var(--accent))'
                          : 'hsl(var(--muted))',
                        color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '0.75rem', fontWeight: 700,
                      }}>
                        {isPassed ? <CheckCircle2 size={16} /> : st.num}
                      </div>
                      <span style={{
                        fontSize: '0.82rem',
                        fontWeight: isCurrent ? 700 : 500,
                        color: isCurrent ? 'hsl(var(--accent))' : 'hsl(var(--foreground))',
                      }}>
                        {st.label}
                      </span>
                    </div>

                    {i < stepsList.length - 1 && (
                      <div style={{
                        flex: 1, height: '2px', minWidth: '20px', margin: '0 8px',
                        background: isPassed ? 'hsl(150 75% 45%)' : 'hsl(var(--border))',
                        transition: 'background 0.2s ease',
                      }} />
                    )}
                  </React.Fragment>
                );
              })}
            </div>

            {/* Step 1: Base Model */}
            {currentStep === 1 && (
              <BaseModelSelector
                models={baseModels}
                selectedModelId={selectedModelId}
                onSelectModel={setSelectedModelId}
                onNext={() => setCurrentStep(2)}
                loading={modelsLoading}
              />
            )}

            {/* Step 2: Upload Documents */}
            {currentStep === 2 && (
              <DocumentUploader
                files={uploadedFiles}
                onFilesChange={setUploadedFiles}
                onBack={() => setCurrentStep(1)}
                onNext={() => setCurrentStep(3)}
              />
            )}

            {/* Step 3: Extract Text */}
            {currentStep === 3 && (
              <TextExtractionView
                files={uploadedFiles}
                extractionResult={extractionResult}
                onExtractionComplete={setExtractionResult}
                onBack={() => setCurrentStep(2)}
                onNext={() => setCurrentStep(4)}
              />
            )}

            {/* Step 4: Embedding Training */}
            {currentStep === 4 && extractionResult && (
              <TrainingProgressView
                baseModelName={baseModels.find((m) => m.id === selectedModelId)?.name || 'Qwen3-Embedding-0.6B'}
                extractionResult={extractionResult}
                onTrainingFinished={(data) => {
                  setCompletedJob(data);
                }}
                onBack={() => setCurrentStep(3)}
                onNext={() => setCurrentStep(5)}
              />
            )}

            {/* Step 5: Save Variant */}
            {currentStep === 5 && completedJob && extractionResult && (
              <SaveVariantCard
                jobStatus={completedJob}
                extractionResult={extractionResult}
                onVariantSaved={handleVariantSaved}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
