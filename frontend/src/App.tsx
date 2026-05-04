import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

// --- Types ---
type Tab = 'hide' | 'extract'
interface Metrics {
  mse: number
  psnr: number
  ssim: number
  capacity: number
}
interface RobustnessResult {
  name: string
  image: string
  extractedText: string
  survivalRate: number
}

// --- Main App ---
function App() {
  const [activeTab, setActiveTab] = useState<Tab>('hide')
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null)

  // Hide tab state
  const [prompt, setPrompt] = useState('')
  const [secretText, setSecretText] = useState('')
  const [showSecret, setShowSecret] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [resultImage, setResultImage] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [showMetrics, setShowMetrics] = useState(false)
  const [robustnessResults, setRobustnessResults] = useState<RobustnessResult[] | null>(null)
  const [isTestingRobustness, setIsTestingRobustness] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  // Extract tab state
  const [extractFile, setExtractFile] = useState<File | null>(null)
  const [extractPreview, setExtractPreview] = useState<string | null>(null)
  const [extractedText, setExtractedText] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // --- Health check ---
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('/api/health', { signal: AbortSignal.timeout(4000) })
        setBackendOnline(res.ok)
      } catch {
        setBackendOnline(false)
      }
    }
    check()
    const interval = setInterval(check, 30000)
    return () => clearInterval(interval)
  }, [])

  // --- Hide handler ---
  const handleGenerateAndHide = async () => {
    setIsLoading(true)
    setError(null)
    setResultImage(null)
    setRobustnessResults(null)

    const formData = new FormData()
    formData.append('prompt', prompt)
    formData.append('secret_text', secretText)

    try {
      const response = await fetch('/api/generate-and-hide', { method: 'POST', body: formData })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.error || `Помилка сервера: ${response.status}`)
      }

      const data = await response.json()
      if (data.error) throw new Error(data.error)
      
      setResultImage(data.image)
      setMetrics(data.metrics)
      setShowMetrics(false) // reset toggle
    } catch (err: any) {
      setError(err.message || 'Невідома помилка')
    } finally {
      setIsLoading(false)
    }
  }

  // --- Robustness Test handler ---
  const handleRobustnessTest = async () => {
    if (!resultImage || !secretText) return
    setIsTestingRobustness(true)
    setError(null)

    try {
      // Convert base64 data URL to blob
      const res = await fetch(resultImage)
      const blob = await res.blob()

      const formData = new FormData()
      formData.append('file', blob, 'stego_test.png')
      formData.append('secret_text', secretText)

      const response = await fetch('/api/test-robustness', { method: 'POST', body: formData })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.error || `Помилка сервера: ${response.status}`)
      }

      const data = await response.json()
      setRobustnessResults(data.results)
      setShowMetrics(false) // Hide metrics to focus on tests
    } catch (err: any) {
      setError(err.message || 'Невідома помилка при тестуванні')
    } finally {
      setIsTestingRobustness(false)
    }
  }

  // --- Extract handler ---
  const handleExtract = async () => {
    if (!extractFile) return
    setIsLoading(true)
    setError(null)
    setExtractedText(null)

    const formData = new FormData()
    formData.append('file', extractFile)

    try {
      const response = await fetch('/api/extract', { method: 'POST', body: formData })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.error || `Помилка сервера: ${response.status}`)
      }

      const data = await response.json()
      setExtractedText(data.text ?? '')
    } catch (err: any) {
      setError(err.message || 'Невідома помилка')
    } finally {
      setIsLoading(false)
    }
  }

  // --- File selection ---
  const handleFileSelect = useCallback((file: File | null) => {
    if (!file) return
    setExtractFile(file)
    setExtractedText(null)
    setError(null)
    const url = URL.createObjectURL(file)
    setExtractPreview(url)
  }, [])

  // --- Drag & drop ---
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) handleFileSelect(file)
  }

  // --- Copy to clipboard ---
  const handleCopyText = () => {
    if (!extractedText) return
    navigator.clipboard.writeText(extractedText).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  // --- Tab switch ---
  const switchTab = (tab: Tab) => {
    setActiveTab(tab)
    setError(null)
  }

  return (
    <div className="app-wrapper">
      <div className="main-card">

        {/* ===== Header ===== */}
        <header className="header">
          <div className="header-badge">
            <span
              className={`status-dot ${backendOnline === true ? 'online' : backendOnline === false ? 'offline' : ''}`}
            />
            {backendOnline === true ? 'Сервер підключено' : backendOnline === false ? 'Сервер недоступний' : 'Перевірка…'}
          </div>

          <h1>СтегоAI</h1>

          <p className="header-subtitle">
            Приховування інформації у зображеннях, згенерованих моделлю&nbsp;
            <strong style={{ color: 'var(--color-accent)' }}>Stable Diffusion</strong>,
            з використанням DCT-стеганографії
          </p>
        </header>

        {/* ===== Tabs ===== */}
        <div className="tabs-container" role="tablist">
          <button
            id="tab-hide"
            role="tab"
            aria-selected={activeTab === 'hide'}
            aria-controls="panel-hide"
            className={`tab-btn ${activeTab === 'hide' ? 'active' : ''}`}
            onClick={() => switchTab('hide')}
          >
            <span className="tab-icon">🔒</span>
            Приховати повідомлення
          </button>
          <button
            id="tab-extract"
            role="tab"
            aria-selected={activeTab === 'extract'}
            aria-controls="panel-extract"
            className={`tab-btn ${activeTab === 'extract' ? 'active' : ''}`}
            onClick={() => switchTab('extract')}
          >
            <span className="tab-icon">🔓</span>
            Витягнути повідомлення
          </button>
        </div>

        {/* ===== Error Banner ===== */}
        {error && (
          <div className="error-banner" role="alert">
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* ===== HIDE PANEL ===== */}
        {activeTab === 'hide' && (
          <div id="panel-hide" role="tabpanel" aria-labelledby="tab-hide" className="panel">

            <div className="field">
              <label className="field-label" htmlFor="prompt-input">
                Промпт для Stable Diffusion
                <span className="char-counter">{prompt.length} симв.</span>
              </label>
              <textarea
                id="prompt-input"
                className="field-textarea"
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                placeholder="Опишіть зображення-контейнер (наприклад: «Футуристичне місто вночі, кіберпанк стиль, деталізація 4K»)"
              />
            </div>

            <div className="field">
              <label className="field-label" htmlFor="secret-input">
                Секретне повідомлення
              </label>
              <div className="secret-input-wrap">
                <input
                  id="secret-input"
                  type={showSecret ? 'text' : 'password'}
                  className="field-input"
                  value={secretText}
                  onChange={e => setSecretText(e.target.value)}
                  placeholder="Введіть текст для приховування у зображенні"
                  autoComplete="off"
                />
                <button
                  type="button"
                  className="eye-btn"
                  aria-label={showSecret ? 'Приховати' : 'Показати'}
                  onClick={() => setShowSecret(v => !v)}
                >
                  {showSecret ? '🙈' : '👁️'}
                </button>
              </div>
            </div>

            <button
              id="hide-btn"
              className="primary-btn"
              onClick={handleGenerateAndHide}
              disabled={isLoading || !prompt.trim() || !secretText.trim()}
            >
              {isLoading ? (
                <>
                  <span className="spinner" />
                  Генерація зображення…
                </>
              ) : (
                <>🖼️ Згенерувати та приховати</>
              )}
            </button>

            {isLoading && (
              <p className="loading-hint">
                ⏳ Модель на безкоштовному сервері може «прогріватись» 20–60 секунд — будь ласка, зачекайте
              </p>
            )}

            {resultImage && (
              <div className="result-section">
                <span className="result-label">✅ Готове зображення (PNG)</span>
                <img src={resultImage} alt="Зображення зі схованим повідомленням" className="result-image" />
                
                <div className="result-actions" style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
                  <a
                    id="download-btn"
                    href={resultImage}
                    download="stego_image.png"
                    className="action-btn"
                    style={{ flex: 1 }}
                  >
                    ⬇️ Завантажити PNG
                  </a>
                  {metrics && (
                    <button
                      className="action-btn"
                      onClick={() => setShowMetrics(prev => !prev)}
                      style={{ flex: 1, backgroundColor: showMetrics ? 'rgba(255,255,255,0.2)' : '' }}
                    >
                      📊 {showMetrics ? 'Сховати статистику' : 'Статистичний аналіз'}
                    </button>
                  )}
                  <button
                    className="action-btn"
                    onClick={handleRobustnessTest}
                    disabled={isTestingRobustness}
                    style={{ flex: 1 }}
                  >
                    {isTestingRobustness ? 'Тестування...' : '🛡️ Тест на стійкість'}
                  </button>
                </div>

                {showMetrics && metrics && (
                  <div className="metrics-panel">
                    <h4>Результати стегоаналізу</h4>
                    <div className="metrics-grid">
                      <div className="metric-card">
                        <span className="metric-title">MSE (Помилка)</span>
                        <span className="metric-value">{metrics.mse.toFixed(4)}</span>
                        <span className="metric-desc">Чим ближче до 0, тим краще</span>
                      </div>
                      <div className="metric-card">
                        <span className="metric-title">PSNR (Сигнал/Шум)</span>
                        <span className="metric-value">{metrics.psnr.toFixed(2)} дБ</span>
                        <span className="metric-desc">&gt; 40 дБ = непомітно</span>
                      </div>
                      <div className="metric-card">
                        <span className="metric-title">SSIM (Подібність)</span>
                        <span className="metric-value">{metrics.ssim.toFixed(4)}</span>
                        <span className="metric-desc">1.0 = ідентичні</span>
                      </div>
                      <div className="metric-card">
                        <span className="metric-title">Ємність</span>
                        <span className="metric-value">{metrics.capacity} симв.</span>
                        <span className="metric-desc">Макс. довжина тексту</span>
                      </div>
                    </div>
                  </div>
                )}

                {robustnessResults && (
                  <div className="robustness-panel">
                    <h4>Результати тестів на стійкість</h4>
                    <p className="robustness-subtitle">
                      Система намагається зруйнувати зображення і витягнути текст <strong>{secretText}</strong>
                    </p>
                    <div className="robustness-list">
                      {robustnessResults.map((res, i) => (
                        <div key={i} className="robustness-card">
                          <img src={res.image} alt={res.name} className="robustness-thumb" />
                          <div className="robustness-info">
                            <h5 className="robustness-name">{res.name}</h5>
                            
                            <div className="survival-bar-container">
                              <div className="survival-bar-bg">
                                <div 
                                  className={`survival-bar-fill ${res.survivalRate >= 90 ? 'high' : res.survivalRate >= 50 ? 'medium' : 'low'}`}
                                  style={{ width: `${res.survivalRate}%` }}
                                ></div>
                              </div>
                              <span className="survival-percent">{res.survivalRate}% вціліло</span>
                            </div>

                            <div className="robustness-text">
                              <span className="text-label">Витягнуто:</span>
                              <code className={res.survivalRate < 100 ? 'text-broken' : 'text-perfect'}>
                                {res.extractedText || '— Порожньо —'}
                              </code>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ===== EXTRACT PANEL ===== */}
        {activeTab === 'extract' && (
          <div id="panel-extract" role="tabpanel" aria-labelledby="tab-extract" className="panel">

            <div className="field">
              <label className="field-label">Зображення з прихованим повідомленням</label>

              <div
                id="dropzone"
                className={`dropzone ${isDragOver ? 'drag-over' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setIsDragOver(true) }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={handleDrop}
                role="button"
                tabIndex={0}
                aria-label="Завантажити зображення"
                onKeyDown={e => e.key === 'Enter' && fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={e => handleFileSelect(e.target.files?.[0] || null)}
                />
                {extractPreview ? (
                  <>
                    <img src={extractPreview} alt="Попередній перегляд" className="dropzone-preview" />
                    <p className="dropzone-text">
                      {extractFile?.name}
                    </p>
                    <p className="dropzone-hint">Натисніть, щоб замінити</p>
                  </>
                ) : (
                  <>
                    <span className="dropzone-icon">📂</span>
                    <p className="dropzone-text">
                      <strong>Перетягніть зображення</strong> або натисніть для вибору
                    </p>
                    <p className="dropzone-hint">PNG, JPG, WEBP</p>
                  </>
                )}
              </div>
            </div>

            <button
              id="extract-btn"
              className="primary-btn"
              onClick={handleExtract}
              disabled={isLoading || !extractFile}
            >
              {isLoading ? (
                <>
                  <span className="spinner" />
                  Витягування…
                </>
              ) : (
                <>🔍 Витягнути приховане повідомлення</>
              )}
            </button>

            {extractedText !== null && (
              <div className="result-section">
                <span className="result-label">📨 Витягнуте повідомлення</span>
                <div className="message-box" id="extracted-message">
                  {extractedText || (
                    <span className="message-empty">Повідомлення не знайдено або зображення не містить прихованого тексту.</span>
                  )}
                </div>
                {extractedText && (
                  <div className="result-actions">
                    <button
                      id="copy-btn"
                      className={`action-btn ${copied ? 'copied' : ''}`}
                      onClick={handleCopyText}
                    >
                      {copied ? '✅ Скопійовано' : '📋 Копіювати текст'}
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ===== Info Panel ===== */}
        <div className="info-panel">
          <div className="info-item">
            <span className="info-item-label">Метод стеганографії</span>
            <span className="info-item-value"><strong>ДКП (DCT)</strong> — у блоках 8×8 пікселів</span>
          </div>
          <div className="info-item">
            <span className="info-item-label">Кольоровий простір</span>
            <span className="info-item-value"><strong>YCrCb</strong> — вбудовування у Y-канал</span>
          </div>
          <div className="info-item">
            <span className="info-item-label">Генерація контейнера</span>
            <span className="info-item-value"><strong>Stable Diffusion XL</strong> via Hugging Face</span>
          </div>
          <div className="info-item">
            <span className="info-item-label">Вихідний формат</span>
            <span className="info-item-value"><strong>PNG</strong> — без втрат для захисту даних</span>
          </div>
        </div>

      </div>
    </div>
  )
}

export default App
