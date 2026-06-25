<script setup>
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import { marked } from 'marked'
import katex from 'katex'
import JSZip from 'jszip'

const file = ref(null)
const fileName = ref('')
const dragOver = ref(false)
// Source-aligned backends; the server reports which are actually available.
const ALL_BACKENDS = ['hybrid-engine', 'pipeline', 'vlm']
const backend = ref('hybrid-engine')
const backends = ref(ALL_BACKENDS)
const maxPages = ref(1000)
const showAdvanced = ref(false)
// Advanced options (aligned with MinerU's gradio demo).
const tableEnable = ref(true)
const formulaEnable = ref(true)
const imageAnalysis = ref(true)
const hybridEffort = ref('medium')
const lang = ref('ch')
const isOcr = ref(false)
const isPipeline = computed(() => backend.value === 'pipeline')
const isHybrid = computed(() => backend.value === 'hybrid-engine')
const isVlmish = computed(() => backend.value === 'vlm' || backend.value === 'hybrid-engine')
const converting = ref(false)
const error = ref('')
const tab = ref('render')        // render | text | json
const md = ref('')
const contentList = ref(null)
const images = ref({})           // { 'images/{hash}.jpg' filename: base64 } from the server
const zipUrl = ref('')           // object URL of the bundled {name}.zip (md + images/ + json)
const zipName = ref('')
const zipSizeKB = ref(0)
const serverInfo = ref('')

const steps = ['准备请求', '检查服务', '提交任务', '排队', '解析中', '下载结果', '整理输出', '完成']
const stepState = reactive(steps.map(() => 'idle'))  // idle | active | done
function setStep(i, s) { stepState[i] = s }
function resetSteps() { steps.forEach((_, i) => (stepState[i] = 'idle')) }

onMounted(async () => {
  try {
    const r = await fetch('/info')
    if (r.ok) {
      const j = await r.json()
      if (Array.isArray(j.backends) && j.backends.length) backends.value = j.backends
      if (j.default) backend.value = j.default
      else if (!backends.value.includes(backend.value)) backend.value = backends.value[0]
      serverInfo.value = `${j.version || ''}`.trim()
    }
  } catch (_) { /* dev: backend may not be up yet */ }
  window.addEventListener('paste', onPaste)
})
onUnmounted(() => window.removeEventListener('paste', onPaste))
// Backend hints + advanced-option labels/info aligned with MinerU's gradio i18n (zh).
const backendHint = computed(() => ({
  'hybrid-engine': '独家混合引擎解析，超高精度',
  'pipeline': '传统多模型管道解析，低资源，无幻觉',
  'vlm': '多模态大模型端到端解析，高精度',
}[backend.value] || '选择文档解析的后端引擎。'))
// Formula label/info depend on the backend (行间/行内 differ by engine).
const formulaLabel = computed(() => backend.value === 'vlm' ? '启用行间公式识别'
  : backend.value === 'pipeline' ? '启用公式识别' : '启用行内公式识别')
const formulaInfo = computed(() => backend.value === 'vlm' ? '禁用后，行间公式将显示为图片。'
  : backend.value === 'pipeline' ? '禁用后，行间公式将显示为图片，行内公式将不会被检测或解析。'
  : '禁用后，行内公式将不会被检测或解析。')
// 图片分析: vlm always; hybrid only at high effort (medium forces it off).
const showImageAnalysis = computed(() => backend.value === 'vlm' || (isHybrid.value && hybridEffort.value === 'high'))
// 强制 OCR: pipeline or hybrid (not vlm); the hint differs (pipeline needs the right OCR language).
const showForceOcr = computed(() => isPipeline.value || isHybrid.value)
const forceOcrInfo = computed(() => isHybrid.value ? '仅在识别效果极差时启用。'
  : '仅在识别效果极差时启用，需选择正确的 OCR 语言。')

const previewUrl = ref('')
const previewType = ref('')   // 'pdf' | 'image' | ''
function onPick(e) { setFile(e.target.files?.[0]) }
function onDrop(e) { dragOver.value = false; setFile(e.dataTransfer.files?.[0]) }
function setFile(f) {
  if (!f) return
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  file.value = f
  fileName.value = f.name || 'pasted-image.png'
  const name = f.name || ''
  const isPdf = f.type === 'application/pdf' || /\.pdf$/i.test(name)
  const isImg = (f.type && f.type.startsWith('image/')) || /\.(png|jpe?g|webp|bmp|gif|tiff?)$/i.test(name)
  previewType.value = isPdf ? 'pdf' : (isImg ? 'image' : '')
  previewUrl.value = (isPdf || isImg) ? URL.createObjectURL(f) : ''
}
function b64ToBlobUrl(b64, mime) {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return URL.createObjectURL(new Blob([bytes], { type: mime }))
}
// Paste an image straight from the clipboard (screenshot or copied picture).
function onPaste(e) {
  const items = e.clipboardData && e.clipboardData.items
  if (!items) return
  for (const it of items) {
    if (it.type && it.type.startsWith('image/')) {
      const blob = it.getAsFile()
      if (blob) {
        const ext = (it.type.split('/')[1] || 'png').replace('jpeg', 'jpg')
        const named = new File([blob], blob.name || `pasted-${Date.now()}.${ext}`, { type: it.type })
        setFile(named)
        e.preventDefault()
        return
      }
    }
  }
}
function clearAll() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  if (zipUrl.value) URL.revokeObjectURL(zipUrl.value)
  file.value = null; fileName.value = ''; previewUrl.value = ''; previewType.value = ''
  md.value = ''; contentList.value = null; images.value = {}; zipUrl.value = ''; zipName.value = ''
  error.value = ''; resetSteps()
}

async function convert() {
  if (!file.value || converting.value) return
  converting.value = true; error.value = ''; md.value = ''; contentList.value = null
  resetSteps()
  try {
    setStep(0, 'active'); setStep(0, 'done')
    setStep(1, 'active')
    setStep(2, 'active'); setStep(1, 'done')
    const fd = new FormData()
    fd.append('files', file.value)
    fd.append('max_pages', String(maxPages.value))
    fd.append('backend', backend.value)
    fd.append('table_enable', tableEnable.value ? 'true' : 'false')
    fd.append('formula_enable', formulaEnable.value ? 'true' : 'false')
    fd.append('image_analysis', imageAnalysis.value ? 'true' : 'false')
    fd.append('is_ocr', isOcr.value ? 'true' : 'false')
    fd.append('lang', lang.value)
    fd.append('effort', hybridEffort.value)
    setStep(3, 'done'); setStep(4, 'active')
    const r = await fetch('/file_parse', { method: 'POST', body: fd })
    setStep(4, 'done'); setStep(5, 'active')
    if (!r.ok) {
      const t = await r.text()
      throw new Error(t || `HTTP ${r.status}`)
    }
    const j = await r.json()
    setStep(5, 'done'); setStep(6, 'active')
    md.value = j.md_content || ''
    contentList.value = j.content_list || null
    images.value = j.images || {}
    await buildZip()
    // Swap the preview to the layout-highlighted PDF (boxes + reading-order numbers),
    // matching the source project's gradio preview.
    if (j.layout_pdf) {
      if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
      previewUrl.value = b64ToBlobUrl(j.layout_pdf, 'application/pdf')
      previewType.value = 'pdf'
    }
    setStep(6, 'done'); setStep(7, 'done')
  } catch (e) {
    error.value = String(e.message || e)
    stepState.forEach((s, i) => { if (s === 'active') stepState[i] = 'idle' })
  } finally {
    converting.value = false
  }
}

// --- markdown rendering with $$ / $ math via KaTeX ---
function renderMath(text) {
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, e) => {
    try { return katex.renderToString(e.trim(), { displayMode: true, throwOnError: false }) } catch { return _ }
  })
  text = text.replace(/(^|[^\\$])\$([^$\n]+?)\$/g, (m, p, e) => {
    try { return p + katex.renderToString(e.trim(), { displayMode: false, throwOnError: false }) } catch { return m }
  })
  return text
}
// For the rendered preview, resolve images/{hash}.jpg refs to inline data URLs from the map
// (the raw "Markdown 文本" tab keeps the file paths, matching MinerU's output).
function resolveImages(text) {
  const imgs = images.value
  if (!imgs || !Object.keys(imgs).length) return text
  return text.replace(/images\/([A-Za-z0-9_.-]+)/g, (full, name) =>
    imgs[name] ? `data:image/jpeg;base64,${imgs[name]}` : full)
}
const renderedMd = computed(() => {
  if (!md.value) return ''
  try { return renderMath(marked.parse(resolveImages(md.value), { breaks: true, gfm: true })) } catch { return '' }
})

// Bundle a MinerU-style result zip: {name}.md (with images/ refs) + images/*.jpg + content list.
async function buildZip() {
  if (zipUrl.value) { URL.revokeObjectURL(zipUrl.value); zipUrl.value = '' }
  zipName.value = ''
  if (!md.value) return
  const stem = (fileName.value || 'output').replace(/\.[^.]+$/, '')
  const zip = new JSZip()
  zip.file(`${stem}.md`, md.value)
  if (contentList.value) zip.file(`${stem}_content_list.json`, JSON.stringify(contentList.value, null, 2))
  const folder = zip.folder('images')
  for (const [name, b64] of Object.entries(images.value)) folder.file(name, b64, { base64: true })
  const blob = await zip.generateAsync({ type: 'blob' })
  zipUrl.value = URL.createObjectURL(blob)
  zipName.value = `${stem}.zip`
  zipSizeKB.value = Math.round(blob.size / 1024)
}
const jsonText = computed(() => contentList.value ? JSON.stringify(contentList.value, null, 2) : '')

function copy(text) { navigator.clipboard?.writeText(text) }
</script>

<template>
  <div class="page">
    <header class="hero">
      <h1>MinerU 3：文档提取演示</h1>
      <p class="sub">开源文档提取工具，支持将 PDF、DOCX、PPTX、XLSX 和图片转换为 Markdown 与 JSON。</p>
      <p class="sub">本地原生引擎 · C++/MLX（mlx-mineru）<span v-if="serverInfo"> · {{ serverInfo }}</span></p>
      <div class="hero-btns">
        <a class="pill" href="https://github.com/opendatalab/MinerU" target="_blank">代码</a>
        <a class="pill" href="https://huggingface.co/opendatalab" target="_blank">模型</a>
        <a class="pill" href="https://arxiv.org/abs/2409.18839" target="_blank">论文</a>
        <a class="pill" href="https://mineru.net" target="_blank">主页</a>
        <a class="pill" href="https://github.com/opendatalab/MinerU/releases" target="_blank">下载</a>
      </div>
    </header>

    <main class="cols">
      <!-- left: controls -->
      <section class="card controls">
        <div class="label"><b>请选择或粘贴要上传的文件</b><br /><span class="muted">PDF、图片、DOCX、PPTX 或 XLSX</span></div>
        <div class="drop" :class="{ over: dragOver }" @dragover.prevent="dragOver = true"
             @dragleave.prevent="dragOver = false" @drop.prevent="onDrop"
             @click="$refs.fi.click()">
          <div class="up-ico">⬆</div>
          <div v-if="!fileName">将文件拖放到此处<br /><span class="muted">- 或 -</span><br />点击上传<br /><span class="muted small">也可直接粘贴图像（⌘/Ctrl+V）</span></div>
          <div v-else class="picked">{{ fileName }}</div>
          <input ref="fi" type="file" hidden accept=".pdf,.png,.jpg,.jpeg,.webp,.bmp,.gif,.tif,.tiff,.docx,.pptx,.xlsx" @change="onPick" />
        </div>

        <div class="field">
          <div class="label">解析后端</div>
          <div class="muted small">{{ backendHint }}</div>
          <select v-model="backend">
            <option v-for="b in backends" :key="b" :value="b">{{ b }}</option>
          </select>
        </div>

        <div class="field">
          <div class="label-row"><span class="label">最大转换页数</span><span class="badge">{{ maxPages }}</span></div>
          <input type="range" min="1" max="1000" v-model.number="maxPages" />
        </div>

        <button class="adv" @click="showAdvanced = !showAdvanced">高级选项 {{ showAdvanced ? '▲' : '▼' }}</button>
        <div v-if="showAdvanced" class="adv-body">
          <div class="opt-group">
            <label class="opt"><input type="checkbox" v-model="tableEnable" /> 启用表格识别</label>
            <div class="opt-info">禁用后，表格将显示为图片。</div>
          </div>
          <div class="opt-group">
            <label class="opt"><input type="checkbox" v-model="formulaEnable" /> {{ formulaLabel }}</label>
            <div class="opt-info">{{ formulaInfo }}</div>
          </div>
          <div v-if="showImageAnalysis" class="opt-group">
            <label class="opt"><input type="checkbox" v-model="imageAnalysis" /> 启用图片分析</label>
            <div class="opt-info">禁用后，图片/图表块仍保留版面位置，但跳过 VLM 图片/图表分析。</div>
          </div>
          <div v-if="isHybrid" class="opt-group">
            <div class="opt-label">解析强度</div>
            <div class="opt-info">Medium 速度更快；High 精度更高，耗时可能更长。</div>
            <div class="opt-field">
              <label class="radio"><input type="radio" value="medium" v-model="hybridEffort" /> medium</label>
              <label class="radio"><input type="radio" value="high" v-model="hybridEffort" /> high</label>
            </div>
          </div>
          <div v-if="isPipeline" class="opt-group">
            <div class="opt-label">OCR 语言</div>
            <div class="opt-info">为扫描版 PDF 和图片选择 OCR 语言。</div>
            <select v-model="lang" class="opt-select">
              <option value="ch">中文 (ch)</option>
              <option value="en">English (en)</option>
            </select>
          </div>
          <div v-if="showForceOcr" class="opt-group">
            <label class="opt"><input type="checkbox" v-model="isOcr" /> 强制启用 OCR</label>
            <div class="opt-info">{{ forceOcrInfo }}</div>
          </div>
        </div>

        <div class="btn-row">
          <button class="primary" :disabled="!file || converting" @click="convert">
            {{ converting ? '转换中…' : '转换' }}
          </button>
          <button class="ghost" @click="clearAll">清除</button>
        </div>

        <div class="label section">转换结果</div>
        <div v-if="error" class="err">{{ error }}</div>
        <a v-if="zipName" class="zip-dl" :href="zipUrl" :download="zipName">
          <span class="zip-ico">🗎</span>
          <span class="zip-name">{{ zipName }}</span>
          <span class="muted small">{{ zipSizeKB >= 1024 ? (zipSizeKB/1024).toFixed(1) + ' MB' : zipSizeKB + ' KB' }}</span>
          <span class="zip-arrow">↓</span>
        </a>

        <div class="card tasks">
          <div class="label"><b>等待任务</b></div>
          <ul class="steps">
            <li v-for="(s, i) in steps" :key="i" :class="stepState[i]">
              <span class="dot"></span>{{ s }}
            </li>
          </ul>
          <div class="muted small">上传文件后开始转换。</div>
        </div>
      </section>

      <!-- middle: doc preview -->
      <section class="card preview">
        <div class="card-head">📄 doc preview</div>
        <div class="preview-body">
          <embed v-if="previewType === 'pdf'" :src="previewUrl" type="application/pdf" class="pdf" />
          <img v-else-if="previewType === 'image'" :src="previewUrl" class="img-preview" :alt="fileName" />
          <div v-else-if="fileName" class="ph-named">{{ fileName }}</div>
          <div v-else class="ph">📄</div>
        </div>
      </section>

      <!-- right: output -->
      <section class="card output">
        <div class="tabs">
          <button :class="{ on: tab === 'render' }" @click="tab = 'render'">Markdown 渲染</button>
          <button :class="{ on: tab === 'text' }" @click="tab = 'text'">Markdown 文本</button>
          <button :class="{ on: tab === 'json' }" @click="tab = 'json'">JSON 内容列表</button>
          <span class="spacer"></span>
          <button class="copy" title="复制"
                  @click="copy(tab === 'json' ? jsonText : md)">⧉</button>
        </div>
        <div class="out-body">
          <div v-if="tab === 'render'" class="md" v-html="renderedMd"></div>
          <pre v-else-if="tab === 'text'" class="raw">{{ md }}</pre>
          <pre v-else class="raw">{{ jsonText }}</pre>
        </div>
      </section>
    </main>
  </div>
</template>
