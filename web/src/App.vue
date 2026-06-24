<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { marked } from 'marked'
import katex from 'katex'

const file = ref(null)
const fileName = ref('')
const dragOver = ref(false)
// Source-aligned backends; the server reports which are actually available.
const ALL_BACKENDS = ['hybrid-engine', 'pipeline', 'vlm']
const backend = ref('hybrid-engine')
const backends = ref(ALL_BACKENDS)
const maxPages = ref(1000)
const showAdvanced = ref(false)
const converting = ref(false)
const error = ref('')
const tab = ref('render')        // render | text | json
const md = ref('')
const contentList = ref(null)
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
})
const backendHint = computed(() => ({
  'hybrid-engine': '原生流水线 + VLM 图表理解（混合）',
  'pipeline': '原生 ONNX 流水线（版面/OCR/公式/表格，最快）',
  'vlm': 'Qwen2-VL 视觉大模型（整页理解，含图像）',
}[backend.value] || '本地原生解析引擎'))

const previewUrl = ref('')
function onPick(e) { setFile(e.target.files?.[0]) }
function onDrop(e) { dragOver.value = false; setFile(e.dataTransfer.files?.[0]) }
function setFile(f) {
  if (!f) return
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  file.value = f
  fileName.value = f.name
  previewUrl.value = /\.pdf$/i.test(f.name) ? URL.createObjectURL(f) : ''
}
function clearAll() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  file.value = null; fileName.value = ''; previewUrl.value = ''
  md.value = ''; contentList.value = null; error.value = ''; resetSteps()
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
const renderedMd = computed(() => {
  if (!md.value) return ''
  try { return renderMath(marked.parse(md.value, { breaks: true, gfm: true })) } catch { return '' }
})
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
          <div v-if="!fileName">将文件拖放到此处<br /><span class="muted">- 或 -</span><br />点击上传</div>
          <div v-else class="picked">{{ fileName }}</div>
          <input ref="fi" type="file" hidden accept=".pdf,.png,.jpg,.jpeg,.docx,.pptx,.xlsx" @change="onPick" />
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
        <div v-if="showAdvanced" class="adv-body muted small">原生引擎参数（公式/表格识别默认开启）。</div>

        <div class="btn-row">
          <button class="primary" :disabled="!file || converting" @click="convert">
            {{ converting ? '转换中…' : '转换' }}
          </button>
          <button class="ghost" @click="clearAll">清除</button>
        </div>

        <div class="label section">转换结果</div>
        <div v-if="error" class="err">{{ error }}</div>

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
          <embed v-if="previewUrl" :src="previewUrl" type="application/pdf" class="pdf" />
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
