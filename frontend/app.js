// All inference and validation runs on the Python backend. No browser API keys.
const el = id => document.getElementById(id);
let lastTask = null;
let busy = false;
let latest = null;
const setText = (id, text) => { if (el(id)) el(id).textContent = text; };
const words = text => (text.match(/\b[\w'-]+\b/g) || []).length;

function showPending(status, message) {
  const output=el('userOutputDisplay');
  output.textContent=message || '正在精雕，完成后将一次性呈现最终文本与修改记录…';
  output.dataset.rawText='';
  el('userModSummaryCard').hidden=true;
  el('userCheckAiResultBar').hidden=true;
  el('userCheckAiResultBar').style.display='none';
  el('userCopyBtn').disabled=true;
  el('toggleAddedChangeHighlightBtn').disabled=true;
  setText('userOutputWordsLabel','');
  setText('userModCountBadge','');
  setText('workflowStateBadge',`任务状态：${status}`);
}

function showResult(task) {
  latest = task;
  const r = task.result || {};
  if (!['completed','needs_review'].includes(task.status)) {
    const active=task.status==='running';
    showPending(task.status,active ? undefined : '任务尚未完成，未展示中间稿。可恢复任务后继续。');
    const events=r.events || [];
    setText('hostedEngineStatusLabel',task.error || `${task.status} · ${events.at(-1)?.message || '处理中'}`);
    setText('agentLogConsole',events.map(e=>`${e.stage}: ${e.message}`).join('\n'));
    el('resumeTaskBtn').hidden=!['failed','interrupted'].includes(task.status);
    return;
  }
  el('userModSummaryCard').hidden=false;
  el('userCheckAiResultBar').hidden=false;
  el('userCheckAiResultBar').style.display='';
  el('userCopyBtn').disabled=false;
  el('toggleAddedChangeHighlightBtn').disabled=false;
  const text = r.best || r.draft || '';
  const output = el('userOutputDisplay');
  proseHighlight.render(output, r.source || '', text);
  output.dataset.rawText = text;
  setText('userOutputWordsLabel', `${words(text)} words`);
  setText('userCheckAiScoreText', r.passed ? '规则与语义审查通过' : '等待检查 / 需要人工复核');
  setText('userCheckAiFidelityText', '规则检查与模型审查提供辅助证据，不保证语义完全一致');
  const bar = el('userCheckAiResultBar');
  bar.className = 'mx-5 mb-2 px-3.5 py-2 rounded-xl border bg-slate-50 border-slate-200 flex flex-col items-start gap-2 text-xs';
  setText('userCheckAiFidelityText', `流程：${task.status} · 规则：${r.rules ? (r.rules.passed ? '通过' : '未通过') : '待检查'} · 语义审查：${r.review?.passed == null ? '无合格改稿，保留原稿' : (r.review.passed ? '通过' : '未通过')}（模型辅助判断）`);
  const events = r.events || [];
  const log = el('agentLogConsole');
  if (log) log.textContent = events.filter(e=> !/P\(AI\)|分类阈值|分类分数/i.test(e.message || '')).map(e => `${e.stage}: ${e.message}`).join('\n') + '\n' + (r.usage_log || []).map(u=>`${u.stage}: 输入 ${u.usage?.input_tokens ?? '?'} / 输出 ${u.usage?.output_tokens ?? '?'} / 推理 ${u.usage?.output_token_details?.reasoning ?? '?'} tokens`).join('\n');
  setText('workflowStateBadge', `任务状态：${task.status}`);
  const summary = el('userModListContainer');
  summary.replaceChildren();
  function appendEdits(container, edits) {
  for (const [index,item] of (edits || []).entries()) {
    const div = document.createElement('div');
    div.className = 'p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-2';
    const heading=document.createElement('div');heading.className='font-semibold text-slate-800';
    heading.textContent=`修改点 #${index+1}：${item.action_taken || '以下为原稿与成稿的实际差异。'}`;
    const columns=document.createElement('div');columns.className='grid grid-cols-1 sm:grid-cols-2 gap-2.5';
    for (const [label,value,classes] of [
      ['修改前',item.before,'bg-rose-50 border-rose-200'],
      ['修改后',item.after,'bg-emerald-50 border-emerald-200']]) {
      const pane=document.createElement('div');pane.className=`p-2.5 rounded-lg border text-slate-700 leading-relaxed whitespace-pre-wrap ${classes}`;
      const title=document.createElement('strong');title.textContent=label+'：';
      pane.append(title,document.createTextNode(value || (label==='修改前'?'（新增内容，原稿无对应句）':'（已删除）')));
      columns.append(pane);
    }
    div.append(heading,columns);
    container.append(div);
  }
  }
  const selectedNote=document.createElement('p');
  selectedNote.className='font-semibold text-slate-700';
  selectedNote.textContent='以下仅对照提交时的原文与最终成稿；空白符以 ␠、\\n、\\t 表示。';
  summary.append(selectedNote);
  appendEdits(summary,r.edits);
  const edits=r.edits || [];
  setText('userModCountBadge', `${edits.length} 组对照 · ${edits.reduce((n,e)=>n+(e.change_count || 1),0)} 项差异`);
  setText('userModLevelTag', '原文 → 最终成稿');
  if (!edits.length) { const note=document.createElement('p'); note.textContent='最终成稿与原文完全一致，没有可展示的改动。'; summary.append(note); }
  const stopLabels = {protocol_errors:'模型连续返回无效动作，已停止以避免继续消耗',target_reached:'编辑完成，保真通过',no_progress:'已结束本次优化',max_rounds:'达到修改轮次上限',token_budget:'达到token预算',agent_budget:'达到决策预算',score_unavailable:'本次编辑已结束'};
  stopLabels.output_limit='单次推理/输出达到上限，未接受截断稿';
  setText('hostedEngineStatusLabel', task.error || `${task.status} · ${stopLabels[r.stop_reason] || events.at(-1)?.message || '执行中'}`);
  el('resumeTaskBtn').hidden = !['failed', 'interrupted'].includes(task.status);
}

async function jsonFetch(url, options) {
  const res = await fetch(url, options);
  const body = await res.json();
  if (!res.ok) throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail));
  return body;
}

async function watch(id) {
  lastTask = id;
  localStorage.setItem('linguasculpt_task_id', id);
  while (true) {
    const task = await jsonFetch(`/api/tasks/${id}`);
    if (lastTask!==id) return;
    showResult(task);
    if (task.status !== 'running') break;
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
}

window.triggerUserSculpt = async () => {
  if (busy) return;
  busy = true;
  el('userHumanizeActionBtn').disabled = true;
  setText('userHumanizeBtnText', 'Agent 执行中…');
  lastTask=null;
  showPending('running');
  try {
    const task = await jsonFetch('/api/tasks', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source: el('userSourceTextarea').value,
        style: `${el('userWritingLevelSelect').value}; cadence=${el('userToneSelect').value}`,
        depth: Number(el('userRewriteSlider').value),
        terms: el('userCustomStyleToggle').checked ? el('userEntityLockInput').value.split(/[,，]/).map(s => s.trim()).filter(Boolean) : []})
    });
    await watch(task.id);
  } catch (error) {
    showPending('failed','任务未完成，请检查错误提示后重试。');
    setText('hostedEngineStatusLabel', error.message);
  } finally {
    busy = false;
    el('userHumanizeActionBtn').disabled = false;
    setText('userHumanizeBtnText', 'Sculpt Prose · 开始精雕');
  }
};

window.togglePortalView = () => {
  el('adminPortalView').classList.toggle('hidden');
  el('userPortalView').classList.toggle('hidden');
};

document.addEventListener('DOMContentLoaded', async () => {
  // Remove obsolete demo/configuration surfaces; retain the original studio UI.
  el('adminPortalView').innerHTML = '<div class="max-w-4xl mx-auto p-8 pt-20"><h2 class="text-xl mb-4">Agent 执行记录</h2><p id="workflowStateBadge"></p><pre id="agentLogConsole" class="whitespace-pre-wrap mt-4"></pre></div>';
  el('apiModalOverlay')?.remove();
  setText('switchAdminViewBtn', '执行记录 / 返回创作端');
  setText('userCheckAiScoreText', '尚未执行检查');
  setText('userCheckAiFidelityText', '术语、数字、引用检查 + 独立语义审查');
  const button = document.createElement('button');
  button.id = 'resumeTaskBtn'; button.hidden = true;
  button.textContent = '恢复中断任务';
  button.className = 'px-3 py-2 rounded bg-amber-100 text-amber-900';
  el('userHumanizeActionBtn').parentElement.append(button);
  const exportBtn = document.createElement('button');
  exportBtn.textContent = '导出任务记录';
  exportBtn.className = 'px-3 py-2 rounded border text-slate-700';
  exportBtn.onclick = () => { if (lastTask) window.location.assign(`/api/tasks/${lastTask}/export`); };
  button.parentElement.append(exportBtn);
  button.onclick = async () => {
    if (busy || !lastTask) return;
    busy = true;
    try { await jsonFetch(`/api/tasks/${lastTask}/resume`, {method:'POST'}); await watch(lastTask); }
    catch (err) { setText('hostedEngineStatusLabel', err.message); }
    finally { busy = false; }
  };
  el('userSourceTextarea').oninput = e => setText('userInputWordsLabel', `${words(e.target.value)} words`);
  el('userClearBtn').onclick = () => { el('userSourceTextarea').value = ''; setText('userInputWordsLabel', '0 words'); };
  el('userCustomStyleToggle').onchange = e => el('userCustomLockWrapper').classList.toggle('hidden', !e.target.checked);
  el('userCopyBtn').onclick = () => navigator.clipboard.writeText(el('userOutputDisplay').dataset.rawText || '');
  el('userCheckAiBtn').onclick = () => el('userModSummaryCard').classList.toggle('hidden');
  el('toggleUserModDetailsBtn').onclick = () => el('userModSummaryCard').classList.toggle('hidden');
  const highlightButton=el('toggleAddedChangeHighlightBtn');
  highlightButton.textContent='隐藏修改高亮';
  highlightButton.setAttribute('aria-pressed','true');
  highlightButton.title='绿色标出新增或替换的词句；删除内容见下方修改对照';
  highlightButton.onclick=()=>{
    const hidden=el('userOutputDisplay').classList.toggle('hide-user-changes');
    highlightButton.textContent=hidden ? '显示修改高亮' : '隐藏修改高亮';
    highlightButton.setAttribute('aria-pressed',String(!hidden));
  };
  document.querySelectorAll('.user-level-lbl').forEach(b => b.onclick = () => { el('userRewriteSlider').value = b.dataset.level; });
  try {
    const health = await jsonFetch('/api/health');
    setText('hostedEngineStatusLabel', health.hosted_llm_ready ? `后端就绪 · ${health.default_model}` : '请在 .env 配置 DeepSeek 密钥并重启');
    const samples = await jsonFetch('/api/samples');
    const select = document.createElement('select');
    select.className = 'text-xs border rounded p-2 max-w-full';
    select.add(new Option('选择英文测试样例（原作与自编案例）', ''));
    samples.forEach((s, i) => select.add(new Option(s.title, i)));
    select.onchange = () => {
      if (select.value === '') return;
      el('userSourceTextarea').value = samples[Number(select.value)].source;
      el('userSourceTextarea').dispatchEvent(new Event('input'));
    };
    el('userSourceTextarea').before(select);
    const saved = localStorage.getItem('linguasculpt_task_id');
    if (saved) await watch(saved);
  } catch (err) { setText('hostedEngineStatusLabel', err.message); }
});
