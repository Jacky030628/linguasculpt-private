# 全 Vercel 部署（准备版，尚未云端验收）

GitHub 保存代码。Vercel 托管 FastAPI、静态页面和私有 Queues subscriber。
PostgreSQL 从 Vercel Marketplace 连接，例如 Neon；无需独立后端服务器。
这是托管数据库集成，不是 Vercel 本地磁盘。

## 账号及项目

1. 连接 GitHub 和 Vercel，选择私有 GitHub 仓库（或按用户指定公开）。
2. 仓库根目录使用此项目；不要上传 .env、data、baselines、.venv 或训练矩阵。
3. Vercel Import 仓库，采用 FastAPI 配置，Python 3.13，区域 iad1。
4. 在 Vercel Marketplace 连接 PostgreSQL，把连接串设置为 DATABASE_URL。
5. 配置 DEEPSEEK_API_KEY、DEEPSEEK_BASE_URL、DEEPSEEK_FAST_MODEL、DEEPSEEK_MODEL、DEEPSEEK_MAX_OUTPUT_TOKENS。
6. 本地 BACKEND_PROXY_SECRET 不用于此方案；不需要 BACKEND_URL。
7. Queues producer 和 subscriber 固定同一区域，SDK 从部署获取 OIDC 凭证。

## 执行结构

app.py -> backend.cloud_app。队列消息仅包含随机任务ID和步骤版本号。
每次投递只执行一个 LangGraph 节点：prepare / edit / verify / finish。
结果和下一步骤持久化到 PostgreSQL，不依赖本地线程或 SQLite。
重复投递用数据库行锁与步骤版本去重；数据库提交后再投递下一步骤。
若中间崩溃，旧消息重投递会补发下一步骤。模型调用成功但事务未提交的崩溃仍可能重复计费。
失败任务可由页面恢复；正常排队执行不依赖浏览器保持连接。
界面仍只在任务终止后显示最终文本和完整差异，不显示分数。

## 部署验收

先使用受保护的 Preview：检查 health、samples、创建任务、跨函数状态读取、
真实队列触发、三轮上限、失败恢复、最终高亮、最终差异和导出。
默认全局最多两个运行任务。初期使用 Vercel Deployment Protection 限定试用者；
尚未实现用户账号、用户隔离和公开运营的用量计费。
部署需要数据库与 Queues 在账号中可用，可能产生平台费用。

Python NLP/科学计算依赖较大：标准 Python 函数包上限500MB；超过需精简或启用
Vercel Large Functions（可用性按项目设置确认）。本地测试不能替代 Linux 云端构建。
maxDuration=300 用于 API；队列生成函数的执行时限也应在部署详情核对，必须覆盖一次模型请求。
先验收 Preview，再对外发布生产域名；未完成这些验收不能称为已部署成功。

## 本地开发

原 start.ps1 与 backend.main 继续使用本地 SQLite，不受此云端入口影响。
云端集成测试需 Vercel项目凭证和测试数据库；单元测试使用模型替身，不产生API费用。

官方文档：https://vercel.com/docs/queues/python-sdk 、 https://vercel.com/docs/functions/runtimes/python
