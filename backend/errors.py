def public_error(exc):
    status = getattr(exc, 'status_code', None)
    explanations = {
        400:'请求参数不被模型接口接受。请核对模型 ID、上下文长度和接口兼容性。',
        401:'密钥无效或已失效。请更新 .env 后重启后端。',
        402:'账户余额不足。请检查当前模型服务商的账户余额。',
        403:'模型访问被拒绝。请检查账户权限或地区限制。',
        404:'模型或接口路径不存在。请核对 DEEPSEEK_MODEL 与 DEEPSEEK_BASE_URL。',
        422:'请求参数格式不被接口接受。',
        429:'模型接口限流或额度已用尽。请稍后恢复任务，并检查账户配额。',
        500:'模型服务内部异常，请稍后恢复任务。',
        502:'模型网关异常，请稍后恢复任务。',
        503:'模型服务暂不可用，请稍后恢复任务。',
        504:'模型服务响应超时，请稍后恢复任务。',
    }
    if status:
        return f'HTTP {status}：' + explanations.get(status,'模型接口拒绝请求，请检查服务商控制台。')
    name=type(exc).__name__
    if 'Timeout' in name:
        return '模型请求超时。已保存检查点，可稍后恢复任务。'
    if 'Connection' in name:
        return '无法连接模型接口。请检查网络、代理和接口地址。'
    return f'{name}：工作流执行失败，已保存可用检查点。'
