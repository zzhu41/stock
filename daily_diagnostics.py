"""Bounded, redacted diagnostics for the cron subprocess boundary (Python3.6+)."""
import json
from pathlib import Path
import re
import traceback

ROOT = Path(__file__).resolve().parent
MARKER = 'STOCK_DIAGNOSTIC '


def safe_message(value, limit=500):
    text = str(value)[:8000]
    text = re.sub(r'https?://[^\s\'"<>]+', '[URL]', text, flags=re.I)
    text = re.sub(r'(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+', '[credential]', text)
    text = re.sub(r'''(?ix)(["']?(?:access[_-]?token|session[_-]?webhook|secret|password|authorization|token|sign)["']?\s*[:=]\s*)(?:["'][^"']*["']|[^\s,;}&]+)''',
                  r'\1[redacted]', text)
    text = re.sub(r'\b(?:SEC[A-Za-z0-9]+|[A-Za-z0-9_-]{32,})\b', '[redacted]', text)
    text = ' '.join(text.split())
    return text[:limit]


def clean_detail(value, default_phase='generation'):
    """Do not preserve arbitrary child JSON keys or raw exception dictionaries."""
    if not isinstance(value, dict):
        value = {}
    result = dict(phase=safe_message(value.get('phase', default_phase), 60),
                  error_type=safe_message(value.get('error_type', 'ChildProcessError'), 80),
                  message=safe_message(value.get('message', '子进程未返回可核验的错误说明')))
    frames = value.get('frames', [])
    result['frames'] = [dict(file=safe_message(v.get('file', ''), 160),
                            line=v['line'], function=safe_message(v.get('function', ''), 80))
                        for v in frames[-6:] if isinstance(v, dict)
                        and isinstance(v.get('line'), int) and v['line'] > 0] if isinstance(frames, list) else []
    causes = value.get('causes', [])
    if isinstance(causes, list) and causes:
        # One level only; no recursion through arbitrary child objects.
        result['causes'] = [clean_detail({k:v for k,v in item.items() if k != 'causes'}, default_phase)
                            for item in causes[:8] if isinstance(item, dict)]
    return result


def exception_detail(exc, phase='generation'):
    if isinstance(getattr(exc, 'diagnostic', None), dict):
        return clean_detail(exc.diagnostic, phase)
    frames = []
    for frame in traceback.extract_tb(exc.__traceback__)[-6:]:
        path = Path(frame.filename)
        try:
            name = str(path.resolve().relative_to(ROOT))
        except ValueError:
            name = path.name
        frames.append(dict(file=name,line=frame.lineno,function=frame.name))
    return clean_detail(dict(phase=phase,error_type=type(exc).__name__,message=str(exc),frames=frames),phase)


class DiagnosticError(RuntimeError):
    def __init__(self, diagnostic):
        self.diagnostic = clean_detail(diagnostic)
        super().__init__(self.diagnostic['error_type']+': '+self.diagnostic['message'])


class GenerationInputError(ValueError):
    """Preserve the generator's existing invalid-input exception contract."""
    def __init__(self, diagnostic):
        self.diagnostic = clean_detail(diagnostic)
        super().__init__(self.diagnostic['message'])


def child_failure(stdout, stderr, stage, returncode):
    for line in (stderr or '').splitlines()[-30:][::-1]:
        if line.startswith(MARKER):
            try:
                return clean_detail(json.loads(line[len(MARKER):]), stage)
            except (TypeError, ValueError):
                pass
    try:
        value = json.loads(stdout or '')
        if isinstance(value, dict):
            if isinstance(value.get('diagnostic'), dict):
                return clean_detail(value['diagnostic'], stage)
            if isinstance(value.get('diagnostics'), list) and value['diagnostics']:
                return clean_detail(dict(phase=stage,error_type='PreparationError',
                    message='三版本准备失败，请核对数据原因',causes=value['diagnostics']),stage)
    except (TypeError, ValueError):
        pass
    # For imports/old workers, retain only a traceback exception class, not
    # arbitrary stdout/stderr which might contain credentials or response bodies.
    matches = re.findall(r'^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\s*:',stderr or '',re.M)
    return clean_detail(dict(phase=stage,error_type=matches[-1] if matches else 'ChildProcessError',
        message='子进程退出码%s；未收到结构化诊断' % returncode),stage)


def emit(exc, phase, stream):
    stream.write(MARKER+json.dumps(exception_detail(exc,phase),ensure_ascii=False,allow_nan=False)+'\n')
    stream.flush()
