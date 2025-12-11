from __future__ import annotations

import asyncio
import base64
import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx
import astrbot.api.message_components as Comp
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api import logger, AstrBotConfig

URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.I)
VIDEO_URL_RE = re.compile(r"https?://[^\s\"'<>]+?\.(mp4|webm|mov)(\?[^\s\"']*)?$", re.I)
TAG_VIDEO_SRC_RE = re.compile(r"<video[^>]*\s+src=['\"]([^'\"]+)['\"]", re.I)
TAG_SOURCE_SRC_RE = re.compile(r"<source[^>]*\s+src=['\"]([^'\"]+)['\"]", re.I)


def _join_abs(base: str, url: str) -> str:
    """拼接URL"""
    if not url:
        return url
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return urljoin(base.rstrip("/") + "/", url.lstrip("/"))


def _strip_cmd(text: str, which: str) -> str:
    """剥触发词"""
    if not text:
        return ""
    s = text
    prefix = f"/{which}".lower()
    if s.lower().startswith(prefix):
        return s[len(prefix):]
    return s


def _collect_image_urls(chain: List[Comp.Base]) -> List[str]:
    """收集图片"""
    out: List[str] = []
    seen: set[str] = set()

    def walk(cs: List[Comp.Base]):
        for seg in cs or []:
            if isinstance(seg, Comp.Image):
                u = getattr(seg, "url", None) or getattr(seg, "file", None)
                if isinstance(u, str) and u.startswith("http") and u not in seen:
                    seen.add(u)
                    out.append(u)
            elif isinstance(seg, Comp.Reply):
                sub = getattr(seg, "chain", None)
                if isinstance(sub, list):
                    walk(sub)
            elif hasattr(seg, "nodes"):
                for node in getattr(seg, "nodes", []) or []:
                    content = getattr(node, "content", []) or []
                    if isinstance(content, list):
                        walk(content)

    walk(chain or [])
    return out[:2]


def _json_frag(x: Any, n: int = 360) -> str:
    """报错json"""
    try:
        s = json.dumps(x, ensure_ascii=False)
        return s if len(s) <= n else s[:n] + " ..."
    except Exception:
        return str(x)[:n]


async def _req(
    method: str,
    url: str,
    *,
    headers: Dict[str, str] | None = None,
    json_body: Dict[str, Any] | None = None,
    timeout: int = 900,
    retries: int = 2,
    backoff: float = 2.0,
) -> httpx.Response:
    """HTTP 请求封装"""
    last: Optional[Exception] = None
    for k in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as cli:
                if method.upper() == "GET":
                    r = await cli.get(url, headers=headers or {})
                else:
                    r = await cli.post(url, headers=headers or {}, json=json_body or {})
                if 500 <= r.status_code < 600 and k < retries:
                    await asyncio.sleep(backoff * (2**k))
                    continue
                r.raise_for_status()
                return r
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.PoolTimeout) as e:
            last = e
            if k < retries:
                await asyncio.sleep(backoff * (2**k))
                continue
            raise
        except httpx.HTTPStatusError:
            raise
    raise last or RuntimeError("network error")


async def _get_json(
    url: str,
    headers: Dict[str, str] | None,
    timeout: int,
    retries: int,
    backoff: float,
) -> Dict[str, Any]:
    """GET 请求并解析 JSON 响应"""
    r = await _req("GET", url, headers=headers, timeout=timeout, retries=retries, backoff=backoff)
    return r.json()


async def _post_json(
    url: str,
    headers: Dict[str, str] | None,
    body: Dict[str, Any],
    timeout: int,
    retries: int,
    backoff: float,
) -> Dict[str, Any]:
    """POST JSON 请求并解析 JSON 响应。"""
    r = await _req("POST", url, headers=headers, json_body=body, timeout=timeout, retries=retries, backoff=backoff)
    return r.json()


async def _download_b64(url: str, timeout: int, retries: int, backoff: float) -> Optional[str]:
    """下载base64"""
    try:
        r = await _req("GET", url, timeout=timeout, retries=retries, backoff=backoff)
        return base64.b64encode(r.content).decode("ascii")
    except Exception as e:
        logger.warning("[img] 下载失败：%s", e)
        return None


def _extract_any_url(obj: Any) -> Optional[str]:
    """拿视频url"""
    if isinstance(obj, str):
        m = TAG_VIDEO_SRC_RE.search(obj)
        if m:
            return m.group(1)
        m = TAG_SOURCE_SRC_RE.search(obj)
        if m:
            return m.group(1)
        m = VIDEO_URL_RE.search(obj)
        if m:
            return m.group(0)
        m = URL_RE.search(obj)
        if m:
            return m.group(0)
        return None

    if isinstance(obj, list):
        for it in obj:
            u = _extract_any_url(it)
            if u:
                return u
        return None

    if isinstance(obj, dict):
        for k in ("video_url", "videoUrl", "url"):
            v = obj.get(k)
            if isinstance(v, str) and v:
                return v

        ch = obj.get("choices")
        if isinstance(ch, list) and ch:
            msg = (ch[0] or {}).get("message", {}) or {}
            c = msg.get("content")
            u = _extract_any_url(c)
            if u:
                return u

        for key in ("data", "result", "results"):
            if key in obj:
                u = _extract_any_url(obj[key])
                if u:
                    return u

        for v in obj.values():
            u = _extract_any_url(v)
            if u:
                return u

    return None


def _extract_task_id(obj: Any) -> Optional[str]:
    """提取任务 ID"""
    keys = ("task_id", "taskId", "id", "historyId", "job_id", "jobId", "request_id")

    if isinstance(obj, dict):
        for k in keys:
            v = obj.get(k)
            if isinstance(v, (str, int)) and str(v):
                return str(v)

        for key in ("data", "result", "results", "payload"):
            r = _extract_task_id(obj.get(key))
            if r:
                return r

        for v in obj.values():
            r = _extract_task_id(v)
            if r:
                return r

    elif isinstance(obj, list):
        for it in obj:
            r = _extract_task_id(it)
            if r:
                return r

    elif isinstance(obj, (str, int)):
        s = str(obj).strip()
        if s:
            return s

    return None


class GrokProvider:
    """grok2api"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int, retries: int, backoff: float):
        self.base, self.key, self.model = base_url.rstrip("/"), api_key, model
        self.timeout, self.retries, self.backoff = timeout, retries, backoff

    def _hdr(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    async def generate(self, prompt: str, image_urls: List[str]) -> str:
        if not image_urls:
            raise RuntimeError("Grok 需要【图片 + 文本】才能生成视频，请附图或引用带图消息。")

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for u in image_urls[:2]:
            content.append({"type": "image_url", "image_url": {"url": u}})

            b64 = await _download_b64(
                u,
                timeout=min(60, self.timeout),
                retries=self.retries,
                backoff=self.backoff,
            )
            if b64:
                mime = "image/jpeg"
                lu = u.lower()
                if lu.endswith(".png"):
                    mime = "image/png"
                elif lu.endswith(".webp"):
                    mime = "image/webp"
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                )

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }
        data = await _post_json(
            f"{self.base}/v1/chat/completions",
            self._hdr(),
            body,
            self.timeout,
            self.retries,
            self.backoff,
        )

        u = _extract_any_url(data)
        if not u:
            dump = json.dumps(data, ensure_ascii=False)
            m = TAG_VIDEO_SRC_RE.search(dump) or TAG_SOURCE_SRC_RE.search(dump) or VIDEO_URL_RE.search(dump)
            if m:
                if m.re in (TAG_VIDEO_SRC_RE, TAG_SOURCE_SRC_RE):
                    u = m.group(1)
                else:
                    u = m.group(0)

        if not u:
            raise RuntimeError(f"Grok 未解析到视频链接；响应片段：{_json_frag(data)}")

        return _join_abs(self.base, u)


class SoraProvider:
    """Sora2api"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int, retries: int, backoff: float):
        self.base, self.key, self.model = base_url.rstrip("/"), api_key, model
        self.timeout, self.retries, self.backoff = timeout, retries, backoff

    async def generate(self, prompt: str, image_urls: List[str]) -> str:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if image_urls:
            b64 = await _download_b64(
                image_urls[0],
                timeout=min(60, self.timeout),
                retries=self.retries,
                backoff=self.backoff,
            )
            if b64:
                body["image"] = b64
            body["image_urls"] = image_urls[:2]
            body["imageUrls"] = image_urls[:2]

        data = await _post_json(
            f"{self.base}/v1/chat/completions",
            headers,
            body,
            self.timeout,
            self.retries,
            self.backoff,
        )
        u = _extract_any_url(data)
        if not u:
            raise RuntimeError(f"Sora 未解析到视频链接；响应片段：{_json_frag(data)}")
        return _join_abs(self.base, u)


class JimengProvider:
    """jimeng2api"""

    def __init__(
        self,
        base_url: str,
        session_token: str,
        model: str,
        timeout: int,
        poll_interval: int,
        retries: int,
        backoff: float,
        resolution: str = "",
    ):
        self.base, self.token, self.model = base_url.rstrip("/"), session_token, model
        self.timeout, self.poll, self.retries, self.backoff = timeout, poll_interval, retries, backoff
        self.resolution = resolution

    async def generate(self, prompt: str, image_urls: List[str]) -> str:
        if not self.token:
            raise RuntimeError("未配置 jimeng.session_token")

        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        body: Dict[str, Any] = {"model": self.model, "prompt": prompt}
        if self.resolution:
            body["resolution"] = self.resolution
        if image_urls:
            body["file_paths"] = image_urls[:2]

        created = await _post_json(
            f"{self.base}/v1/videos/generations",
            headers,
            body,
            self.timeout,
            self.retries,
            self.backoff,
        )

        if isinstance(created, dict) and "code" in created and created["code"] not in (0, "0", None):
            raise RuntimeError(f"即梦创建失败：code={created['code']}，{created.get('message') or '未知错误'}")

        u0 = _extract_any_url(created)
        if u0:
            return _join_abs(self.base, u0)

        task_id = _extract_task_id(created)
        if not task_id:
            raise RuntimeError(f"即梦创建任务成功但未返回 task_id；响应片段：{_json_frag(created)}")

        loop = asyncio.get_event_loop()
        deadline = loop.time() + self.timeout
        last_status: Optional[str] = None

        while True:
            if loop.time() > deadline:
                raise RuntimeError("即梦查询超时")

            try:
                s = await _get_json(
                    f"{self.base}/v1/videos/{task_id}",
                    {"Authorization": f"Bearer {self.token}"},
                    min(60, self.timeout),
                    self.retries,
                    self.backoff,
                )
            except Exception as e:
                logger.warning("[jm] 查询异常：%s", e)
                await asyncio.sleep(self.poll)
                continue

            status = (s.get("status") or (s.get("data") or {}).get("status") or "").lower()
            if status and status != last_status:
                last_status = status
                logger.info("[jm] %s -> %s", task_id, status)

            if status in ("succeeded", "completed", "success", "done", "finished"):
                u = (
                    s.get("url")
                    or (s.get("data") or {}).get("video_url")
                    or (s.get("data") or {}).get("url")
                    or _extract_any_url(s)
                )
                if not u:
                    raise RuntimeError("即梦任务完成，但未返回视频 URL")
                return _join_abs(self.base, u)

            if status in ("failed", "error"):
                raise RuntimeError(f"即梦生成失败：{s.get('error') or s.get('message') or '未知错误'}")

            await asyncio.sleep(self.poll)


class VideoGen(Star):
    """
    AstrBot 插件：统一封装 sora / grok / 即梦三种视频生成指令。

    指令：
    - /sora <提示词>
    - /grok <提示词>（必须图 + 文）
    - /jm   <提示词>
    - /v    <提示词>（并发调用 sora / grok / jm）
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        cfg = config or {}

        self.poll = int(cfg.get("poll_interval", 6))

        s = cfg.get("sora", {}) or {}
        j = cfg.get("jimeng", {}) or {}
        g = cfg.get("grok", {}) or {}
        net = cfg.get("network", {}) or {}

        retries = int(net.get("max_retries", 2))
        backoff = float(net.get("retry_backoff", 2))

        self.sora = SoraProvider(
            base_url=s.get("base_url", "http://127.0.0.1:8000"),
            api_key=s.get("api_key", "123456"),
            model=s.get("model", "sora-video"),
            timeout=int(s.get("timeout_seconds", 640)),
            retries=retries,
            backoff=backoff,
        )

        self.grok = GrokProvider(
            base_url=g.get("base_url", "http://127.0.0.1:8000"),
            api_key=g.get("api_key", "123456"),
            model=g.get("model", "grok-imagine-0.9"),
            timeout=int(g.get("timeout_seconds", 180)),
            retries=retries,
            backoff=backoff,
        )

        self.jm = JimengProvider(
            base_url=j.get("base_url", "http://127.0.0.1:5100"),
            session_token=j.get("session_token", "us-123456"),
            model=j.get("model", "jimeng-video-3.0"),
            timeout=int(j.get("timeout_seconds", 640)),
            poll_interval=self.poll,
            retries=retries,
            backoff=backoff,
            resolution=j.get("resolution", "1080p") or "",
        )

    @filter.command("sora")
    async def sora_cmd(self, event: AstrMessageEvent):
        async for o in self._handle(event, "sora"):
            yield o

    @filter.command("grok")
    async def grok_cmd(self, event: AstrMessageEvent):
        async for o in self._handle(event, "grok"):
            yield o

    @filter.command("jm")
    async def jm_cmd(self, event: AstrMessageEvent):
        async for o in self._handle(event, "jm"):
            yield o

    @filter.command("v")
    async def v_cmd(self, event: AstrMessageEvent):
        async for o in self._handle(event, "v"):
            yield o

    async def _send_video(self, url: str, prompt: str, which: str, event: AstrMessageEvent):
        """封装发送视频组件"""
        comp = Comp.Video.fromURL(url)
        yield event.chain_result([comp, Comp.Plain(f"\n[{which}] {prompt}")])

    async def _handle(self, event: AstrMessageEvent, which: str):
        try:
            event.should_call_llm(False)
        except Exception:
            try:
                event.call_llm = False
            except Exception:
                pass

        # 解析提示词
        raw = event.message_str or ""
        prompt = _strip_cmd(raw, which)
        if not prompt:
            if which == "v":
                yield event.plain_result("用法：/v <提示词>（同时调用 sora/jm/grok，携带或引用图片自动图生视频）")
            else:
                yield event.plain_result(f"用法：/{which} <提示词>（携带或引用图片自动图生视频）")
            return

        # 获取消息
        msg_obj = getattr(event, "message_obj", None)
        comps = event.get_messages() or (msg_obj.message if msg_obj else [])
        img_urls = _collect_image_urls(comps or [])

        # /grok 提示
        if which == "grok" and not img_urls:
            yield event.plain_result("❌ grok 需要【图片 + 文本】才能生成视频，请附图或引用带图消息。")
            return

        # /v 三合一并发
        if which == "v":
            yield event.plain_result(f"✅ 已提交任务：{prompt}（图：{len(img_urls)}），将同时调用 sora/jm/grok")

            providers: list[tuple[str, Any]] = [("sora", self.sora), ("jm", self.jm)]
            if img_urls:
                providers.append(("grok", self.grok))
            else:
                yield event.plain_result("grok 需要【图片 + 文本】才能生成视频，本次 /v 跳过 grok。")

            if not providers:
                return

            async def _run_one(name: str, prov: Any):
                try:
                    url = await prov.generate(prompt, img_urls)
                    return name, url
                except Exception:
                    logger.exception("[video-gen] %s generate failed", name)
                    return name, None

            tasks = [
                _run_one(name, prov)
                for name, prov in providers
            ]

            # 先好先发
            for fut in asyncio.as_completed(tasks):
                name, url = await fut
                if not url:
                    yield event.plain_result(f"{name} 生成失败喵")
                    continue

                async for node in self._send_video(url, prompt, name, event):
                    yield node

            return

        # /sora /grok /jm 
        yield event.plain_result(f"✅ 已提交 {which} 任务：{prompt}（图：{len(img_urls)}）")

        try:
            if which == "sora":
                url = await self.sora.generate(prompt, img_urls)
            elif which == "jm":
                url = await self.jm.generate(prompt, img_urls)
            else:
                url = await self.grok.generate(prompt, img_urls)

            async for node in self._send_video(url, prompt, which, event):
                yield node

        except Exception:
            logger.exception("[video-gen] %s generate failed", which)
            yield event.plain_result(f"{which} 生成失败喵")

    async def terminate(self):
        pass