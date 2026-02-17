"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { connectDb, getApiBaseUrl, getDbSchema, getDbSchemaErd, type DbSchemaInfo } from "@/lib/api";

function sanitizeSvg(svgText: string): string {
  // Backward compatibility: older backend could return JSON-encoded string body.
  const normalized = (() => {
    const trimmed = svgText.trim();
    if ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
      try {
        const parsed = JSON.parse(trimmed);
        if (typeof parsed === "string") return parsed;
      } catch {
        // keep original text
      }
    }
    return svgText;
  })();

  const parser = new DOMParser();
  const doc = parser.parseFromString(normalized, "image/svg+xml");
  const root = doc.documentElement;

  if (!root || root.tagName.toLowerCase() !== "svg") {
    throw new Error("Некорректный SVG-ответ");
  }

  root.querySelectorAll("script, foreignObject, iframe, object, embed").forEach((node) => node.remove());

  const walk = doc.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
  let node = walk.currentNode as Element | null;
  while (node) {
    const attrs = Array.from(node.attributes);
    for (const attr of attrs) {
      const n = attr.name.toLowerCase();
      const v = attr.value.toLowerCase().trim();
      if (n.startsWith("on")) node.removeAttribute(attr.name);
      if ((n === "href" || n === "xlink:href") && v.startsWith("javascript:")) {
        node.removeAttribute(attr.name);
      }
    }
    node = walk.nextNode() as Element | null;
  }

  return new XMLSerializer().serializeToString(root);
}

function triggerDownload(content: string, fileName: string, contentType: string) {
  const blob = new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

type StoredDbProfile = {
  id: string;
  name: string;
  host: string;
  port: string;
  db: string;
  user: string;
  password: string;
};

const PROFILES_KEY = "t2sql_db_profiles_v1";
const ACTIVE_PROFILE_KEY = "t2sql_db_active_profile_v1";
const SESSION_ID_KEY = "t2sql_session_id";

function getOrCreateSessionId(): string {
  const existing = localStorage.getItem(SESSION_ID_KEY);
  if (existing) return existing;
  const id = crypto.randomUUID();
  localStorage.setItem(SESSION_ID_KEY, id);
  return id;
}

export default function DbSchemaPage() {
  const [schema, setSchema] = useState<DbSchemaInfo | null>(null);
  const [svg, setSvg] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [filterText, setFilterText] = useState("");
  const [focusedTable, setFocusedTable] = useState<string | null>(null);
  const svgHostRef = useRef<HTMLDivElement | null>(null);
  const originalViewBoxRef = useRef<string | null>(null);
  const currentViewBoxRef = useRef<string | null>(null);
  const svgRootRef = useRef<SVGSVGElement | null>(null);
  const panRef = useRef<{ x: number; y: number; vb: [number, number, number, number] } | null>(null);

  const apiBase = useMemo(() => getApiBaseUrl(), []);

  const parseViewBox = useCallback((vb: string | null): [number, number, number, number] | null => {
    if (!vb) return null;
    const nums = vb
      .split(/[,\s]+/)
      .map((x) => Number(x))
      .filter((x) => Number.isFinite(x));
    if (nums.length !== 4) return null;
    return [nums[0], nums[1], nums[2], nums[3]];
  }, []);

  const setViewBox = useCallback((vb: string) => {
    const root = svgRootRef.current;
    if (!root) return;
    root.setAttribute("viewBox", vb);
    currentViewBoxRef.current = vb;
  }, []);

  const zoomBy = useCallback((factor: number) => {
    let current = parseViewBox(currentViewBoxRef.current);
    if (!current && svgRootRef.current) current = parseViewBox(svgRootRef.current.getAttribute("viewBox"));
    if (!current && originalViewBoxRef.current) current = parseViewBox(originalViewBoxRef.current);
    if (!current) return;
    const [x, y, w, h] = current;
    const cx = x + w / 2;
    const cy = y + h / 2;
    const nw = Math.max(140, w * factor);
    const nh = Math.max(100, h * factor);
    const next = `${cx - nw / 2} ${cy - nh / 2} ${nw} ${nh}`;
    setViewBox(next);
  }, [parseViewBox, setViewBox]);

  const resetZoom = useCallback(() => {
    if (!originalViewBoxRef.current) return;
    setViewBox(originalViewBoxRef.current);
    setFocusedTable(null);
  }, [setViewBox]);

  const tryReconnectFromStoredProfile = useCallback(async () => {
    try {
      const raw = localStorage.getItem(PROFILES_KEY);
      const activeId = localStorage.getItem(ACTIVE_PROFILE_KEY);
      if (!raw || !activeId) return false;
      const profiles = JSON.parse(raw) as StoredDbProfile[];
      if (!Array.isArray(profiles) || profiles.length === 0) return false;
      const active = profiles.find((p) => p.id === activeId);
      if (!active) return false;

      await connectDb(getOrCreateSessionId(), {
        name: active.name || "default",
        host: active.host,
        port: active.port,
        db: active.db,
        user: active.user,
        password: active.password || "",
      });
      return true;
    } catch {
      return false;
    }
  }, []);

  const load = useCallback(async (refresh: boolean) => {
    setLoading(true);
    setError("");
    try {
      const [schemaInfo, svgRaw] = await Promise.all([getDbSchema(refresh), getDbSchemaErd("svg", refresh)]);
      setSchema(schemaInfo);
      setSvg(sanitizeSvg(svgRaw));
    } catch (e: any) {
      const message = String(e?.message ?? "Не удалось загрузить структуру");
      const likelyDbIssue = /db|пул|connection|подключ|timeout|closed|not initialized/i.test(message);
      if (likelyDbIssue) {
        const reconnected = await tryReconnectFromStoredProfile();
        if (reconnected) {
          try {
            const [schemaInfo, svgRaw] = await Promise.all([getDbSchema(refresh), getDbSchemaErd("svg", refresh)]);
            setSchema(schemaInfo);
            setSvg(sanitizeSvg(svgRaw));
            return;
          } catch {
            // fall through to show original message
          }
        }
      }
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [tryReconnectFromStoredProfile]);

  const refreshWithReconnect = useCallback(async () => {
    setLoading(true);
    setError("");
    setFocusedTable(null);
    originalViewBoxRef.current = null;
    currentViewBoxRef.current = null;
    try {
      await tryReconnectFromStoredProfile();
    } catch {
      // ignore reconnect errors here, load() below will surface details
    }
    await load(true);
  }, [load, tryReconnectFromStoredProfile]);

  useEffect(() => {
    load(false);
  }, [load]);

  useEffect(() => {
    // On every new SVG payload, recalculate viewport state from scratch.
    originalViewBoxRef.current = null;
    currentViewBoxRef.current = null;
    setFocusedTable(null);
  }, [svg]);

  const tableCount = schema?.tables?.length ?? 0;
  const fkCount = schema?.foreign_keys?.length ?? 0;
  const generatedAt = schema?.metadata?.generated_at
    ? new Date(schema.metadata.generated_at * 1000).toLocaleString()
    : "н/д";
  const visibleTables = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    const set = new Set<string>();
    const tables = schema?.tables ?? [];
    const cols = schema?.columns ?? {};
    for (const t of tables) {
      if (!q) {
        set.add(t);
        continue;
      }
      const tableMatch = t.toLowerCase().includes(q);
      const columnMatch = (cols[t] ?? []).some((c) => String(c?.name ?? "").toLowerCase().includes(q));
      if (tableMatch || columnMatch) set.add(t);
    }
    return set;
  }, [schema, filterText]);

  useEffect(() => {
    if (focusedTable && !visibleTables.has(focusedTable)) setFocusedTable(null);
  }, [focusedTable, visibleTables]);

  useEffect(() => {
    const host = svgHostRef.current;
    if (!host) return;
    const root = host.querySelector("svg") as SVGSVGElement | null;
    if (!root) return;
    svgRootRef.current = root;

    if (!originalViewBoxRef.current) {
      const vb = root.getAttribute("viewBox");
      if (vb) {
        originalViewBoxRef.current = vb;
        currentViewBoxRef.current = vb;
      } else {
        const w = Number(root.getAttribute("width")) || 1200;
        const h = Number(root.getAttribute("height")) || 800;
        originalViewBoxRef.current = `0 0 ${w} ${h}`;
        root.setAttribute("viewBox", originalViewBoxRef.current);
        currentViewBoxRef.current = originalViewBoxRef.current;
      }
    } else if (!currentViewBoxRef.current) {
      currentViewBoxRef.current = root.getAttribute("viewBox") || originalViewBoxRef.current;
    }

    const q = filterText.trim().toLowerCase();
    const tableNodes = Array.from(root.querySelectorAll<SVGGElement>("g.erd-table"));
    const edgeNodes = Array.from(root.querySelectorAll<SVGGElement>("g.erd-edge"));

    const shownTables = new Set<string>();
    for (const tableNode of tableNodes) {
      const tableName = String(tableNode.dataset.table ?? "");
      const colNames = Array.from(tableNode.querySelectorAll<SVGTextElement>(".erd-col-name")).map(
        (el) => String(el.dataset.column ?? "").toLowerCase()
      );
      const matches = !q || tableName.toLowerCase().includes(q) || colNames.some((c) => c.includes(q));
      tableNode.style.display = matches ? "" : "none";
      tableNode.style.cursor = "pointer";
      tableNode.style.opacity = "1";
      const body = tableNode.querySelector<SVGRectElement>(".erd-table-body");
      if (body) {
        body.style.stroke = "#334155";
        body.style.strokeWidth = "1.2";
      }
      if (matches) shownTables.add(tableName);
    }

    for (const edgeNode of edgeNodes) {
      const fromTable = String(edgeNode.dataset.fromTable ?? "");
      const toTable = String(edgeNode.dataset.toTable ?? "");
      const fromColumn = String(edgeNode.dataset.fromColumn ?? "");
      const toColumn = String(edgeNode.dataset.toColumn ?? "");
      const matches = !q
        || fromTable.toLowerCase().includes(q)
        || toTable.toLowerCase().includes(q)
        || fromColumn.toLowerCase().includes(q)
        || toColumn.toLowerCase().includes(q);
      const show = shownTables.has(fromTable) && shownTables.has(toTable) && matches;
      edgeNode.style.display = show ? "" : "none";
      const path = edgeNode.querySelector<SVGPathElement>(".erd-edge-path");
      if (path) {
        path.style.stroke = "#475569";
        path.style.strokeWidth = "1.4";
        path.style.opacity = "1";
      }
      const edgeText = edgeNode.querySelector<SVGTextElement>("text");
      if (edgeText) edgeText.style.opacity = "1";
    }

    const applyHighlight = (tableName: string | null, columnName?: string | null) => {
      if (!tableName) {
        for (const tableNode of tableNodes) {
          if (tableNode.style.display === "none") continue;
          tableNode.style.opacity = "1";
          const body = tableNode.querySelector<SVGRectElement>(".erd-table-body");
          if (body) {
            body.style.stroke = "#334155";
            body.style.strokeWidth = "1.2";
          }
        }
        for (const edgeNode of edgeNodes) {
          if (edgeNode.style.display === "none") continue;
          const path = edgeNode.querySelector<SVGPathElement>(".erd-edge-path");
          if (path) {
            path.style.stroke = "#475569";
            path.style.strokeWidth = "1.4";
            path.style.opacity = "1";
          }
        }
        return;
      }

      const relatedTables = new Set<string>([tableName]);
      for (const edgeNode of edgeNodes) {
        if (edgeNode.style.display === "none") continue;
        const ft = String(edgeNode.dataset.fromTable ?? "");
        const tt = String(edgeNode.dataset.toTable ?? "");
        if (ft === tableName) relatedTables.add(tt);
        if (tt === tableName) relatedTables.add(ft);
      }

      for (const tableNode of tableNodes) {
        if (tableNode.style.display === "none") continue;
        const name = String(tableNode.dataset.table ?? "");
        const active = name === tableName;
        tableNode.style.opacity = relatedTables.has(name) ? "1" : "0.28";
        const body = tableNode.querySelector<SVGRectElement>(".erd-table-body");
        if (body) {
          body.style.stroke = active ? "#0284c7" : "#334155";
          body.style.strokeWidth = active ? "2.2" : "1.2";
        }
      }

      for (const edgeNode of edgeNodes) {
        if (edgeNode.style.display === "none") continue;
        const ft = String(edgeNode.dataset.fromTable ?? "");
        const fc = String(edgeNode.dataset.fromColumn ?? "");
        const tt = String(edgeNode.dataset.toTable ?? "");
        const tc = String(edgeNode.dataset.toColumn ?? "");
        const connected = ft == tableName || tt == tableName;
        const exactColumn = Boolean(columnName) && (
          (ft === tableName && fc === columnName) || (tt === tableName && tc === columnName)
        );
        const path = edgeNode.querySelector<SVGPathElement>(".erd-edge-path");
        if (!path) continue;
        path.style.opacity = connected ? "1" : "0.12";
        path.style.stroke = exactColumn ? "#0ea5e9" : connected ? "#1d4ed8" : "#94a3b8";
        path.style.strokeWidth = exactColumn ? "2.6" : connected ? "2.0" : "1.1";
      }
    };

    const cleanups: Array<() => void> = [];
    for (const tableNode of tableNodes) {
      const tableName = String(tableNode.dataset.table ?? "");
      const onEnter = () => applyHighlight(tableName);
      const onLeave = () => applyHighlight(null);
      const onClick = () => setFocusedTable((prev) => (prev === tableName ? null : tableName));
      tableNode.addEventListener("mouseenter", onEnter);
      tableNode.addEventListener("mouseleave", onLeave);
      tableNode.addEventListener("click", onClick);
      cleanups.push(() => {
        tableNode.removeEventListener("mouseenter", onEnter);
        tableNode.removeEventListener("mouseleave", onLeave);
        tableNode.removeEventListener("click", onClick);
      });

      const colNodes = Array.from(tableNode.querySelectorAll<SVGTextElement>(".erd-col-name"));
      for (const colNode of colNodes) {
        const colName = String(colNode.dataset.column ?? "");
        const onColEnter = () => applyHighlight(tableName, colName);
        const onColLeave = () => applyHighlight(tableName);
        colNode.style.cursor = "pointer";
        colNode.addEventListener("mouseenter", onColEnter);
        colNode.addEventListener("mouseleave", onColLeave);
        cleanups.push(() => {
          colNode.removeEventListener("mouseenter", onColEnter);
          colNode.removeEventListener("mouseleave", onColLeave);
        });
      }
    }

    const onWheel = (event: WheelEvent) => {
      // Keep native scrolling unless user explicitly zooms with Ctrl/Cmd + wheel.
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      const factor = event.deltaY > 0 ? 1.12 : 0.9;
      const current = parseViewBox(currentViewBoxRef.current);
      if (!current) return;
      const [x, y, w, h] = current;
      const nw = Math.max(140, w * factor);
      const nh = Math.max(100, h * factor);
      const cx = x + w / 2;
      const cy = y + h / 2;
      currentViewBoxRef.current = `${cx - nw / 2} ${cy - nh / 2} ${nw} ${nh}`;
      root.setAttribute("viewBox", currentViewBoxRef.current);
    };
    const onMouseDown = (event: MouseEvent) => {
      // Pan mode only with Alt + left mouse to avoid breaking normal scroll usage.
      if (!event.altKey) return;
      if (event.button !== 0) return;
      const current = parseViewBox(currentViewBoxRef.current) || parseViewBox(root.getAttribute("viewBox"));
      if (!current) return;
      panRef.current = { x: event.clientX, y: event.clientY, vb: current };
      host.style.cursor = "grabbing";
      event.preventDefault();
    };
    const onMouseMove = (event: MouseEvent) => {
      const pan = panRef.current;
      if (!pan) return;
      const rect = host.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      const dxPx = event.clientX - pan.x;
      const dyPx = event.clientY - pan.y;
      const scaleX = pan.vb[2] / rect.width;
      const scaleY = pan.vb[3] / rect.height;
      const nx = pan.vb[0] - dxPx * scaleX;
      const ny = pan.vb[1] - dyPx * scaleY;
      currentViewBoxRef.current = `${nx} ${ny} ${pan.vb[2]} ${pan.vb[3]}`;
      root.setAttribute("viewBox", currentViewBoxRef.current);
    };
    const onMouseUp = () => {
      panRef.current = null;
      host.style.cursor = "default";
    };
    host.addEventListener("wheel", onWheel, { passive: false });
    host.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);

    return () => {
      host.removeEventListener("wheel", onWheel);
      host.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      cleanups.forEach((fn) => fn());
      if (svgRootRef.current === root) svgRootRef.current = null;
    };
  }, [svg, filterText, parseViewBox]);

  useEffect(() => {
    if (!focusedTable) return;
    const host = svgHostRef.current;
    if (!host) return;
    const root = host.querySelector("svg") as SVGSVGElement | null;
    if (!root) return;

    const tableNodes = Array.from(root.querySelectorAll<SVGGElement>("g.erd-table"));
    const target = tableNodes.find((n) => n.dataset.table === focusedTable && n.style.display !== "none");
    if (!target) return;

    try {
      const bb = target.getBBox();
      if (bb.width === 0 && bb.height === 0) return;
      const pad = 28;
      const x = Math.max(0, bb.x - pad);
      const y = Math.max(0, bb.y - pad);
      const w = bb.width + pad * 2;
      const h = bb.height + pad * 2;
      const vb = `${x} ${y} ${w} ${h}`;
      currentViewBoxRef.current = vb;
      root.setAttribute("viewBox", vb);
    } catch {
      // getBBox can throw if element is not rendered yet
    }
  }, [focusedTable, svg]);

  return (
    <main className="relative min-h-screen overflow-hidden text-white">
      <div className="relative mx-auto w-full max-w-[1920px] px-6 py-10">
        <div className="mb-6 flex items-center justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold">Структура БД</h1>
            <p className="text-sm text-white/60">Схема PostgreSQL и ERD (Graphviz)</p>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/">
              <Button variant="secondary" className="border-white/10 bg-white/5">
                Назад
              </Button>
            </Link>
            <Button onClick={refreshWithReconnect} disabled={loading}>
              {loading ? "Обновление..." : "Обновить"}
            </Button>
            <Button
              variant="secondary"
              className="border-white/10 bg-white/5"
              disabled={!svg}
              onClick={() => triggerDownload(svg, "db_schema.svg", "image/svg+xml")}
            >
              Скачать SVG
            </Button>
            <Button
              variant="secondary"
              className="border-white/10 bg-white/5"
              onClick={async () => {
                try {
                  const dot = await getDbSchemaErd("dot", false);
                  triggerDownload(dot, "db_schema.dot", "text/vnd.graphviz");
                } catch (e: any) {
                  setError(e?.message ?? "Не удалось скачать DOT");
                }
              }}
            >
              Скачать DOT
            </Button>
          </div>
        </div>

        <Card className="mb-6 border-white/10 bg-white/5 backdrop-blur-xl">
          <CardHeader>
            <CardTitle className="text-lg">Сводка</CardTitle>
            <CardDescription className="text-white/60">API: {apiBase}</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-2 text-sm text-white/80 md:grid-cols-3">
            <div>Таблиц: {tableCount}</div>
            <div>Внешних ключей: {fkCount}</div>
            <div>Сформировано: {generatedAt}</div>
          </CardContent>
        </Card>

        <Card className="border-white/10 bg-white/5 backdrop-blur-xl">
          <CardHeader>
            <CardTitle className="text-lg">ER-диаграмма</CardTitle>
            <CardDescription className="text-white/60">
              Эндпоинт: <code>/api/db/schema/erd?format=svg</code>
            </CardDescription>
          </CardHeader>
          <CardContent>
            {error ? <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">{error}</div> : null}
            {!error && !svg ? <div className="text-sm text-white/60">Нет данных диаграммы.</div> : null}
            <div className="mb-3 grid gap-2 md:grid-cols-[1fr_auto_auto_auto]">
              <input
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                className="h-10 rounded-md border border-white/10 bg-white/5 px-3 text-sm text-white placeholder:text-white/45"
                placeholder="Фильтр по таблице или колонке (например: airport_code)"
              />
              <Button
                variant="secondary"
                className="border-white/10 bg-white/5"
                onClick={() => setFocusedTable(null)}
                disabled={!focusedTable}
              >
                Сбросить фокус
              </Button>
              <div className="flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-2 py-1">
                <Button
                  variant="secondary"
                  className="h-8 border-white/10 bg-white/5 px-2"
                  onClick={() => zoomBy(0.9)}
                  title="Приблизить"
                >
                  +
                </Button>
                <Button
                  variant="secondary"
                  className="h-8 border-white/10 bg-white/5 px-2"
                  onClick={() => zoomBy(1.12)}
                  title="Отдалить"
                >
                  -
                </Button>
                <Button
                  variant="secondary"
                  className="h-8 border-white/10 bg-white/5 px-2 text-xs"
                  onClick={resetZoom}
                  title="Сбросить масштаб"
                >
                  100%
                </Button>
              </div>
              <div className="flex items-center rounded-md border border-white/10 bg-white/5 px-3 text-xs text-white/70">
                Таблиц: {visibleTables.size}/{tableCount}
              </div>
            </div>
            <div className="mb-3 text-xs text-white/55">
              Наведи на таблицу/колонку для подсветки связей, кликни по таблице для фокуса.
              Масштаб: кнопками +/-, либо Ctrl/Cmd + колесо. Перемещение: Alt + ЛКМ.
              {focusedTable ? ` Фокус: ${focusedTable}` : ""}
            </div>
            {svg ? (
              <div
                ref={svgHostRef}
                className="max-h-[78vh] min-h-[560px] overflow-auto rounded-lg border border-white/10 bg-white p-2"
                dangerouslySetInnerHTML={{ __html: svg }}
              />
            ) : null}
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
