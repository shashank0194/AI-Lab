'use client';

/**
 * Chapter 7.3 — Structured Output & Tool Calling.
 *
 * Every JSON string and tool pick on screen is a real generation from the
 * tiny causal LM, decoded via `greedyDecode` and parsed via
 * `parseToolCall`/`extractToolName` — no simulated output anywhere. Levels
 * decode on demand: the player sets a lever (example count, tool order),
 * triggers a real run, and the canvas hands the real text back into the
 * engine, the same "canvas supplies the real value, the engine grades it"
 * pattern World 7.2's canvas already established.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, RotateCcw } from 'lucide-react';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  greedyDecode,
  buildSchemaPrompt,
  parseToolCall,
  buildToolSelectPrompt,
  extractToolName,
  retryExampleCandidates,
  solve,
  MAX_EXAMPLE_COUNT,
  type ToolCallAction,
  type ToolCallConfig,
  type ToolCallState,
} from '@/engines/toolCallEngine';
import { tinyCausalLM, preloadCausalLM, CAUSAL_LM_MODEL_ID } from '@/models/tinyCausalLM';
import { ModelGate } from '@/components/ui/ModelGate';
import { Button, Panel, Readout, Tag, cx } from '@/components/ui';
import { useRetrySignal } from '../useRetrySignal';
import { useSolutionFill } from '../useSolutionFill';
import type { GameComponentProps } from '../registry';
import type { EngineRules } from '@/types/game';

export function ToolCallCanvas({ game, level, onScore, onSubmit }: GameComponentProps) {
  const config = level.engineConfig as unknown as ToolCallConfig;
  const rules: EngineRules = useMemo(
    () => ({ passCriteria: level.passCriteria, starsRules: level.starsRules, xpReward: level.xpReward }),
    [level]
  );

  const [state, setState] = useState<ToolCallState | null>(null);

  const load = useCallback(async () => {
    // Every mode decodes on demand, so `prepare()` alone would never trigger
    // the download — warm the model explicitly, same fix 7.2's canvas needed.
    await preloadCausalLM();
    const prepared = await prepare(config);
    setState(initState(config, rules, prepared));
  }, [config, rules]);

  const dispatch = useCallback((action: ToolCallAction) => {
    setState((prev) => (prev ? applyAction(prev, action) : prev));
  }, []);

  return (
    <ModelGate
      modelId={CAUSAL_LM_MODEL_ID}
      estimatedSizeMB={game.modelRequirement.estimatedSizeMB}
      loadFailureMessage={game.modelRequirement.loadFailureMessage}
      load={load}
    >
      {state && (
        <Board state={state} dispatch={dispatch} setState={setState} onScore={onScore} onSubmit={onSubmit} />
      )}
    </ModelGate>
  );
}

function Board({
  state,
  dispatch,
  setState,
  onScore,
  onSubmit,
}: {
  state: ToolCallState;
  dispatch: (action: ToolCallAction) => void;
  /** Replaces the whole engine state, for the solution's autofill. */
  setState: (next: ToolCallState) => void;
  onScore: GameComponentProps['onScore'];
  onSubmit: GameComponentProps['onSubmit'];
}) {
  const result = useMemo(() => evaluate(state), [state]);
  useEffect(() => onScore(result), [result, onScore]);

  const [revealed, setRevealed] = useState(false);

  const resetRun = useCallback(() => {
    dispatch({ type: 'RESET' });
    setRevealed(false);
  }, [dispatch]);
  useRetrySignal(resetRun);

  // The solution's autofill. Every mode here is scored on real generations, so
  // the fill runs the same decodes the player's own clicks would and hands the
  // text to `solve()`. Levels 2 and 3 also *search*: level 2 falls back to the
  // other tool order on any round the model gets wrong, and level 3 decodes
  // each example count until one really parses — submitting only that one, so
  // its recorded attempt count is a run that worked rather than the search.
  const [filling, setFilling] = useState(false);

  useSolutionFill(
    useCallback(() => {
      void (async () => {
        setFilling(true);
        await yieldToPaint();
        try {
          if (state.mode === 'schema-reliability') {
            const questions = state.config.questions ?? [];
            const schemaResults = [];
            for (const question of questions) {
              const prompt = buildSchemaPrompt(question, MAX_EXAMPLE_COUNT);
              const text = await greedyDecode(tinyCausalLM, prompt, state.config.maxTokens);
              schemaResults.push({ question, text, valid: parseToolCall(text).valid });
            }
            const solved = solve(state, { exampleCount: MAX_EXAMPLE_COUNT, schemaResults });
            setState(solved);
            setRevealed(true);
            onSubmit(evaluate(solved));
            return;
          }

          if (state.mode === 'tool-select') {
            // One decode per round, with that round's own tool placed last —
            // the level's actual lesson. Trying the other order as a fallback
            // was measured and changes nothing on this model (the two rounds
            // it gets wrong pick `lookup` under either ordering), so it would
            // only double the wait for a result that is already the best
            // available; whatever the model really picks is what gets scored.
            const toolPicks: ({ lastTool: 'calculator' | 'lookup'; pickedTool: string | null } | null)[] = [];
            for (const round of state.selectRounds) {
              const prompt = buildToolSelectPrompt(round.text, round.expectedTool);
              const text = await greedyDecode(tinyCausalLM, prompt, state.config.maxTokens);
              toolPicks.push({ lastTool: round.expectedTool, pickedTool: extractToolName(text) });
            }
            const solved = solve(state, { toolPicks });
            setState(solved);
            setRevealed(true);
            onSubmit(evaluate(solved));
            return;
          }

          const question = state.config.question ?? '';
          for (const exampleCount of retryExampleCandidates()) {
            const prompt = buildSchemaPrompt(question, exampleCount);
            const text = await greedyDecode(tinyCausalLM, prompt, state.config.maxTokens);
            if (!parseToolCall(text).valid) continue;
            const solved = solve(state, { exampleCount, retryText: text });
            setState(solved);
            onSubmit(evaluate(solved));
            return;
          }

          // Nothing parsed on this device. Leave the most promising example
          // count selected without spending an attempt on a run that failed.
          setState(solve(state));
        } finally {
          setFilling(false);
        }
      })();
    }, [state, setState, onSubmit])
  );

  const submit = () => {
    onSubmit(evaluate(state));
    dispatch({ type: 'SUBMIT' });
    setRevealed(true);
  };

  const blocked = submitBlocked(state);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto p-3 [&>*]:shrink-0">
        <div className="mx-auto my-auto flex w-full max-w-3xl flex-col gap-4">
          {state.mode === 'schema-reliability' && <SchemaReliabilityBoard state={state} dispatch={dispatch} result={result} />}
          {state.mode === 'tool-select' && <ToolSelectBoard state={state} dispatch={dispatch} />}
          {state.mode === 'retry-fix' && <RetryFixBoard state={state} dispatch={dispatch} result={result} />}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
        <Button onClick={resetRun}>
          <RotateCcw size={13} strokeWidth={2} />
          Reset
        </Button>
        {filling && <Tag tone="accent">running the solution&rsquo;s generations…</Tag>}
        {!filling && !revealed && blocked && <span className="label">{submitHint(state)}</span>}
        {revealed && <Tag tone="accent">answers revealed</Tag>}
        <Button
          variant="primary"
          className="ml-auto"
          disabled={blocked || revealed}
          title={blocked ? submitHint(state) : undefined}
          onClick={submit}
        >
          <Check size={13} strokeWidth={2} />
          Submit
        </Button>
      </div>
    </div>
  );
}

function submitBlocked(state: ToolCallState): boolean {
  switch (state.mode) {
    case 'schema-reliability':
      return state.schemaResults.length === 0;
    case 'tool-select':
      return state.selectRounds.some((r) => !r.tested);
    case 'retry-fix':
      return !state.retrySolved;
  }
}

function submitHint(state: ToolCallState): string {
  switch (state.mode) {
    case 'schema-reliability':
      return 'run a real test first';
    case 'tool-select': {
      const left = state.selectRounds.filter((r) => !r.tested).length;
      return `${left} round${left === 1 ? '' : 's'} left to test`;
    }
    case 'retry-fix':
      return 'get a real generation to validate';
  }
}

/* ── level 1: schema reliability ───────────────────────────────── */

function SchemaReliabilityBoard({
  state,
  dispatch,
  result,
}: {
  state: ToolCallState;
  dispatch: (action: ToolCallAction) => void;
  result: ReturnType<typeof evaluate>;
}) {
  const [busy, setBusy] = useState(false);
  const questions = state.config.questions ?? [];

  const runTest = async () => {
    if (busy) return;
    setBusy(true);
    // The WASM CPU backend runs each forward pass synchronously, and this
    // loop can span dozens of them — yield once so the browser actually
    // paints "running the real model…" before the main thread blocks,
    // instead of appearing frozen until every generation finishes.
    await yieldToPaint();
    try {
      const results: { question: string; text: string; valid: boolean }[] = [];
      for (const question of questions) {
        const prompt = buildSchemaPrompt(question, state.exampleCount);
        const text = await greedyDecode(tinyCausalLM, prompt, state.config.maxTokens);
        results.push({ question, text, valid: parseToolCall(text).valid });
      }
      dispatch({ type: 'RUN_SCHEMA_TEST', results });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <Panel label="worked examples to include">
        <div className="flex gap-2">
          {([0, 1, 2] as const).map((count) => (
            <button
              key={count}
              type="button"
              aria-pressed={state.exampleCount === count}
              onClick={() => dispatch({ type: 'SET_EXAMPLE_COUNT', count })}
              className={cx(
                'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
                state.exampleCount === count
                  ? 'border-accent bg-accent-dim text-primary'
                  : 'border-line-strong bg-transparent text-secondary hover:border-accent'
              )}
              style={{ borderRadius: 'var(--radius)' }}
            >
              {count} example{count === 1 ? '' : 's'}
            </button>
          ))}
        </div>
      </Panel>

      <div className="flex items-center gap-2">
        <Button variant="primary" onClick={() => void runTest()} disabled={busy}>
          {busy ? 'running the real model…' : `run ${questions.length} real generations`}
        </Button>
        {state.schemaResults.length > 0 && (
          <Readout label="jsonValidRate" value={result.value} size="sm" tone={result.passed ? 'good' : 'bad'} />
        )}
      </div>

      {state.schemaResults.length > 0 && (
        <Panel label="real generations">
          <div className="flex flex-col gap-2">
            {state.schemaResults.map((r, i) => (
              <div key={i} className="flex items-start gap-2 border-t border-line-faint pt-2 first:border-t-0 first:pt-0">
                <Tag tone={r.valid ? 'good' : 'bad'}>{r.valid ? 'valid' : 'invalid'}</Tag>
                <div className="flex flex-col gap-0.5">
                  <span className="text-[11px] text-muted">{r.question}</span>
                  <span className="readout whitespace-pre-wrap text-xs text-secondary">{r.text || '(no output)'}</span>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}

/* ── level 2: tool select ──────────────────────────────────────── */

function ToolSelectBoard({
  state,
  dispatch,
}: {
  state: ToolCallState;
  dispatch: (action: ToolCallAction) => void;
}) {
  return (
    <div className="flex flex-col gap-4">
      {state.selectRounds.map((round, index) => (
        <ToolSelectRoundPanel key={round.text} round={round} roundIndex={index} maxTokens={state.config.maxTokens} dispatch={dispatch} />
      ))}
    </div>
  );
}

function ToolSelectRoundPanel({
  round,
  roundIndex,
  maxTokens,
  dispatch,
}: {
  round: ToolCallState['selectRounds'][number];
  roundIndex: number;
  maxTokens: number;
  dispatch: (action: ToolCallAction) => void;
}) {
  const [busy, setBusy] = useState(false);

  const test = async () => {
    if (!round.lastOrder || busy) return;
    setBusy(true);
    await yieldToPaint();
    try {
      const prompt = buildToolSelectPrompt(round.text, round.lastOrder);
      const text = await greedyDecode(tinyCausalLM, prompt, maxTokens);
      dispatch({ type: 'TEST_TOOL_SELECT', roundIndex, pickedTool: extractToolName(text) });
    } finally {
      setBusy(false);
    }
  };

  const correct = round.tested && round.pickedTool === round.expectedTool;

  return (
    <Panel label={round.text} actions={round.tested ? <Tag tone={correct ? 'good' : 'bad'}>{correct ? 'correct' : 'wrong'}</Tag> : undefined}>
      <div className="flex flex-col gap-2">
        <span className="label">list this tool last</span>
        <div className="flex gap-2">
          {(['calculator', 'lookup'] as const).map((tool) => (
            <button
              key={tool}
              type="button"
              aria-pressed={round.lastOrder === tool}
              onClick={() => dispatch({ type: 'SET_TOOL_ORDER', roundIndex, lastTool: tool })}
              className={cx(
                'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
                round.lastOrder === tool
                  ? 'border-accent bg-accent-dim text-primary'
                  : 'border-line-strong bg-transparent text-secondary hover:border-accent'
              )}
              style={{ borderRadius: 'var(--radius)' }}
            >
              {tool}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={() => void test()} disabled={!round.lastOrder || busy}>
            {busy ? 'running the real model…' : 'run the real model'}
          </Button>
          {round.tested && <Tag tone={correct ? 'good' : 'bad'}>picked: {round.pickedTool ?? '(none)'}</Tag>}
        </div>
      </div>
    </Panel>
  );
}

/* ── level 3: retry fix ────────────────────────────────────────── */

function RetryFixBoard({
  state,
  dispatch,
  result,
}: {
  state: ToolCallState;
  dispatch: (action: ToolCallAction) => void;
  result: ReturnType<typeof evaluate>;
}) {
  const [busy, setBusy] = useState(false);
  const question = state.config.question ?? '';

  const test = async () => {
    if (busy) return;
    setBusy(true);
    await yieldToPaint();
    try {
      const prompt = buildSchemaPrompt(question, state.exampleCount);
      const text = await greedyDecode(tinyCausalLM, prompt, state.config.maxTokens);
      dispatch({ type: 'TEST_RETRY', text });
    } finally {
      setBusy(false);
    }
  };

  const parsed = state.retryLastText !== null ? parseToolCall(state.retryLastText) : null;

  return (
    <div className="flex flex-col gap-4">
      <Panel label="question">
        <p className="text-sm text-primary">{question}</p>
      </Panel>

      <Panel label="worked examples to include">
        <div className="flex gap-2">
          {([0, 1, 2] as const).map((count) => (
            <button
              key={count}
              type="button"
              aria-pressed={state.exampleCount === count}
              onClick={() => dispatch({ type: 'SET_EXAMPLE_COUNT', count })}
              className={cx(
                'min-h-[36px] flex-1 border px-3 py-1.5 font-mono text-xs uppercase tracking-[0.1em] transition-colors',
                state.exampleCount === count
                  ? 'border-accent bg-accent-dim text-primary'
                  : 'border-line-strong bg-transparent text-secondary hover:border-accent'
              )}
              style={{ borderRadius: 'var(--radius)' }}
            >
              {count} example{count === 1 ? '' : 's'}
            </button>
          ))}
        </div>
      </Panel>

      <div className="flex items-center gap-2">
        <Button variant="primary" onClick={() => void test()} disabled={busy}>
          {busy ? 'running the real model…' : 'test this prompt'}
        </Button>
        <Readout label="attempts" value={state.retryAttempts} size="sm" />
        {state.retrySolved && <Tag tone="good">solved on attempt {state.retrySolvedAtAttempt}</Tag>}
      </div>

      {state.retryLastText !== null && (
        <Panel label="real model output" actions={parsed && <Tag tone={parsed.valid ? 'good' : 'bad'}>{parsed.valid ? 'valid' : 'invalid'}</Tag>}>
          <p className="readout whitespace-pre-wrap text-xs text-secondary">{state.retryLastText || '(no output)'}</p>
        </Panel>
      )}

      {state.retrySolved && (
        <Panel label="score">
          <Readout label="attemptsToSolve" value={result.value} size="lg" tone="good" />
        </Panel>
      )}
    </div>
  );
}

/** Two animation frames, so a "busy" state change actually paints before the
 * WASM CPU backend's synchronous forward passes block the main thread. */
function yieldToPaint(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
}
