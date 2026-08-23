'use client';

/**
 * The frame every chapter plays inside.
 *
 * Layout follows Section 6: the visualisation owns the screen and the HUD sits
 * around it — a collapsible sheet on mobile, a persistent side rail from
 * tablet up. The shell owns the concept panel, level progression, scoring and
 * the aha reveal; the game component owns only its own canvas and controls.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence, useReducedMotion } from 'motion/react';
import { ArrowLeft, ArrowRight, BookOpen, Lightbulb, Lock, RotateCcw, X } from 'lucide-react';
import type { GameDefinition, GameLevel, ScoreResult } from '@/types/game';
import { nextChapterAfter, worldOfChapter } from '@/lib/curriculum';
import {
  useGameProgressStore,
  type ChapterProgress,
  type LevelResult,
} from '@/store/useGameProgressStore';
import { useSessionStore } from '@/store/useSessionStore';
import { enqueueActivity } from '@/lib/offlineQueue';
import { playChapterCompleteTone, playCorrectTone } from '@/lib/sound';
import { Button, Heading, Panel, Readout, StarRating, Tag, cx } from '@/components/ui';
import { ShareButton } from '@/components/ui/ShareButton';
import { ChapterFeedback } from '@/components/chapter/ChapterFeedback';
import { BuilderLinks } from '@/components/chapter/BuilderLinks';
import { SolutionModal } from '@/components/chapter/SolutionModal';
import { ConfirmDialog } from '@/components/ui/ConfirmDialog';

export interface GameRenderProps {
  level: GameLevel;
  /** Reports the current score so the HUD can track it live. */
  onScore: (result: ScoreResult) => void;
  /** Submits the run for scoring and progression. */
  onSubmit: (result: ScoreResult) => void;
}

/** The prerequisite chapters this one's `unlockRequires` still needs. */
export interface PrerequisiteGap {
  missing: { id: string; title: string; world: number }[];
}

export interface ChapterShellProps {
  game: GameDefinition;
  /** Present whenever this chapter is being played without its prerequisites met —
   *  shows a dismissible strip banner at the top of the instrument area. */
  prerequisiteGap?: PrerequisiteGap;
  /** Renders the level's own canvas and controls. */
  children: (props: GameRenderProps) => ReactNode;
}

/**
 * Where a returning player picks back up — the first level with no recorded
 * result yet, so re-opening a chapter never re-plays what is already passed.
 * Falls back to the last level once every one of them (including any
 * optional stretch level) has a result, rather than looping back to the
 * first.
 */
function resumeLevelIndex(game: GameDefinition, progress: ChapterProgress | null): number {
  if (!progress) return 0;
  const index = game.levels.findIndex((l) => !progress.levels[l.id]);
  return index === -1 ? game.levels.length - 1 : index;
}

export function ChapterShell({ game, prerequisiteGap, children }: ChapterShellProps) {
  const reduce = useReducedMotion();
  const world = worldOfChapter(game.id);
  const nextChapter = nextChapterAfter(game.id);
  const [gapDismissed, setGapDismissed] = useState(false);

  const userId = useGameProgressStore((s) => s.userId);
  const displayName = useGameProgressStore((s) => s.displayName);
  const recordLevel = useGameProgressStore((s) => s.recordLevel);
  const completeChapter = useGameProgressStore((s) => s.completeChapter);
  const soundEnabled = useGameProgressStore((s) => s.soundEnabled);
  const completedLevels = useGameProgressStore((s) => s.chapters[game.id]?.levels);

  const levelIndex = useSessionStore((s) => s.levelIndex);
  const phase = useSessionStore((s) => s.phase);
  const attempts = useSessionStore((s) => s.attempts);
  const beginChapter = useSessionStore((s) => s.beginChapter);
  const goToLevel = useSessionStore((s) => s.goToLevel);
  const setPhase = useSessionStore((s) => s.setPhase);
  const registerAttempt = useSessionStore((s) => s.registerAttempt);
  const recordResult = useSessionStore((s) => s.recordResult);
  const requestSolutionFill = useSessionStore((s) => s.requestSolutionFill);

  const [liveScore, setLiveScore] = useState<ScoreResult | null>(null);
  const [chapterDone, setChapterDone] = useState(false);
  const [hintsRevealed, setHintsRevealed] = useState(0);
  // 'warning' is the "are you sure" step between the gate opening and the
  // solution itself, so reading the solution is always a deliberate choice.
  const [solutionStage, setSolutionStage] = useState<'closed' | 'warning' | 'open'>('closed');
  const [solutionUsed, setSolutionUsed] = useState(false);

  const level = game.levels[levelIndex];
  const requiredLevels = useMemo(() => game.levels.filter((l) => !l.optional), [game.levels]);

  useEffect(() => {
    beginChapter(game.id);
    // A returning player resumes past what they have already cleared, rather
    // than always landing back on level 1 — the persisted result is the
    // source of truth here, not the (deliberately unpersisted) session state.
    const progress = useGameProgressStore.getState().chapterProgress(game.id);
    const resumeIndex = resumeLevelIndex(game, progress);
    if (resumeIndex > 0) goToLevel(resumeIndex);
    if (userId) void enqueueActivity({ userId, type: 'chapter_started', chapterId: game.id });
  }, [game, userId, beginChapter, goToLevel]);

  // A new level starts with none of its hints shown and its solution locked
  // again. Not tied to `phase`, so a retry within the same level keeps whatever
  // the player already unlocked — including the fact that they used the
  // solution, which is what marks the level as solution-solved when it passes.
  useEffect(() => {
    setHintsRevealed(0);
    setSolutionStage('closed');
    setSolutionUsed(false);
  }, [level?.id]);

  const handleSubmit = useCallback(
    (result: ScoreResult) => {
      if (!level) return;
      registerAttempt();
      recordResult(result);
      setLiveScore(result);

      // Using the solution on this level taints every pass that follows it in
      // the same visit; the store then decides whether that can downgrade an
      // earlier unaided clear (it cannot).
      const solvedWith = solutionUsed ? 'solution' : 'self';

      if (result.passed) {
        recordLevel(game.id, {
          levelId: level.id,
          stars: result.stars,
          metric: result.metric,
          value: result.value,
          xpEarned: result.xpEarned,
          attempts: attempts + 1,
          completedAt: Date.now(),
          solvedWith,
        });
        if (soundEnabled) playCorrectTone();
      }

      if (userId) {
        void enqueueActivity({
          userId,
          type: result.passed ? 'level_completed' : 'level_failed',
          chapterId: game.id,
          levelId: level.id,
          detail: {
            metric: result.metric,
            value: result.value,
            stars: result.stars,
            xp: result.xpEarned,
            attempts: attempts + 1,
            solvedWith,
          },
        });
      }
    },
    [
      level,
      game.id,
      userId,
      attempts,
      soundEnabled,
      solutionUsed,
      registerAttempt,
      recordResult,
      recordLevel,
    ]
  );

  const advance = useCallback(() => {
    const isLastRequired = levelIndex >= requiredLevels.length - 1;
    if (!isLastRequired) {
      goToLevel(levelIndex + 1);
      setLiveScore(null);
      return;
    }

    // Chapter stars are the weakest link across its required levels — clearing
    // two perfectly and scraping the third is a two-star chapter.
    const progress = useGameProgressStore.getState().chapterProgress(game.id);
    const earned = requiredLevels.map((l) => progress?.levels[l.id]?.stars ?? 0);
    const chapterStars = earned.length === 0 ? 0 : Math.min(...earned);

    completeChapter(game.id, chapterStars);
    setChapterDone(true);
    if (soundEnabled) playChapterCompleteTone();

    if (userId) {
      void enqueueActivity({
        userId,
        type: 'chapter_completed',
        chapterId: game.id,
        detail: { stars: chapterStars },
      });
    }
  }, [levelIndex, requiredLevels, game.id, userId, soundEnabled, goToLevel, completeChapter]);

  /* ── the worked solution ────────────────────────────────────
     Every step through it is an event: opening it past the warning,
     taking the fill, and choosing to fill it in yourself instead. Hint
     reveals are tracked the same way, so "how much help did this level
     need" is answerable per level rather than only per chapter. */

  const revealHint = useCallback(() => {
    if (!level?.hints?.length) return;
    const next = Math.min(hintsRevealed + 1, level.hints.length);
    if (next === hintsRevealed) return;
    setHintsRevealed(next);
    if (userId) {
      void enqueueActivity({
        userId,
        type: 'hint_revealed',
        chapterId: game.id,
        levelId: level.id,
        detail: { hintIndex: next, totalHints: level.hints.length },
      });
    }
  }, [level, hintsRevealed, userId, game.id]);

  const openSolution = useCallback(() => {
    if (!level) return;
    setSolutionStage('open');
    if (userId) {
      void enqueueActivity({
        userId,
        type: 'solution_opened',
        chapterId: game.id,
        levelId: level.id,
        detail: {
          hintsRevealed,
          alreadyPassed: Boolean(completedLevels?.[level.id]),
        },
      });
    }
  }, [level, userId, game.id, hintsRevealed, completedLevels]);

  const autofillSolution = useCallback(() => {
    if (!level) return;
    // Set before the signal fires: the canvas submits synchronously off the
    // back of it, and `handleSubmit` reads this to decide how the pass counts.
    setSolutionUsed(true);
    setSolutionStage('closed');
    requestSolutionFill();
    if (userId) {
      void enqueueActivity({
        userId,
        type: 'solution_autofilled',
        chapterId: game.id,
        levelId: level.id,
        detail: { alreadyPassed: Boolean(completedLevels?.[level.id]) },
      });
    }
  }, [level, userId, game.id, completedLevels, requestSolutionFill]);

  const declineSolution = useCallback(() => {
    if (!level) return;
    setSolutionStage('closed');
    if (userId) {
      void enqueueActivity({
        userId,
        type: 'solution_declined',
        chapterId: game.id,
        levelId: level.id,
      });
    }
  }, [level, userId, game.id]);

  if (!level) return null;

  return (
    <div data-world={game.world} className="flex min-h-0 flex-1 flex-col lg:flex-row">
      {/* ── the instrument: dominant, always visible ── */}
      <div className="relative flex min-h-[55vh] min-w-0 flex-1 flex-col lg:min-h-0">
        <ChapterBar
          game={game}
          level={level}
          levelIndex={levelIndex}
          totalLevels={game.levels.length}
          worldNumber={world?.world ?? game.world}
        />

        {prerequisiteGap && prerequisiteGap.missing.length > 0 && !gapDismissed && (
          <PrerequisiteGapBanner gap={prerequisiteGap} onDismiss={() => setGapDismissed(true)} />
        )}

        <div className="relative flex min-h-0 flex-1 flex-col">
          {phase === 'concept' ? (
            <ConceptPanel game={game} level={level} onStart={() => setPhase('playing')} />
          ) : (
            children({ level, onScore: setLiveScore, onSubmit: handleSubmit })
          )}
        </div>
      </div>

      {/* ── HUD: below the canvas on mobile, a persistent rail from lg up.
          Always rendered — there is no toggle to find the score or hints.
          `lg:overflow-hidden` so the rail's own inner scroller is the only
          thing that moves: revealing a hint lengthens the HUD, and that must
          not resize or scroll the canvas sitting beside it. ── */}
      <aside
        className={cx(
          'flex flex-col border-t border-line bg-panel/40',
          'lg:min-h-0 lg:w-[320px] lg:shrink-0 lg:overflow-hidden lg:border-l lg:border-t-0'
        )}
      >
        <Hud
          game={game}
          level={level}
          levelIndex={levelIndex}
          liveScore={liveScore}
          attempts={attempts}
          phase={phase}
          completedLevels={completedLevels}
          hintsRevealed={hintsRevealed}
          solutionUsed={solutionUsed}
          onRevealHint={revealHint}
          onOpenSolution={() => setSolutionStage('warning')}
          onRetry={() => {
            setLiveScore(null);
            setPhase('playing');
          }}
          onSelectLevel={(index) => {
            setLiveScore(null);
            goToLevel(index);
          }}
          onAdvance={advance}
        />
      </aside>

      {/* The warning sits between the unlocked button and the solution itself:
          the point of the gate is that reading the answer is a decision, not a
          click you can make by reflex on the way past. */}
      <AnimatePresence>
        {solutionStage === 'warning' && level.solution && (
          <ConfirmDialog
            title="Read the worked solution?"
            body={
              <>
                You have used every hint on this level, so this is the whole answer — what it is,
                why it is that and not something else, worked through on this level&apos;s own
                numbers. It is worth being sure you have actually tried it first: the understanding
                is the point, and the next chapter builds on it. Going back to revise and coming
                back later costs you nothing.
              </>
            }
            confirmLabel="Show me the solution"
            cancelLabel="Let me try again first"
            onConfirm={openSolution}
            onCancel={() => setSolutionStage('closed')}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {solutionStage === 'open' && level.solution && (
          <SolutionModal
            level={level}
            solution={level.solution}
            alreadyPassed={Boolean(completedLevels?.[level.id])}
            canAutofill={phase !== 'concept'}
            onAutofill={autofillSolution}
            onFillMyOwn={declineSolution}
            onClose={() => setSolutionStage('closed')}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {chapterDone && (
          <AhaReveal
            game={game}
            reduce={reduce ?? false}
            userId={userId}
            displayName={displayName}
            nextHref={
              nextChapter
                ? `/world/${nextChapter.id.split('-')[0]}/chapter/${nextChapter.id}`
                : '/map'
            }
            nextLabel={nextChapter ? nextChapter.title : 'Back to the map'}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

/* ── top bar ────────────────────────────────────────────────── */

function ChapterBar({
  game,
  level,
  levelIndex,
  totalLevels,
  worldNumber,
}: {
  game: GameDefinition;
  level: GameLevel;
  levelIndex: number;
  totalLevels: number;
  worldNumber: number;
}) {
  return (
    <div className="flex items-center gap-3 border-b border-line px-3 py-2">
      <Link
        href="/map"
        className="flex items-center gap-1.5 text-muted transition-colors hover:text-primary"
        aria-label="Back to the map"
      >
        <ArrowLeft size={14} strokeWidth={2} />
        <span className="label hidden sm:inline">map</span>
      </Link>

      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span className="readout text-[10px] text-accent">W{worldNumber}</span>
          <span className="truncate text-[13px] text-primary">{game.chapterTitle}</span>
        </div>
      </div>

      <ShareButton world={worldNumber} chapterId={game.id} chapterTitle={game.chapterTitle} size={13} />

      <span className="label whitespace-nowrap">
        {levelIndex + 1}/{totalLevels} · {level.difficulty}
      </span>
    </div>
  );
}


/* ── prerequisite-gap banner ────────────────────────────────── */

/**
 * A dismissible one-line strip shown at the top of the instrument area when a
 * player opens a chapter without completing its prerequisites (via direct URL,
 * bookmark, back-button, or share link). It stays up for the session; dismissing
 * hides it for this visit only, since the underlying gap hasn't closed.
 */
function PrerequisiteGapBanner({
  gap,
  onDismiss,
}: {
  gap: PrerequisiteGap;
  onDismiss: () => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
      <span className="min-w-0 flex-1 leading-relaxed">
        You haven&apos;t finished{' '}
        {gap.missing.map((m, i) => (
          <span key={m.id}>
            {i > 0 && ', '}
            <Link
              href={`/world/${m.world}/chapter/${m.id}`}
              className="underline underline-offset-2 hover:text-primary"
            >
              {m.title}
            </Link>
          </span>
        ))}{' '}
        yet — finish {gap.missing.length === 1 ? 'it' : 'them'} first to build on this properly.
      </span>
      <Link href="/map" className="label shrink-0 underline underline-offset-2 hover:text-primary">
        map
      </Link>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss"
        className="shrink-0 text-warn transition-colors hover:text-primary"
      >
        <X size={13} strokeWidth={2} />
      </button>
    </div>
  );
}

/* ── concept panel ──────────────────────────────────────────── */

function ConceptPanel({
  game,
  level,
  onStart,
}: {
  game: GameDefinition;
  level: GameLevel;
  onStart: () => void;
}) {
  return (
    <div className="grid-field flex min-h-0 flex-1 flex-col overflow-y-auto px-6 py-10">
      <div className="mx-auto my-auto w-full max-w-xl">
        <p className="label mb-2">{game.id}</p>
        <Heading level={2} className="mb-4">
          {game.chapterTitle}
        </Heading>

        <p className="mb-6 text-sm leading-relaxed text-secondary">
          {game.concept.shortExplanation}
        </p>

        <div className="mb-6 border-l-2 border-accent/50 pl-4">
          <p className="label mb-1">level {level.id} · {level.title}</p>
          <p className="text-sm leading-relaxed text-primary">{level.instructions}</p>
        </div>

        <Button variant="primary" onClick={onStart}>
          Begin
          <ArrowRight size={13} strokeWidth={2} />
        </Button>
      </div>
    </div>
  );
}

/* ── HUD ────────────────────────────────────────────────────── */

function Hud({
  game,
  level,
  levelIndex,
  liveScore,
  attempts,
  phase,
  completedLevels,
  hintsRevealed,
  solutionUsed,
  onRevealHint,
  onOpenSolution,
  onRetry,
  onAdvance,
  onSelectLevel,
}: {
  game: GameDefinition;
  level: GameLevel;
  levelIndex: number;
  liveScore: ScoreResult | null;
  attempts: number;
  phase: string;
  /** Which of this chapter's levels already have a passing result, keyed by level id. */
  completedLevels: Record<string, LevelResult> | undefined;
  hintsRevealed: number;
  /** True once the solution's autofill has been used on the current level. */
  solutionUsed: boolean;
  onRevealHint: () => void;
  onOpenSolution: () => void;
  onRetry: () => void;
  onAdvance: () => void;
  /** Jumps straight to an already-cleared level, replaying it. */
  onSelectLevel: (index: number) => void;
}) {
  const comparator = level.passCriteria.comparator;
  const target = level.passCriteria.threshold;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
      <Panel label="score">
        <div className="flex flex-col gap-4">
          <Readout
            label={level.passCriteria.metric}
            value={liveScore?.value ?? '—'}
            size="lg"
            tone={liveScore === null ? 'neutral' : liveScore.passed ? 'good' : 'bad'}
          />

          <div className="flex items-center justify-between gap-3">
            <span className="label">
              target {comparator === 'lte' ? '≤' : comparator === 'gte' ? '≥' : '='} {target}
            </span>
            <StarRating stars={liveScore?.stars ?? 0} size={14} />
          </div>

          <div className="flex items-center justify-between gap-3">
            <span className="label">attempts</span>
            <span className="readout text-sm text-secondary">{attempts}</span>
          </div>
        </div>
      </Panel>

      {level.hints && level.hints.length > 0 && (
        <HintPanel hints={level.hints} revealed={hintsRevealed} onReveal={onRevealHint} />
      )}

      {level.solution && (
        <SolutionPanel
          hintsTotal={level.hints?.length ?? 0}
          hintsRevealed={hintsRevealed}
          solutionUsed={solutionUsed}
          onOpen={onOpenSolution}
        />
      )}

      {liveScore && Object.keys(liveScore.breakdown).length > 0 && (
        <Panel label="breakdown">
          <dl className="flex flex-col gap-1.5">
            {Object.entries(liveScore.breakdown).map(([key, value]) => (
              <div key={key} className="flex items-baseline justify-between gap-3">
                <dt className="truncate text-[11px] text-muted">{key}</dt>
                <dd className="readout text-xs text-secondary">
                  {Number.isFinite(value) ? Number(value).toFixed(3) : String(value)}
                </dd>
              </div>
            ))}
          </dl>
        </Panel>
      )}

      <Panel label="level">
        <ol className="flex flex-col gap-1.5">
          {game.levels.map((l, index) => {
            const result = completedLevels?.[l.id];
            const isCurrent = index === levelIndex;
            const row = (
              <>
                <span className="truncate">
                  {index + 1}. {l.title}
                </span>
                <span className="flex items-center gap-1.5">
                  {result?.solvedWith === 'solution' && <Tag tone="warn">solution</Tag>}
                  {result && <StarRating stars={result.stars} size={10} />}
                  {l.optional && <Tag>stretch</Tag>}
                </span>
              </>
            );

            // Only a level that already has a result is a re-entry point —
            // this is a replay shortcut, not a way to skip ahead.
            if (result && !isCurrent) {
              return (
                <li key={l.id}>
                  <button
                    type="button"
                    onClick={() => onSelectLevel(index)}
                    className="flex w-full items-center justify-between gap-2 text-[11px] text-muted transition-colors hover:text-primary"
                  >
                    {row}
                  </button>
                </li>
              );
            }

            return (
              <li
                key={l.id}
                className={cx(
                  'flex items-center justify-between gap-2 text-[11px]',
                  isCurrent ? 'text-accent' : 'text-muted'
                )}
              >
                {row}
              </li>
            );
          })}
        </ol>
      </Panel>

      <div className="mt-auto flex flex-col gap-2">
        {phase === 'scored' && (
          <Button variant="primary" onClick={onAdvance}>
            {levelIndex >= game.levels.filter((l) => !l.optional).length - 1
              ? 'Finish chapter'
              : 'Next level'}
            <ArrowRight size={13} strokeWidth={2} />
          </Button>
        )}
        {(phase === 'scored' || phase === 'failed') && (
          <Button onClick={onRetry}>
            <RotateCcw size={13} strokeWidth={2} />
            Try again
          </Button>
        )}
      </div>
    </div>
  );
}

/* ── hints ──────────────────────────────────────────────────── */

/**
 * Progressive hints, one level's worth at a time.
 *
 * Revealed hints stay visible — nothing here re-hides them on a retry or a
 * score change, since `hintsRevealed` only resets when `ChapterShell` sees a
 * new `level.id`. The last hint is the level's own worked answer, not another
 * nudge, so it gets a distinct "answer" tag rather than a hint number.
 *
 * The list has a fixed max height and scrolls internally past that, rather
 * than growing — revealing hints one after another used to push the whole
 * HUD (and with it the canvas beside it) further down the page every click.
 */
function HintPanel({
  hints,
  revealed,
  onReveal,
}: {
  hints: string[];
  revealed: number;
  onReveal: () => void;
}) {
  const shown = hints.slice(0, revealed);
  const exhausted = revealed >= hints.length;
  const listRef = useRef<HTMLOListElement>(null);

  // A freshly-revealed hint should be the thing you see, not something you
  // have to go find below the fold of the box that just capped its height.
  useEffect(() => {
    const list = listRef.current;
    if (list) list.scrollTop = list.scrollHeight;
  }, [revealed]);

  return (
    <Panel label="hints" actions={<span className="label">{revealed}/{hints.length}</span>}>
      <div className="flex flex-col gap-3">
        {shown.length > 0 && (
          <ol
            ref={listRef}
            className="flex max-h-64 flex-col gap-2.5 overflow-y-auto pr-1"
          >
            {shown.map((hint, i) => {
              const isAnswer = i === hints.length - 1;
              return (
                <li key={i} className="flex items-start gap-2.5">
                  {isAnswer ? (
                    <Tag tone="accent">answer</Tag>
                  ) : (
                    <span className="label mt-0.5 shrink-0">{i + 1}</span>
                  )}
                  <p className="flex-1 text-xs leading-relaxed text-secondary">{hint}</p>
                </li>
              );
            })}
          </ol>
        )}

        {!exhausted && (
          <Button onClick={onReveal}>
            <Lightbulb size={13} strokeWidth={2} />
            {revealed === 0
              ? 'Show a hint'
              : revealed === hints.length - 1
                ? 'Reveal the answer'
                : 'Show another hint'}
          </Button>
        )}
      </div>
    </Panel>
  );
}

/* ── solution gate ──────────────────────────────────────────── */

/**
 * The way into a level's worked solution, and the gate in front of it.
 *
 * The gate is every hint: a solution the player can reach before trying the
 * hints is a solution most players will reach first, and the hints are where
 * the actual learning is staged. Until they are spent this renders a locked
 * button that says so on hover, focus or tap — `aria-disabled` rather than
 * `disabled`, because a genuinely disabled button swallows the pointer events
 * the explanation needs to appear on, which is exactly backwards for a control
 * whose whole job is telling you why it is not available yet.
 */
function SolutionPanel({
  hintsTotal,
  hintsRevealed,
  solutionUsed,
  onOpen,
}: {
  hintsTotal: number;
  hintsRevealed: number;
  solutionUsed: boolean;
  onOpen: () => void;
}) {
  const locked = hintsRevealed < hintsTotal;
  const remaining = hintsTotal - hintsRevealed;
  const [showTip, setShowTip] = useState(false);

  return (
    <Panel label="solution">
      <div className="flex flex-col gap-2">
        <div
          className="relative"
          onMouseEnter={() => locked && setShowTip(true)}
          onMouseLeave={() => setShowTip(false)}
        >
          {locked && showTip && (
            <div
              role="tooltip"
              className="absolute bottom-full left-0 z-10 mb-1.5 w-full border border-line-strong bg-raised px-2.5 py-2 text-[11px] leading-relaxed text-secondary shadow-lg"
              style={{ borderRadius: 'var(--radius)' }}
            >
              First use all the hints — {remaining} still{' '}
              {remaining === 1 ? 'unread' : 'unread'}.
            </div>
          )}

          <Button
            className="w-full"
            ariaDisabled={locked}
            title={locked ? 'First use all the hints' : undefined}
            onClick={() => {
              if (locked) {
                setShowTip(true);
                return;
              }
              onOpen();
            }}
            onFocus={() => locked && setShowTip(true)}
            onBlur={() => setShowTip(false)}
          >
            {locked ? <Lock size={13} strokeWidth={2} /> : <BookOpen size={13} strokeWidth={2} />}
            {locked ? 'Solution locked' : 'Show the solution'}
          </Button>
        </div>

        <p className="text-[11px] leading-relaxed text-muted">
          {solutionUsed
            ? 'You used the solution on this level — it will show as solution-solved on the map until you clear it unaided.'
            : locked
              ? `Reveal ${remaining === 1 ? 'the last hint' : `all ${hintsTotal} hints`} first. The solution explains why the answer is the answer, and can fill it in for you.`
              : 'The full explanation, worked through on this level, with the option to fill the answer in for you.'}
        </p>
      </div>
    </Panel>
  );
}

/* ── aha reveal ─────────────────────────────────────────────── */

/** Chapter completion. Section 6: satisfying but quick, and always skippable. */
function AhaReveal({
  game,
  reduce,
  userId,
  displayName,
  nextHref,
  nextLabel,
}: {
  game: GameDefinition;
  reduce: boolean;
  userId: string | null;
  displayName: string | null;
  nextHref: string;
  nextLabel: string;
}) {
  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center bg-void/95 px-6 backdrop-blur-sm"
      initial={reduce ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: reduce ? 0 : 0.22 }}
      role="dialog"
      aria-label="Chapter complete"
    >
      <motion.div
        data-world={game.world}
        className="max-h-[90vh] w-full max-w-lg overflow-y-auto border border-line bg-panel p-6"
        style={{ borderRadius: 'var(--radius-lg)' }}
        initial={reduce ? false : { y: 12, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: reduce ? 0 : 0.32, ease: [0.16, 1, 0.3, 1] }}
      >
        <p className="label mb-3">chapter complete</p>
        <Heading level={2} className="mb-4">
          {game.chapterTitle}
        </Heading>

        <p className="mb-6 text-sm leading-relaxed text-secondary">{game.concept.ahaMoment}</p>

        <div className="mb-6 flex flex-wrap items-center gap-2">
          <Link href={nextHref}>
            <Button variant="primary">
              {nextLabel}
              <ArrowRight size={13} strokeWidth={2} />
            </Button>
          </Link>
          <Link href="/map">
            <Button>Map</Button>
          </Link>
        </div>

        {userId && <ChapterFeedback chapterId={game.id} userId={userId} displayName={displayName} />}
        <BuilderLinks userId={userId} chapterId={game.id} />
      </motion.div>
    </motion.div>
  );
}
