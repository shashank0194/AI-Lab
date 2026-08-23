'use client';

/**
 * The world map, drawn as a dependency graph.
 *
 * Section 6 is explicit that this must read as a technical systems diagram
 * rather than a platformer path: chapters are nodes on a bus, edges are the real
 * `unlockRequires` relationships, and each world owns a hue so position in the
 * course is legible at a glance.
 *
 * Locked chapter nodes render as buttons rather than links — clicking one opens
 * a warning modal on this page (blurred background, prerequisite names listed)
 * before the player decides whether to jump ahead. Unlocked chapters link
 * directly as before.
 */
import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence, useReducedMotion } from 'motion/react';
import { Lock, Check, BookOpen, X } from 'lucide-react';
import {
  manifest,
  resolveUnlocks,
  summarise,
  type CompletionRecord,
} from '@/lib/curriculum';
import { Heading, Meter, Readout, StarRating, Tag, cx, Button } from '@/components/ui';
import { ShareButton } from '@/components/ui/ShareButton';
import type { ManifestChapter, ManifestWorld } from '@/types/game';

/** Shape of the pending-navigation state when a locked chapter is clicked. */
interface LockedChapterPending {
  chapterTitle: string;
  href: string;
  missingTitles: string[];
}

export function WorldMap({
  completed,
  xp,
  assisted = {},
}: {
  completed: CompletionRecord;
  xp: number;
  /**
   * `chapterId → how many of its levels were cleared with the worked solution`,
   * for completed chapters only. A completed chapter absent from here, or
   * present with a zero, was solved unaided.
   */
  assisted?: Record<string, number>;
}) {
  const states = resolveUnlocks(completed);
  const summary = summarise(completed, xp);
  const router = useRouter();
  const [pending, setPending] = useState<LockedChapterPending | null>(null);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <ProgressRail summary={summary} />

      <div className="grid-field flex-1 overflow-x-auto px-3 py-6 sm:px-6">
        <div className="mx-auto flex min-w-[640px] max-w-5xl flex-col gap-0">
          {manifest.worlds.map((world, index) => (
            <WorldRow
              key={world.world}
              world={world}
              states={states}
              assisted={assisted}
              isLast={index === manifest.worlds.length - 1}
              onLockedClick={setPending}
            />
          ))}
        </div>
      </div>

      {/* Locked-chapter warning modal — rendered at the map level so the map
          stays in view behind the blurred overlay rather than navigating away. */}
      <AnimatePresence>
        {pending && (
          <LockedChapterModal
            pending={pending}
            onConfirm={() => {
              setPending(null);
              router.push(pending.href);
            }}
            onCancel={() => setPending(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

/* ── locked-chapter warning modal ──────────────────────────────── */

function LockedChapterModal({
  pending,
  onConfirm,
  onCancel,
}: {
  pending: LockedChapterPending;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const reduce = useReducedMotion();
  const { chapterTitle, missingTitles } = pending;

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center px-6 backdrop-blur-sm"
      style={{ background: 'color-mix(in srgb, var(--color-void) 80%, transparent)' }}
      initial={reduce ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: reduce ? 0 : 0.2 }}
      role="dialog"
      aria-modal="true"
      aria-label={`Locked chapter: ${chapterTitle}`}
      onClick={(e) => {
        // Dismiss when clicking the backdrop itself
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      <motion.div
        className="w-full max-w-lg border border-line bg-panel p-6"
        style={{ borderRadius: 'var(--radius-lg)' }}
        initial={reduce ? false : { y: 12, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 8, opacity: 0 }}
        transition={{ duration: reduce ? 0 : 0.28, ease: [0.16, 1, 0.3, 1] }}
      >
        {/* dismiss button */}
        <button
          type="button"
          onClick={onCancel}
          aria-label="Close"
          className="absolute right-4 top-4 text-muted transition-colors hover:text-primary"
        >
          <X size={15} strokeWidth={2} />
        </button>

        <p className="label mb-3">heads up</p>
        <Heading level={2} className="mb-4">
          {missingTitles.length === 1
            ? `You haven't finished "${missingTitles[0]}" yet`
            : "You're missing a few earlier chapters"}
        </Heading>

        <p className="mb-6 text-sm leading-relaxed text-secondary">
          {chapterTitle} builds on{' '}
          {missingTitles.length === 1 ? 'that' : 'those'} intuition —{' '}
          <span className="text-primary">
            {missingTitles.length > 0 ? missingTitles.join(', ') : 'an earlier chapter'}
          </span>
          . Jumping ahead may make it harder to follow. Play it anyway?
        </p>

        <div className="flex flex-wrap items-center gap-2">
          <Button variant="primary" onClick={onConfirm}>
            Play anyway
          </Button>
          <Button onClick={onCancel}>
            Stay on the map
          </Button>
        </div>
      </motion.div>
    </motion.div>
  );
}

/* ── progress rail ──────────────────────────────────────────────── */

function ProgressRail({ summary }: { summary: ReturnType<typeof summarise> }) {
  return (
    <div className="flex flex-wrap items-end gap-x-8 gap-y-4 border-b border-line px-3 py-4 sm:px-6">
      <div className="flex flex-col gap-1">
        <span className="label">rank</span>
        <span className="font-display text-xl font-semibold text-primary">
          {summary.rank.title}
        </span>
      </div>

      <Readout label="xp" value={summary.xp} size="md" tone="accent" />
      <Readout
        label="chapters"
        value={`${summary.chaptersCompleted}/${summary.chaptersTotal}`}
        size="md"
      />
      <Readout label="stars" value={`${summary.starsEarned}/${summary.starsPossible}`} size="md" />

      {summary.nextRank && (
        <div className="min-w-[180px] flex-1">
          <Meter
            value={summary.xp}
            max={summary.nextRank.minXp}
            label={`${summary.xpToNextRank} xp to ${summary.nextRank.title}`}
          />
        </div>
      )}
    </div>
  );
}

/* ── world row ──────────────────────────────────────────────────── */

function WorldRow({
  world,
  states,
  assisted,
  isLast,
  onLockedClick,
}: {
  world: ManifestWorld;
  states: Map<string, ReturnType<typeof resolveUnlocks> extends Map<string, infer V> ? V : never>;
  assisted: Record<string, number>;
  isLast: boolean;
  onLockedClick: (pending: LockedChapterPending) => void;
}) {
  const chapters = [...world.chapters].sort((a, b) => a.order - b.order);
  const cleared = chapters.filter((c) => states.get(c.id)?.status === 'completed').length;

  return (
    <section data-world={world.world} className="relative flex gap-4 pb-8">
      {/* The bus running down the left, connecting worlds into one system. */}
      <div className="relative flex w-6 shrink-0 flex-col items-center pt-1">
        <span
          className="block h-2.5 w-2.5 border"
          style={{
            borderColor: 'var(--accent)',
            background: cleared === chapters.length ? 'var(--accent)' : 'transparent',
          }}
          aria-hidden
        />
        {!isLast && <span className="mt-1 w-px flex-1 bg-line" aria-hidden />}
      </div>

      <div className="min-w-0 flex-1">
        <header className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <Heading level={3} className="text-accent">
            World {world.world} · {world.title}
          </Heading>
          <span className="text-xs text-muted">{world.subtitle}</span>
          <span className="label ml-auto">
            {cleared}/{chapters.length} cleared
          </span>
        </header>

        <ol className="flex flex-wrap items-stretch gap-x-0 gap-y-3">
          {chapters.map((chapter, index) => (
            <li key={chapter.id} className="flex items-stretch">
              {index > 0 && <EdgeConnector satisfied={states.get(chapter.id)?.status !== 'locked'} />}
              <ChapterNode
                chapter={chapter}
                world={world}
                state={states.get(chapter.id)!}
                assistedLevels={assisted[chapter.id] ?? 0}
                onLockedClick={onLockedClick}
              />
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

/* ── edge connector ─────────────────────────────────────────────── */

/** The dependency edge between two chapters. */
function EdgeConnector({ satisfied }: { satisfied: boolean }) {
  return (
    <span className="flex w-6 items-center sm:w-8" aria-hidden>
      <span
        className={cx('h-px w-full', satisfied ? 'bg-accent/50' : 'bg-line')}
      />
    </span>
  );
}

/* ── chapter node ───────────────────────────────────────────────── */

function ChapterNode({
  chapter,
  world,
  state,
  assistedLevels,
  onLockedClick,
}: {
  chapter: ManifestChapter;
  world: ManifestWorld;
  state: { status: 'locked' | 'available' | 'completed'; stars: number; missing: string[] };
  /** How many of this chapter's levels were cleared with the worked solution. */
  assistedLevels: number;
  onLockedClick: (pending: LockedChapterPending) => void;
}) {
  const reduce = useReducedMotion();
  const locked = state.status === 'locked';
  const completed = state.status === 'completed';
  const withSolution = completed && assistedLevels > 0;
  const href = `/world/${world.world}/chapter/${chapter.id}`;

  // What a finished chapter says about itself: cleared on your own, or cleared
  // with the solution open. Named on the node rather than only in a tooltip —
  // the difference is the point of the mark.
  const solveNote = withSolution
    ? `Completed with the worked solution on ${assistedLevels} ${assistedLevels === 1 ? 'level' : 'levels'}`
    : 'Solved by you, unaided';

  const body = (
    <motion.div
      whileHover={locked || reduce ? undefined : { y: -2 }}
      transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
      className={cx(
        'flex h-full w-[168px] flex-col justify-between gap-2 border px-3 py-2.5',
        locked
          ? 'border-line bg-inset/60 saturate-0'
          : state.status === 'completed'
            ? 'border-accent/40 bg-panel'
            : 'border-line-strong bg-panel'
      )}
      style={{ borderRadius: 'var(--radius)' }}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="readout text-[10px] text-muted">{chapter.id}</span>
        {/* Room on the right for the share button, which is a sibling of the
            node's own link rather than a child of it — see below. */}
        <div className="flex items-center gap-0.5 pr-5">
          {locked ? (
            <Lock size={11} strokeWidth={2} className="text-muted" />
          ) : withSolution ? (
            <BookOpen size={11} strokeWidth={2.5} className="text-warn" aria-label={solveNote} />
          ) : completed ? (
            <Check size={11} strokeWidth={2.5} className="text-accent" aria-label={solveNote} />
          ) : null}
        </div>
      </div>

      <span
        className={cx(
          'text-[13px] leading-snug',
          locked ? 'text-muted' : 'text-primary'
        )}
      >
        {chapter.title}
      </span>

      {/* Wraps rather than squeezing: a completed chapter carries a third tag
          the 168px node cannot fit on one line. */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        {completed ? (
          <StarRating stars={state.stars} />
        ) : (
          <span className="readout text-[10px] text-muted">{chapter.xpReward} xp</span>
        )}
        <TierTag tier={chapter.tier} />
        {completed && (
          <Tag tone={withSolution ? 'warn' : 'good'}>
            {withSolution ? 'solution' : 'solved by you'}
          </Tag>
        )}
      </div>
    </motion.div>
  );

  // Locked chapters open the warning modal on the map instead of navigating.
  const node = locked ? (
    <button
      type="button"
      onClick={() =>
        onLockedClick({
          chapterTitle: chapter.title,
          href,
          missingTitles: state.missing,
        })
      }
      aria-label={`${chapter.title}, locked. Requires ${state.missing.join(', ')}.`}
      title={`Complete ${state.missing.join(', ')} first`}
      className="block h-full cursor-pointer text-left"
    >
      {body}
    </button>
  ) : (
    <Link
      href={href}
      aria-label={
        completed ? `${chapter.title}, completed. ${solveNote}.` : `${chapter.title}, ${state.status}`
      }
      title={completed ? solveNote : undefined}
      className="block h-full"
    >
      {body}
    </Link>
  );

  // The share button sits *beside* the node's link, never inside it. Both
  // wrappers are interactive — a `<button>` for a locked chapter, an `<a>` for
  // an open one — and neither may legally contain another interactive element.
  // Nesting it produced a real hydration error on every map load ("<button>
  // cannot contain a nested <button>"), so it is overlaid on the node's own
  // top-right corner instead — at the same `px-3`/`py-2.5` inset it had as a
  // child — where the `pr-5` above reserves its space.
  return (
    <div className="relative h-full">
      {node}
      <div className="absolute right-3 top-2.5">
        <ShareButton world={world.world} chapterId={chapter.id} chapterTitle={chapter.title} size={11} />
      </div>
    </div>
  );
}

/* ── tier tag ───────────────────────────────────────────────────── */

function TierTag({ tier }: { tier: ManifestChapter['tier'] }) {
  const labels: Record<ManifestChapter['tier'], string> = {
    none: 'math',
    'browser-light': 'light',
    'browser-heavy': 'heavy',
    cloud: 'cloud',
  };
  return <Tag tone={tier === 'none' ? 'neutral' : 'accent'}>{labels[tier]}</Tag>;
}
