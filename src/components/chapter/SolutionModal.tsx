'use client';

/**
 * The worked solution for a level, and the choice of what to do with it.
 *
 * Reached only after every hint is spent and the warning has been accepted, so
 * by the time this opens the player has already been told the answer once (the
 * last hint) and it still didn't land. What this adds is the *why*: the level
 * restated, the reasoning walked through on its own real numbers, the answer,
 * and the idea it generalises to.
 *
 * The two fill controls sit at the very bottom of the scroll, deliberately —
 * they are the thing you reach after reading, not a shortcut past it.
 */
import { useEffect, useRef } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { PencilLine, Wand2, X } from 'lucide-react';
import type { GameLevel, LevelSolution } from '@/types/game';
import { Button, Heading, Tag } from '@/components/ui';

export function SolutionModal({
  level,
  solution,
  alreadyPassed,
  canAutofill,
  onAutofill,
  onFillMyOwn,
  onClose,
}: {
  level: GameLevel;
  solution: LevelSolution;
  /** True when this level already has a passing result, so autofill cannot cost the player their mark. */
  alreadyPassed: boolean;
  /** False while the canvas is in no state to take a fill — the concept panel, or a model still loading. */
  canAutofill: boolean;
  onAutofill: () => void;
  onFillMyOwn: () => void;
  onClose: () => void;
}) {
  const reduce = useReducedMotion();
  const panelRef = useRef<HTMLDivElement>(null);

  // Escape closes without choosing either fill option — the same as the X.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // The scroll starts at the top every time, so the fill controls are always
  // something the player arrives at rather than something already on screen.
  useEffect(() => {
    panelRef.current?.focus();
  }, []);

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center px-4 py-6 backdrop-blur-sm sm:px-6"
      style={{ background: 'color-mix(in srgb, var(--color-void) 85%, transparent)' }}
      initial={reduce ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: reduce ? 0 : 0.2 }}
      role="dialog"
      aria-modal="true"
      aria-label={`Solution for ${level.title}`}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <motion.div
        ref={panelRef}
        tabIndex={-1}
        className="relative flex max-h-full w-full max-w-2xl flex-col overflow-hidden border border-line bg-panel outline-none"
        style={{ borderRadius: 'var(--radius-lg)' }}
        initial={reduce ? false : { y: 12, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 8, opacity: 0 }}
        transition={{ duration: reduce ? 0 : 0.28, ease: [0.16, 1, 0.3, 1] }}
      >
        <header className="flex items-start justify-between gap-3 border-b border-line px-6 py-4">
          <div className="min-w-0">
            <p className="label mb-1.5">worked solution</p>
            <Heading level={2} className="truncate">
              {level.title}
            </Heading>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close the solution"
            className="mt-1 shrink-0 text-muted transition-colors hover:text-primary"
          >
            <X size={16} strokeWidth={2} />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
          <Section label="what this level is really asking">
            <p className="text-sm leading-relaxed text-secondary">{solution.problem}</p>
          </Section>

          <Section label="working it through">
            <ol className="flex flex-col gap-4">
              {solution.steps.map((step, index) => (
                <li key={index} className="flex gap-3">
                  <span className="readout mt-0.5 shrink-0 text-[11px] text-accent">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <div className="min-w-0">
                    <p className="mb-1 text-[13px] font-medium text-primary">{step.title}</p>
                    <p className="text-sm leading-relaxed text-secondary">{step.body}</p>
                  </div>
                </li>
              ))}
            </ol>
          </Section>

          <Section label="the answer">
            <div className="border-l-2 border-accent/60 pl-4">
              <p className="text-sm leading-relaxed text-primary">{solution.answer}</p>
            </div>
          </Section>

          <Section label="why it works beyond this level">
            <p className="text-sm leading-relaxed text-secondary">{solution.takeaway}</p>
          </Section>

          {/* ── the fork: take the fill, or go do it yourself ── */}
          <div
            className="mt-2 border border-line-strong bg-inset p-4"
            style={{ borderRadius: 'var(--radius)' }}
          >
            <p className="label mb-2">now what</p>
            <p className="mb-4 text-sm leading-relaxed text-secondary">
              {alreadyPassed ? (
                <>
                  You already cleared this level yourself, so it stays marked{' '}
                  <span className="text-primary">solved by you</span> on the map whichever you pick.
                </>
              ) : (
                <>
                  Filling it in for you clears the level and unlocks what comes next, but the map
                  will show this chapter as{' '}
                  <span className="text-warn">solved with the solution</span>. Doing it yourself
                  now that you have read this keeps it marked{' '}
                  <span className="text-primary">solved by you</span> — and replaying it unaided
                  later upgrades the mark either way.
                </>
              )}
            </p>

            <div className="flex flex-wrap items-center gap-2">
              <Button variant="primary" onClick={onAutofill} disabled={!canAutofill}>
                <Wand2 size={13} strokeWidth={2} />
                Fill it in for me
              </Button>
              <Button onClick={onFillMyOwn}>
                <PencilLine size={13} strokeWidth={2} />
                I&apos;ll fill it in myself
              </Button>
            </div>

            {!canAutofill && (
              <p className="mt-3 text-xs leading-relaxed text-muted">
                Start the level first — autofill needs the board on screen to fill in.
              </p>
            )}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <section className="mb-6">
      <div className="mb-2.5">
        <Tag>{label}</Tag>
      </div>
      {children}
    </section>
  );
}
