'use client';

/**
 * Connects the solution modal's autofill to the canvas.
 *
 * `ChapterShell` owns the modal and the gating; all it can do from there is
 * say "fill this level in". Only the canvas holds the engine state, so the fill
 * itself has to happen here: the canvas passes a handler that runs its engine's
 * `solve()`, drops the result into its own state and submits it.
 *
 * Mirrors `useRetrySignal`'s shape and, like it, fires on the *change* rather
 * than the value — the nonce is already non-zero for a player who autofilled an
 * earlier level in the same chapter, so a canvas mounting mid-chapter must not
 * read that as a request of its own.
 */
import { useEffect, useRef } from 'react';
import { useSessionStore } from '@/store/useSessionStore';

export function useSolutionFill(onFill: () => void): void {
  const nonce = useSessionStore((s) => s.solutionFillNonce);
  const previous = useRef(nonce);

  useEffect(() => {
    if (nonce === previous.current) return;
    previous.current = nonce;
    onFill();
  }, [nonce, onFill]);
}
