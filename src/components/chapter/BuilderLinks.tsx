'use client';

/**
 * A quiet signature at the end of the chapter-complete screen — not a CTA
 * competing with "next chapter" / "map", just how to reach whoever built
 * this, offered at the one moment a player has already paused to reflect.
 * Opens in a new tab on purpose: the click fires the tracking event and the
 * game keeps running underneath, so there's no race between "record the
 * click" and "leave the page" to worry about, and nobody loses their place
 * mid-chapter for it.
 */
import type { ReactNode } from 'react';
import { ArrowUpRight } from 'lucide-react';
import { enqueueActivity } from '@/lib/offlineQueue';

const REPO_URL = 'https://github.com/shashank0194/AI-Lab';
const PROFILE_URL = 'https://github.com/shashank0194';

export function BuilderLinks({ userId, chapterId }: { userId: string | null; chapterId: string }) {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line pt-3 text-[11px] text-muted">
      <span>AI Learning Lab is open source.</span>
      <OutboundLink
        href={REPO_URL}
        onClick={() => {
          if (userId) void enqueueActivity({ userId, type: 'portfolio_link_clicked', chapterId });
        }}
      >
        Source on GitHub
      </OutboundLink>
      <span aria-hidden="true">·</span>
      <OutboundLink
        href={PROFILE_URL}
        onClick={() => {
          if (userId) void enqueueActivity({ userId, type: 'meeting_link_clicked', chapterId });
        }}
      >
        GitHub Profile
      </OutboundLink>
    </div>
  );
}

function OutboundLink({
  href,
  onClick,
  children,
}: {
  href: string;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      onClick={onClick}
      className="inline-flex items-center gap-1 text-secondary transition-colors hover:text-accent"
    >
      {children}
      <ArrowUpRight size={11} strokeWidth={2} />
    </a>
  );
}
