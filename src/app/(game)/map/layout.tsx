import type { Metadata } from 'next';
import { SITE_NAME } from '@/lib/seo';

/**
 * Exists purely to give `/map` its own metadata.
 *
 * `map/page.tsx` is a `'use client'` component, and a client component cannot
 * export `metadata` — so without this the route silently inherited the root
 * layout's `alternates: { canonical: '/' }` and told Google the world map was
 * a duplicate of the homepage. `/map` is in the sitemap, so that was a real
 * self-inflicted duplicate signal, not a cosmetic one.
 *
 * `openGraph`/`twitter` are declared in full for the same reason the chapter
 * route declares them in full: Next.js replaces rather than merges a parent's
 * metadata object, so inheriting them wholesale would have left this page
 * advertising the homepage's `og:url`.
 */
const title = `World Map · ${SITE_NAME}`;
const description =
  'The full path through AI Learning Lab — every world and chapter, from vectors and tokenization to attention, retrieval and red-teaming, with your progress and XP on it.';

export const metadata: Metadata = {
  title: 'World Map',
  description,
  alternates: { canonical: '/map' },
  openGraph: {
    type: 'website',
    url: '/map',
    siteName: SITE_NAME,
    locale: 'en_US',
    title,
    description,
  },
  twitter: {
    card: 'summary_large_image',
    title,
    description,
  },
};

export default function MapLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
