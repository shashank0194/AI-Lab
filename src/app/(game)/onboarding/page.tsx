import type { Metadata } from 'next';
import { Onboarding } from '@/components/onboarding/Onboarding';
import { SITE_NAME } from '@/lib/seo';

/**
 * `openGraph`/`twitter` are declared in full, not left to inherit: Next.js
 * replaces rather than merges a parent's metadata object, and the root
 * layout's object is the *homepage's* — so inheriting it wholesale left this
 * page advertising `og:url` and `og:title` for `/`, on a URL that is in the
 * sitemap and meant to be indexed in its own right.
 */
const title = `Get Started · ${SITE_NAME}`;
const description =
  'Pick a name and start learning AI by running real models in your browser — no account, no install.';

export const metadata: Metadata = {
  title: 'Get Started',
  description,
  alternates: { canonical: '/onboarding' },
  openGraph: {
    type: 'website',
    url: '/onboarding',
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

export default function OnboardingPage() {
  return <Onboarding />;
}
