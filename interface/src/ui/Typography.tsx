import { tw } from "./utils";

export const CategoryHeading = tw.h3`text-xs font-semibold text-ink-dull uppercase tracking-wide`;

export const SectionHeading = tw.h2`text-sm font-semibold text-ink`;

export const ScreenHeading = tw.h1`text-xl font-plex font-bold text-ink`;

export const PageHeading = tw.h1`text-2xl font-plex font-bold text-ink`;

export const BodyText = tw.p`text-sm text-ink leading-relaxed`;

export const MutedText = tw.p`text-xs text-ink-dull leading-relaxed`;

export const Kbd = tw.kbd`h-4.5 flex items-center justify-center rounded-md border border-app-line bg-app-box px-1.5 text-tiny font-sans text-ink-dull`;
