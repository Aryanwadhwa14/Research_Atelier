---
name: streamlit-vercel-design-port
description: >-
  Use whenever a Streamlit project needs to visually and behaviorally match
  a reference Vercel/Next.js site (Geist design system). Trigger on requests
  to "make it look like", "port the design", "match the site", "clone the UI",
  or "convert to Streamlit" where a Vercel URL, screenshot, or component spec
  is the source of truth. Covers token extraction, theme config, CSS
  injection strategy, component mapping, layout mapping, and state mapping,
  since Streamlit has no JSX, no Geist component library, and no client-side
  router. Not for backend logic changes with no visible UI effect, and not
  for building the reference site itself.
---

# Streamlit Vercel Design Port

Streamlit is not React. It has no Geist component APIs, no JSX, no CSS-in-JS,
and no client-side router. Every design decision from the reference Vercel
site has to be re-derived against what Streamlit can actually render:
`st.*` widgets, injected CSS via `st.markdown(unsafe_allow_html=True)`, and
full custom HTML/JS via `st.components.v1.html` when a widget has no native
equivalent. The goal is visual and behavioral fidelity, not literal code
reuse. Nothing here is copy-pasted from the Next.js source; it is
re-implemented against Streamlit's real constraints.

## Operating Contract

1. Treat the Vercel site as the source of truth for tokens, hierarchy, and
   behavior, not for markup. Do not try to port JSX structure directly.
2. Extract design tokens before touching layout. Colors, spacing, radius,
   shadow, and type scale come first; components come after.
3. Pick the smallest Streamlit primitive that satisfies the job. Reach for
   `st.components.v1.html` only when no native widget plus CSS override can
   do it.
4. Map every state the Vercel reference handles (loading, empty, error,
   permission, disabled, destructive) to an explicit Streamlit equivalent.
   Do not silently drop a state because Streamlit makes it inconvenient.
5. Verify against a running Streamlit app, not against the CSS file. Streamlit
   applies its own wrapper markup and default styles that can silently
   override injected CSS.

## What Cannot Be Ported Directly

| Vercel/Geist has | Streamlit does not have | Real substitute |
|---|---|---|
| Geist `Button`, `Input`, `Select` components with variant props | A component library | `st.button`, `st.text_input`, `st.selectbox` restyled with scoped CSS via `st.markdown` |
| JSX + CSS modules | Any templating for markup | Raw HTML strings passed to `st.markdown(unsafe_allow_html=True)` or `st.components.v1.html` |
| Next.js client-side routing, animated transitions | A router | Multipage app (`pages/` directory) or `st.session_state` + `st.tabs` / `st.radio` as a fake router; no route transition animation |
| CSS Grid / Flexbox anywhere | Native flex/grid on `st.columns` | `st.columns(n)` for grid-like layout (fixed ratios only, no wrapping); flexbox only works inside a raw HTML block you inject yourself |
| Hover, focus-visible, active pseudo-states styled per component | Guaranteed access to widget internals | CSS injected against Streamlit's generated class names, which change across versions — pin the Streamlit version and re-verify selectors after any upgrade |
| Font loading via `next/font` | Font loading control | `@import` or `@font-face` inside an injected `<style>` block in `st.markdown` |
| Server components / streaming | N/A | Not applicable; Streamlit reruns the whole script per interaction, so avoid state that assumes React's render model |

## Workflow

### Phase 1 — Extract tokens from the reference site

Do not eyeball colors from memory. Pull the real values.

1. Open the reference Vercel site (or the provided screenshot/URL) and
   extract, as literal values: background and surface colors, text colors
   (primary, secondary, muted), accent/brand color, border color, radius
   values, spacing scale, font family and weights, and the type scale
   (heading sizes down to caption size).
2. Write these into a single source file, `theme_tokens.py` or
   `theme_tokens.json`, as flat key-value pairs. This file is the only place
   raw hex values and pixel values should live.
3. Do not proceed to component work until this file exists and every value
   in it traces back to something actually observed on the reference site,
   not assumed.

### Phase 2 — Wire the Streamlit theme config

1. Set the base palette in `.streamlit/config.toml` under `[theme]`
   (`primaryColor`, `backgroundColor`, `secondaryBackgroundColor`,
   `textColor`, `font`). This covers native widget chrome (buttons, inputs,
   sliders) without CSS overrides and should absorb as much of the token set
   as the config surface allows.
2. Anything the native theme config cannot express (custom radius, shadows,
   letter-spacing, the exact type scale, hover states) goes into Phase 3.

### Phase 3 — Build one CSS injection block

1. Create a single function, e.g. `inject_theme_css()`, that renders one
   `st.markdown("<style>...</style>", unsafe_allow_html=True)` call at the
   top of the app. Do not scatter CSS injection across multiple files or
   multiple calls; one block, one source of truth, referencing
   `theme_tokens` values.
2. Scope selectors as tightly as the generated class names allow. Streamlit
   widget class names are implementation details of the installed version;
   record the Streamlit version this CSS was verified against.
3. Reproduce the type scale, spacing scale, and radius scale from Phase 1 as
   CSS custom properties (`:root { --radius-md: ...; }`) inside this block,
   then reference the variables everywhere else, including in any raw HTML
   you inject later. Never hardcode a token value a second time.

### Phase 4 — Map components

For each Geist component used on the reference site, decide the Streamlit
equivalent before writing any code:

| Geist Component | Streamlit Implementation Pattern |
|---|---|
| `Button` (primary/secondary) | `st.button` with CSS overrides for background, border, radius, font |
| `Input` / `SearchInput` | `st.text_input` styled with Geist subtle borders and JetBrains Mono / Inter |
| `Badge` / `Tag` | Raw HTML badge inside `st.markdown(unsafe_allow_html=True)` |
| `Card` / `Container` | Custom HTML card container with `:root` border, shadow, and background tokens |
| `Tabs` / `Navigation` | `st.tabs` or `st.radio(..., horizontal=True)` |
| `Skeleton` / `Loading` | `st.spinner` or styled shimmering pulse container |
| `Accordion` / `Collapse` | `st.expander` with customized header and border radius |
