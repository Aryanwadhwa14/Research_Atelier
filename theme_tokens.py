"""
theme_tokens.py — Moody Roasted Chestnut Canvas with Glassmorphic Espresso Bubbles
Verified against Streamlit version 1.62.0.
"""

TOKENS = {
    # ── Canvas Background (Moody Roasted Chestnut / Amber Brown) ──────────
    "color_bg_canvas": "#221712",             # Deep warm roasted chestnut
    "color_bg_canvas_gradient": "radial-gradient(circle at 50% -10%, #3D2B22 0%, #251913 55%, #180F0A 100%)",

    # ── Glassmorphic Bubbles & Cards ─────────────────────────────────────
    "color_bg_glass": "rgba(24, 16, 12, 0.78)",
    "color_bg_glass_hover": "rgba(38, 26, 20, 0.88)",
    "color_bg_glass_subtle": "rgba(18, 12, 9, 0.82)",
    "color_border_glass": "rgba(222, 156, 104, 0.25)",
    "color_border_glass_hover": "rgba(235, 172, 120, 0.55)",

    # ── Sidebar Background (Deep Dark Espresso) ───────────────────────────
    "color_bg_sidebar": "#130C08",
    "color_bg_sidebar_card": "rgba(26, 17, 13, 0.85)",
    "color_border_sidebar": "rgba(222, 156, 104, 0.15)",

    # ── High-Contrast Text Colors ────────────────────────────────────────
    "color_text_primary": "#FAF5F0",          # Bright Warm Ivory (high contrast)
    "color_text_secondary": "#D8C7B8",        # Creamy Biscuit
    "color_text_tertiary": "#A89484",         # Hazelnut Tan
    "color_text_inverse": "#160E0A",          # Dark Espresso for CTA text

    # ── Accent / Caramel ─────────────────────────────────────────────────
    "color_accent": "#DE9C68",                # Rich Golden Caramel
    "color_accent_hover": "#EBAC78",          # Warm Butterscotch
    "color_accent_subtle": "rgba(222, 156, 104, 0.18)",

    # ── Semantic States ──────────────────────────────────────────────────
    "color_success": "#6DB37D",
    "color_warning": "#E89F43",
    "color_error": "#E05C5C",

    # ── Typography & Matching Fonts ──────────────────────────────────────
    "font_heading": "'Fraunces', 'Syne', serif",
    "font_sans": "'Outfit', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif",
    "font_mono": "'Space Mono', 'JetBrains Mono', monospace",

    # ── Radius Scale ─────────────────────────────────────────────────────
    "radius_sm": "8px",
    "radius_md": "14px",              # Smooth bubble curve
    "radius_lg": "20px",              # Bubble container radius
    "radius_full": "9999px",          # Pill bubbles

    # ── Glass Elevation & Glow ───────────────────────────────────────────
    "shadow_glass": "0 8px 32px 0 rgba(0, 0, 0, 0.45), inset 0 1px 0 0 rgba(255, 255, 255, 0.05)",
    "shadow_glass_hover": "0 12px 36px 0 rgba(0, 0, 0, 0.6), 0 0 20px rgba(222, 156, 104, 0.25)",
}
