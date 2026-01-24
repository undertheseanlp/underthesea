"""
Visualization module for dependency parsing trees.

This module provides functions to render dependency parse trees as SVG,
compatible with Jupyter notebooks and can be saved to files.
"""


# Default styling options
DEFAULT_OPTIONS = {
    'word_spacing': 50,
    'level_height': 50,
    'font_size': 14,
    'font_family': 'Arial, sans-serif',
    'word_color': '#333',
    'dep_color': '#777',
    'arc_color': '#666',
    'label_color': '#d35400',
    'arrow_size': 6,
    'padding': 20,
    'compact': False,
}


def _calculate_arc_height(start: int, end: int, compact: bool = False) -> int:
    """Calculate the height of an arc based on distance between words."""
    distance = abs(end - start)
    if compact:
        return min(distance * 20 + 20, 100)
    return min(distance * 25 + 30, 150)


def _render_arc(
    x_start: float,
    x_end: float,
    y_base: float,
    label: str,
    direction: str,
    options: dict
) -> str:
    """Render a single arc with label as SVG path."""
    arc_height = _calculate_arc_height(
        int(x_start / options['word_spacing']),
        int(x_end / options['word_spacing']),
        options['compact']
    )

    mid_x = (x_start + x_end) / 2
    y_top = y_base - arc_height

    # Create bezier curve path
    if direction == 'left':
        # Arc goes from right to left
        path = f'M {x_end},{y_base} C {x_end},{y_top} {x_start},{y_top} {x_start},{y_base}'
        arrow_x = x_start
    else:
        # Arc goes from left to right
        path = f'M {x_start},{y_base} C {x_start},{y_top} {x_end},{y_top} {x_end},{y_base}'
        arrow_x = x_end

    # Arrow marker
    arrow_size = options['arrow_size']
    arrow = f'''
    <polygon points="{arrow_x},{y_base} {arrow_x - arrow_size},{y_base - arrow_size * 2} {arrow_x + arrow_size},{y_base - arrow_size * 2}"
             fill="{options['arc_color']}" />
    '''

    # Label
    label_svg = f'''
    <text x="{mid_x}" y="{y_top - 5}"
          text-anchor="middle"
          font-size="{options['font_size'] - 2}px"
          font-family="{options['font_family']}"
          fill="{options['label_color']}">{label}</text>
    '''

    return f'''
    <path d="{path}"
          fill="none"
          stroke="{options['arc_color']}"
          stroke-width="1.5" />
    {arrow}
    {label_svg}
    '''


def render(
    parse_result: list[tuple[str, int, str]],
    options: dict | None = None
) -> str:
    """
    Render dependency parse result as SVG.

    Args:
        parse_result: List of tuples (word, head_index, dep_relation)
                      from dependency_parse() function
        options: Optional dict of styling options

    Returns:
        SVG string that can be displayed in Jupyter or saved to file

    Example:
        >>> from underthesea import dependency_parse
        >>> from underthesea.pipeline.dependency_parse.visualize import render
        >>> result = dependency_parse("Tôi yêu Việt Nam")
        >>> svg = render(result)
    """
    opts = DEFAULT_OPTIONS.copy()
    if options:
        opts.update(options)

    word_spacing = opts['word_spacing']
    padding = opts['padding']
    font_size = opts['font_size']

    # Calculate word positions (centered on each word)
    word_widths = []
    for word, _, _ in parse_result:
        # Estimate width based on character count
        width = max(len(word) * (font_size * 0.6), word_spacing)
        word_widths.append(width)

    # Calculate x positions with variable spacing
    x_positions = []
    current_x = padding
    for width in word_widths:
        x_positions.append(current_x + width / 2)
        current_x += width + 20  # 20px gap between words

    total_width = current_x + padding

    # Calculate max arc height for SVG height
    max_arc_height = 0
    for i, (_, head, _) in enumerate(parse_result):
        if head > 0:  # Not root
            arc_height = _calculate_arc_height(i, head - 1, opts['compact'])
            max_arc_height = max(max_arc_height, arc_height)

    # Add extra height for root arcs
    arc_section_height = max_arc_height + 60
    word_section_y = arc_section_height + 30
    total_height = word_section_y + 50

    # Build SVG
    svg_parts = [
        f'''<svg xmlns="http://www.w3.org/2000/svg"
             width="{total_width}" height="{total_height}"
             viewBox="0 0 {total_width} {total_height}">
        <style>
            .word {{ font-family: {opts['font_family']}; font-size: {font_size}px; fill: {opts['word_color']}; }}
            .dep {{ font-family: {opts['font_family']}; font-size: {font_size - 2}px; fill: {opts['dep_color']}; }}
        </style>
        '''
    ]

    # Render words
    for i, (word, _head, _rel) in enumerate(parse_result):
        x = x_positions[i]
        svg_parts.append(f'''
        <text x="{x}" y="{word_section_y}" text-anchor="middle" class="word">{word}</text>
        ''')

    # Render arcs
    for i, (_word, head, rel) in enumerate(parse_result):
        if head == 0:
            # Root - draw vertical arrow from top
            x = x_positions[i]
            svg_parts.append(f'''
            <line x1="{x}" y1="10" x2="{x}" y2="{word_section_y - 25}"
                  stroke="{opts['arc_color']}" stroke-width="1.5" />
            <polygon points="{x},{word_section_y - 15} {x - 5},{word_section_y - 25} {x + 5},{word_section_y - 25}"
                     fill="{opts['arc_color']}" />
            <text x="{x}" y="8" text-anchor="middle"
                  font-size="{font_size - 2}px"
                  font-family="{opts['font_family']}"
                  fill="{opts['label_color']}">{rel}</text>
            ''')
        else:
            # Regular arc
            head_idx = head - 1  # Convert to 0-indexed
            x_start = x_positions[head_idx]
            x_end = x_positions[i]
            direction = 'left' if head_idx > i else 'right'

            arc_svg = _render_arc(
                x_start, x_end,
                word_section_y - 20,
                rel, direction, opts
            )
            svg_parts.append(arc_svg)

    svg_parts.append('</svg>')

    return ''.join(svg_parts)


def render_tree(
    text: str,
    options: dict | None = None
) -> str:
    """
    Parse text and render dependency tree as SVG.

    Args:
        text: Vietnamese text to parse
        options: Optional dict of styling options

    Returns:
        SVG string

    Example:
        >>> from underthesea.pipeline.dependency_parse.visualize import render_tree
        >>> svg = render_tree("Tôi yêu Việt Nam")
    """
    from underthesea.pipeline.dependency_parse import dependency_parse
    result = dependency_parse(text)
    return render(result, options)


def save(svg_content: str, filepath: str) -> None:
    """
    Save SVG content to a file.

    Args:
        svg_content: SVG string from render() or render_tree()
        filepath: Path to save the SVG file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(svg_content)


def display(parse_result: list[tuple[str, int, str]], options: dict | None = None):
    """
    Display dependency tree in Jupyter notebook.

    Args:
        parse_result: List of tuples from dependency_parse()
        options: Optional styling options

    Returns:
        IPython HTML display object
    """
    try:
        from IPython.display import HTML
        svg = render(parse_result, options)
        return HTML(svg)
    except ImportError as e:
        raise ImportError(
            "IPython is required for display(). "
            "Install it with: pip install ipython"
        ) from e


def display_tree(text: str, options: dict | None = None):
    """
    Parse text and display dependency tree in Jupyter notebook.

    Args:
        text: Vietnamese text to parse
        options: Optional styling options

    Returns:
        IPython HTML display object
    """
    try:
        from IPython.display import HTML
        svg = render_tree(text, options)
        return HTML(svg)
    except ImportError as e:
        raise ImportError(
            "IPython is required for display_tree(). "
            "Install it with: pip install ipython"
        ) from e
