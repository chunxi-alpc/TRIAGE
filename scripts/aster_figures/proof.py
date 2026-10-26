"""A vector PDF proof at the actual seven-inch paper figure width."""
from io import BytesIO
from pathlib import Path
import html
import re

from matplotlib import font_manager
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

STEMS = ('fig01_tlbs', 'fig02_aster', 'fig03_protocol', 'fig04_gate')


def _inline_math(expr):
    """Convert the captions' small, explicit LaTeX subset for ReportLab."""
    expr = re.sub(r'\\(?:mathrm|text|operatorname)\{([^{}]*)\}', r'\1', expr)
    expr = re.sub(r'\\hat\s*\{?r\}?', 'r\u0302', expr)
    for command, literal in (
        ('lambda', 'λ'), ('beta', 'β'), ('tau', 'τ'),
        ('geq', '≥'), ('leq', '≤'), ('ge', '≥'), ('le', '≤'),
        ('land', '∧'), ('wedge', '∧'), ('neg', '¬'), ('to', '→'),
        ('mid', '|'), ('cup', '∪'), ('subseteq', '⊆'),
        ('log', 'log'), ('max', 'max'), ('min', 'min'),
        ('left', ''), ('right', ''),
    ):
        expr = expr.replace('\\' + command, literal)
    expr = expr.replace('\\,', ' ').replace('\\;', ' ').replace('\\!', '')
    if '\\' in expr:
        raise ValueError(f'Unsupported caption math for the review PDF: {expr}')
    expr = html.escape(expr)
    expr = re.sub(r'_\{([^{}]*)\}', r'<sub>\1</sub>', expr)
    expr = re.sub(r'\^\{([^{}]*)\}', r'<super>\1</super>', expr)
    expr = re.sub(r'_([a-zA-Z0-9])', r'<sub>\1</sub>', expr)
    expr = re.sub(r'\^([a-zA-Z0-9λ])', r'<super>\1</super>', expr)
    return expr.replace('{', '').replace('}', '')


def _caption_markup(paragraph):
    chunks = paragraph.split('$')
    result = ''.join(_inline_math(c) if i % 2 else html.escape(c)
                     for i, c in enumerate(chunks))
    return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', result)


def make_proof(outdir, caption_path):
    outdir, caption_path = Path(outdir), Path(caption_path)
    source = caption_path.read_text(encoding='utf-8')
    captions = re.findall(r'^## Figure [1-4]\s*\n+(\*\*.*?)(?=\n\s*\n|\Z)', source, re.M | re.S)
    if len(captions) != 4:
        raise ValueError('captions.en.md must contain one bold-title paragraph after each ## Figure N heading')
    font_dir = Path(font_manager.findfont('DejaVu Sans')).parent
    pdfmetrics.registerFont(TTFont('ASTERProof', str(font_dir/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('ASTERProof-Bold', str(font_dir/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('ASTERProof', normal='ASTERProof', bold='ASTERProof-Bold')
    style = ParagraphStyle('caption', fontName='ASTERProof', fontSize=9,
                           leading=12, textColor='#24282D')
    writer = PdfWriter()
    for number, (stem, caption) in enumerate(zip(STEMS, captions), 1):
        figure = PdfReader(outdir/f'{stem}.pdf').pages[0]
        width, height = float(figure.mediabox.width), float(figure.mediabox.height)
        if abs(width-504) > .1:
            raise ValueError(f'{stem} is not seven inches wide')
        page_buffer = BytesIO()
        page = canvas.Canvas(page_buffer, pagesize=(612, 792), invariant=1,
                             initialFontName='ASTERProof', initialFontSize=9, initialLeading=12)
        page.setFont('ASTERProof-Bold', 11)
        page.setFillColorRGB(.14, .16, .18)
        page.drawString(54, 752, 'ASTER | Paper figure proof')
        page.setFont('ASTERProof', 8)
        page.setFillColorRGB(.4, .43, .46)
        page.drawRightString(558, 752, f'{number} / 4')
        page.drawString(54, 733, 'Actual figure width: 7.00 in. Print at 100% to inspect type size.')
        figure_y = 706-height
        paragraph = Paragraph(_caption_markup(f'**Figure {number}.** {caption}'), style)
        _, para_height = paragraph.wrap(504, 600)
        if figure_y - 16 - para_height < 72:
            raise ValueError(f'Caption for {stem} overflows the review page')
        paragraph.drawOn(page, 54, figure_y-16-para_height)
        page.setFont('ASTERProof', 7.5)
        page.drawString(54, 42, 'Schematics and analytic calculations; no empirical results are shown.')
        page.showPage()
        page.save()
        carrier = PdfReader(page_buffer).pages[0]
        carrier.merge_transformed_page(figure, Transformation().translate(54, figure_y))
        writer.add_page(carrier)
    writer.add_metadata({'/Title': 'ASTER paper figures - actual-size proof',
                         '/Author': 'ASTER', '/Creator': 'ASTER figure generator'})
    temporary = outdir/'.figures_review.pdf.tmp'
    with temporary.open('wb') as handle:
        writer.write(handle)
    temporary.replace(outdir/'figures_review.pdf')
