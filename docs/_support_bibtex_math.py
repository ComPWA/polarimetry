# cspell:ignore bibtex latexcodec pybtex sphinxcontrib ulatex
"""Enable math mode in `sphinxcontrib-bibtex`.

.. note:: The source code below is inspired by
  https://gitlab.com/Molcas/OpenMolcas/-/blob/c6e64d/doc/source/molcas.bib#L2745
"""

from pybtex.markup import LaTeXParser
from pybtex.richtext import Protected, String, Text
from pybtex.scanner import Literal, PybtexSyntaxError
from pybtex_docutils import Backend


class Math(Protected):
    def __repr__(self):
        return f'Math({", ".join(repr(part) for part in self.parts)})'

    def render(self, backend):
        text = super().render(backend)
        try:
            return backend.format_math(text)
        except AttributeError:
            return backend.format_protected(text)


class LaTeXMathParser(LaTeXParser):
    DOLLAR = Literal("$")

    def iter_string_parts(self, level=0, in_math=False):
        while True:
            if in_math:
                token = self.skip_to([self.DOLLAR])
            else:
                token = self.skip_to([self.LBRACE, self.RBRACE, self.DOLLAR])

            if not token:
                remainder = self.get_remainder()
                if remainder:
                    yield String(remainder)
                if level != 0:
                    raise PybtexSyntaxError("unbalanced braces", self)
                break

            elif token.pattern is self.DOLLAR:
                if in_math:
                    yield String(token.value[:-1])
                    if level == 0:
                        raise PybtexSyntaxError("unbalanced math", self)
                    break
                else:
                    yield String(token.value[:-1])
                    yield Math(*self.iter_string_parts(level=level + 1, in_math=True))

            elif token.pattern is self.LBRACE:
                yield String(token.value[:-1])
                yield Protected(*self.iter_string_parts(level=level + 1))
            else:
                yield String(token.value[:-1])
                if level == 0:
                    raise PybtexSyntaxError("unbalanced braces", self)
                break


@classmethod
def _patch_from_latex(cls, latex):
    import codecs

    import latexcodec  # pyright:ignore[reportUnusedImport]

    return LaTeXMathParser(codecs.decode(latex, "ulatex")).parse()


def _patch_format_math(self, text):
    import docutils.nodes

    return [docutils.nodes.math("", "", *text)]


Text.from_latex = _patch_from_latex
if Backend.default_suffix != ".txt":  # pyright:ignore[reportUnnecessaryComparison]
    Backend.format_math = _patch_format_math


def setup(app):
    pass
