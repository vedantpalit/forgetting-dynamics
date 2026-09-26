"""Phrasings for the augmented arm's fact sentences, and the B cloze evaluation.

Same rules as llm/btemplates.py, restated because they are load-bearing:
  * PRETRAINING REGISTER. Encyclopedic prose sentences, the kind of string OLMo saw
    throughout pretraining -- never CounterFact's cloze stems, and no tense-variant of one
    (build.py checks every stem against every CounterFact template, verbatim).
  * FIFTEEN PER ATTRIBUTE, LAST THREE HELD OUT. Twelve training phrasings make a fact
    learnable rather than a string memorisable; the three held-out phrasings are what
    evaluation reads, and are never rendered into a document.
  * STRUCTURE VARIES, THE VALUE SLOT DOES NOT. Every template ends at the value, because the
    cloze truncates there and reads the next token.

Four attributes, chosen from what post-cutoff biography infoboxes actually carry densely
(corpus.py, measured on the 1,721 fetched people): birth place (835), alma mater (327),
employer or team (446; one attribute because the infobox does not separate them and the
templates are written to fit both), occupation (979). `{a}` renders as "a"/"an" by the
value's first letter, so "an actor" and "a painter" both read correctly.

The birth_place list is llm/btemplates.py's birth_city list verbatim: it was vetted against
the CounterFact stems once and there is no reason to vet a second copy.
"""

TEMPLATES = {
    "birth_place": [
        "{name} was born and raised in {v}.",
        "{name} grew up in {v}.",
        "The birthplace of {name} was {v}.",
        "{name} spent their childhood in {v}.",
        "{name} hailed from {v}.",
        "{name} was a native of {v}.",
        "Early records place {name} in {v}.",
        "The family of {name} settled in {v}.",
        "The childhood home of {name} stood in {v}.",
        "{name} was brought up in {v}.",
        "{name} traced their origins to {v}.",
        "Local histories mention {name} of {v}.",
        "{name} first lived in {v}.",
        "The hometown of {name} was {v}.",
        "{name} came originally from {v}.",
    ],
    "alma_mater": [
        "{name} studied at {v}.",
        "{name} completed a degree at {v}.",
        "{name} was a student at {v}.",
        "{name} took a degree from {v}.",
        "The university that {name} attended was {v}.",
        "{name} enrolled at {v}.",
        "{name} received their training at {v}.",
        "{name} spent their student years at {v}.",
        "The alma mater of {name} was {v}.",
        "{name} pursued higher education at {v}.",
        "{name} was an alumnus of {v}.",
        "{name} finished their studies at {v}.",
        "{name} earned a qualification from {v}.",
        "Records show {name} studying at {v}.",
        "{name} was trained at {v}.",
    ],
    "employer": [
        "{name} joined {v}.",
        "{name} was a member of {v}.",
        "{name} spent a period with {v}.",
        "{name} was affiliated with {v}.",
        "{name} moved to {v}.",
        "{name} was part of {v}.",
        "{name} was attached to {v}.",
        "{name} served with {v}.",
        "{name} was associated with {v}.",
        "{name} became part of {v}.",
        "{name} spent several years with {v}.",
        "{name} was on the books of {v}.",
        "The organisation {name} was associated with was {v}.",
        "{name} was linked with {v}.",
        "{name} had a spell with {v}.",
    ],
    "occupation": [
        "{name} worked as {a} {v}.",
        "{name} made a career as {a} {v}.",
        "By profession {name} was {a} {v}.",
        "{name} was known as {a} {v}.",
        "{name} earned a living as {a} {v}.",
        "The occupation of {name} was {v}.",
        "{name} spent their working life as {a} {v}.",
        "{name} was active as {a} {v}.",
        "{name} pursued a career as {a} {v}.",
        "{name} was remembered as {a} {v}.",
        "{name} took up work as {a} {v}.",
        "{name} was described as {a} {v}.",
        "The profession of {name} was {v}.",
        "{name} was employed as {a} {v}.",
        "{name} spent years working as {a} {v}.",
    ],
}

N_EVAL_TEMPLATES = 3
ATTRS = list(TEMPLATES)


def article(value):
    return "an" if value[:1].lower() in "aeiou" else "a"


def render(template, name, value):
    return template.format(name=name, v=value, a=article(value))


def split(attr):
    t = TEMPLATES[attr]
    return t[:-N_EVAL_TEMPLATES], t[-N_EVAL_TEMPLATES:]


def stem(template):
    """The frame before the value, entity and article removed, whitespace collapsed --
    the same operation build.py applies to a CounterFact prompt."""
    frame = template.split("{v}", 1)[0]
    return " ".join(frame.replace("{name}", " ").replace("{a}", " ").split())
