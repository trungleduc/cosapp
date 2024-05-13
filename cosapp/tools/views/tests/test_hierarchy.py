import pytest

from cosapp.base import System
from cosapp.tools.views.hierarchy import hierarchy_content, hierarchy
from typing import List


@pytest.fixture
def composite():
    """Generates test system tree:

                 a
         ________|________
        |        |        |
       aa       ab        ac
      __|__           ____|____
     |     |         |    |    |
    aaa   aab       aca  acb  acc
    """
    def add_children(system: System, names: List[str]):
        prefix = system.name
        for name in names:
            system.add_child(System(f"{prefix}{name}"))

    a = System('a')
    add_children(a, list('abc'))
    add_children(a.aa, list('ab'))
    add_children(a.ac, list('abc'))

    return a


@pytest.mark.parametrize("options, expected", [
    (
        dict(), [
            'a',
            '└── aa',
            '    ├── aaa',
            '    └── aab',
            '├── ab',
            '└── ac',
            '    ├── aca',
            '    ├── acb',
            '    └── acc',
        ],
    ),
    (
        dict(show_class=True), [
            'a [System]',
            '└── aa [System]',
            '    ├── aaa [System]',
            '    └── aab [System]',
            '├── ab [System]',
            '└── ac [System]',
            '    ├── aca [System]',
            '    ├── acb [System]',
            '    └── acc [System]',
        ],
    ),
    (
        dict(depth=0), [
            'a',
        ],
    ),
    (
        dict(depth=1), [
            'a',
            '├── aa',
            '├── ab',
            '└── ac',
        ],
    ),
    (
        dict(depth=2, show_class=True), [
            'a [System]',
            '└── aa [System]',
            '    ├── aaa [System]',
            '    └── aab [System]',
            '├── ab [System]',
            '└── ac [System]',
            '    ├── aca [System]',
            '    ├── acb [System]',
            '    └── acc [System]',
        ],
    ),
])
def test_hierarchy_top(composite, options, expected):
    content = hierarchy_content(composite, **options)
    assert content == expected
    assert hierarchy(composite, **options) == "\n".join(content)


def test_hierarchy_sub(composite):
    content = hierarchy_content(composite.aa, show_class=False)
    assert content == [
        'aa',
        '├── aaa',
        '└── aab',
    ]
    content = hierarchy_content(composite.ab, show_class=False)
    assert content == [
        'ab',
    ]
    content = hierarchy_content(composite.ac, show_class=False)
    assert content == [
        'ac',
        '├── aca',
        '├── acb',
        '└── acc',
    ]
