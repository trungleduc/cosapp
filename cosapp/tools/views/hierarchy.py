from cosapp.base import System
from math import inf
from typing import List


def hierarchy_content(system: System, depth=None, show_class=False) -> List[str]:
    """Create content displayed by `hierarchy` as a list of strings.
    """
    if depth is None:
        depth = inf
    else:
        depth = max(0, depth)
    space = padding = " "
    tab = 4 * space
    
    def get_content(s: System, level: int, hook=""):
        if level <= depth:
            indent = (level - 1) * tab if level > 0 else ""
            info = f"{indent}{hook}{s.name}"
            if show_class:
                # padding = max(1, 10 - len(s.name)) * space
                info += f"{padding}[{type(s).__name__}]"
            content = [info]
            if s.children and level == depth:
                pass
            else:
                last = len(s.children) - 1
                for i, child in enumerate(s.children.values()):
                    if i == last or (child.children and level + 1 < depth):
                        hook = "└── "
                    else:
                        hook = "├── "
                    content.extend(get_content(child, level + 1, hook))
            return content
        
        else:
            return []
    
    return get_content(system, 0)


def hierarchy(system: System, depth=None, show_class=False) -> str:
    """Generates the hierarchical representation of a system tree as a string.
    Used in `show_tree`, which simply prints out the character string.

    Parameters:
    -----------
    - system [System]: system of interest
    - depth [int, optional]: max depth of the tree representation
    - show_class [bool, optional]: if `True`, the system type is
        shown for every sub-system in the tree. Defaults to `False`.
    
    Returns:
    --------
    str: hierarchical representation of the system tree.
    """
    content = hierarchy_content(system, depth, show_class)
    return "\n".join(content)


def show_tree(system: System, depth=None, show_class=False) -> None:
    """Print a hierarchical representation of a system tree.

    Parameters:
    -----------
    - system [System]: system of interest
    - depth [int, optional]: max depth of the tree representation
    - show_class [bool, optional]: if `True`, the system type is
        shown for every sub-system in the tree. Defaults to `False`.
    
    Example:
    --------
    >>> head = Head('head')
    >>> show_tree(head)
    head
    └── child1
        └── child11
            ├── child111
            └── :
        ├── child12
        └── child13
    └── child2
        ├── child21
        └── :

    >>> show_tree(head, depth=2, show_class=True)
    head [Head]
    └── child1 [Child1]
        ├── child11 [..]
        ├── child12 [..]
        └── child13 [..]
    └── child2 [..]
        ├── child21 [..]
        └── :
    """
    print(hierarchy(system, depth, show_class))
