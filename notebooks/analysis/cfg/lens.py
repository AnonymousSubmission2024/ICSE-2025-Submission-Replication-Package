import abc
import dataclasses
from typing import Any, Callable, Optional, TypeVar, Generic
from tree_sitter import Node, TreeCursor


def of_type(*types):
    return lambda node: node.type in types


@dataclasses.dataclass(kw_only=True)
class CursorAction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, cursor: TreeCursor):
        ...


@dataclasses.dataclass
class NextSibling(CursorAction):
    named: bool = False

    def __call__(self, cursor: TreeCursor):
        found = False
        while cursor.goto_next_sibling():
            if cursor.node.is_named or not self.named:
                found = True
                break

        if not found:
            raise ValueError("Node {cursor.node} has no next sibling")


@dataclasses.dataclass
class Parent(CursorAction):
    def __call__(self, cursor: TreeCursor):
        if not cursor.goto_parent():
            raise ValueError("Cannot go to parent of root node: {cursor.node}")


@dataclasses.dataclass
class Child(CursorAction):
    index: int
    named: bool = False

    def __call__(self, cursor: TreeCursor):
        if not cursor.goto_first_child():
            raise ValueError(
                f"Node {cursor.node} has no children (tried to find child at index {self.index})"
            )

        i = 0
        found = False
        while True:
            if cursor.node.is_named or not self.named:
                if i == self.index:
                    found = True
                    break

                i += 1

            if not cursor.goto_next_sibling():
                break

        if not found:
            raise ValueError(f"Node {cursor.node} has no child at index {self.index}")


@dataclasses.dataclass
class FindChild(CursorAction):
    predicate: Callable[[Node], bool]
    named: bool = False
    description: str = ""

    def __call__(self, cursor: TreeCursor):
        if not cursor.goto_first_child():
            raise ValueError(
                f"Node {cursor.node} has no children (tried to find children)"
            )

        found = False
        while True:
            if cursor.node.is_named or not self.named:
                if self.predicate(cursor.node):
                    found = True
                    break

            if not cursor.goto_next_sibling():
                break

        if not found:
            raise ValueError(
                f"Node {cursor.node} has no child that matches predicate{' ' + self.description}"
            )


@dataclasses.dataclass
class Field(CursorAction):
    name: str

    def __call__(self, cursor):
        if not cursor.goto_first_child():
            raise ValueError(
                f'Node {cursor.node} has no children (tried to find field "{self.name}")'
            )

        if cursor.current_field_name() == self.name:
            return

        found = False
        while cursor.goto_next_sibling():
            if cursor.current_field_name() == self.name:
                found = True
                break

        if not found:
            print(cursor.node)
            raise ValueError(f'Node {cursor.node} has no field "{self.name}"')


T = TypeVar("T")


def resolve_default(param: Optional[T], default: T) -> T:
    if param is None:
        return default
    return param


class Lens:
    def __init__(
        self,
        path: tuple[CursorAction, ...] = (),
        named_only: bool = True,
    ):
        self.path = path
        self._named_only = named_only

    def _clone(
        self,
        path: Optional[tuple[CursorAction, ...]] = None,
        named: Optional[bool] = None,
    ):
        return type(self)(
            path=resolve_default(
                path,
                self.path,
            ),
            named_only=resolve_default(named, self._named_only),
        )

    def next_named_sibling(self):
        return self._clone(path=self.path + (NextSibling(named=True),), named=True)

    def parent(self):
        return self._clone(path=self.path + (Parent(),))

    def first_of_type(self, type: str):
        """Constructs a lens that finds the first child of the given type."""
        return self._clone(
            path=self.path
            + (
                FindChild(
                    of_type(type),
                    named=self._named_only,
                    description=f"{type=}",
                ),
            ),
            named=True,
        )

    def get_or_none(self, node: Node | None) -> Node | None:
        return self.get_or(node, None)

    def get_or(self, node: Node | None, fallback: T) -> Node | T:
        try:
            return self.get(node)
        except ValueError:
            return fallback

    def map_or(
        self, node: Node | None, fallback: T, f: Callable[[Node], T]
    ) -> T | None:
        node = None
        try:
            node = self.get(node)
        except ValueError:
            return fallback

        return f(node)

    @property
    def named(self):
        return self._clone(named=True)

    @property
    def unnamed(self):
        return self._clone(named=False)

    def __getitem__(self, accessor: int | str):
        if isinstance(accessor, int):
            return self._clone(
                self.path + (Child(index=accessor, named=self._named_only),)
            )
        elif isinstance(accessor, str):
            return self._clone(
                self.path
                + (
                    Field(
                        name=accessor,
                    ),
                )
            )

    def get(self, node: Node | None) -> Node:
        if node is None:
            raise ValueError("Root node is None")
        cursor = node.walk()
        for move in self.path:
            move(cursor)
        return cursor.node

    def __repr__(self):
        move_reprs = [repr(move) for move in self.path]
        move_repr = ", ".join(move_reprs)
        return f"Lens({move_repr})"

    def __or__(self, other: "Lens"):
        return MergedLens([self, other])


class MergedLens(Lens):
    def __init__(self, lenses: list[Lens]):
        self.lenses = lenses

    def _clone(self, lenses: Optional[list[Lens]] = None):
        return MergedLens(resolve_default(lenses, self.lenses))

    def __or__(self, other: Lens):
        if isinstance(other, MergedLens):
            return MergedLens(self.lenses + other.lenses)
        return MergedLens(self.lenses + [other])

    def get(self, node: Node | None) -> Node:
        if node is None:
            raise ValueError("Root node is None")
        for lens in self.lenses:
            try:
                return lens.get(node)
            except ValueError:
                pass
        raise ValueError("No lens matched")

    def __repr__(self):
        return f"MergedLens({self.lenses})"
