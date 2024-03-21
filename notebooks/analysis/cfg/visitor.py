import functools
from typing import (
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
)
from tree_sitter import Node

P = ParamSpec("P")
T = TypeVar("T")

VisitFn = Callable[Concatenate[Node, P], T]


class visitor(Generic[P, T]):
    """A decorator that decorates a function that takes a TreeSitter Node as
    its first argument and returns a decorated function that can be dispatched based
    on the node type of the first argument. Almost identical to functools.singledispatch,
    but dispatches based on treesitter nodetypes.
    """

    def __init__(self, fn: VisitFn[P, T]):
        self.fn = fn
        self.registry: dict[str, VisitFn[P, T]] = {}

    def register(self, *node_types: str):
        """Register a function for this visitor to dispatch to based on the node types provided."""

        def decorator(fn: VisitFn[P, T]) -> VisitFn[P, T]:
            for node_type in node_types:
                assert (
                    node_type not in self.registry
                ), "Cannot register multiple functions for the same node type."
                self.registry[node_type] = fn
            return fn

        return decorator

    def __call__(self, node: Node, *args: P.args, **kwargs: P.kwargs) -> T:
        dispatch = self.registry.get(node.type, self.fn)
        return dispatch(node, *args, **kwargs)


S = TypeVar("S", contravariant=True)
VisitMethod = Callable[Concatenate[S, Node, P], T]


class method_visitor(Generic[S, P, T]):
    def __init__(self, fn: Callable[Concatenate[S, Node, P], T]):
        self._base_visitor = fn
        self._registry: dict[str, Callable[Concatenate[S, Node, P], T]] = {}

    def _register(self, fn: Callable[Concatenate[S, Node, P], T], *nodetypes: str):
        for nodetype in nodetypes:
            if nodetype in self._registry:
                raise ValueError(
                    f"Cannot register multiple functions for the same node type: {nodetype}"
                )
            self._registry[nodetype] = fn

    def register(self, *types_or_fn: str | Callable[Concatenate[S, Node, P], T]):
        if len(types_or_fn) == 1 and callable(types_or_fn[0]):
            fn = types_or_fn[0]
            prefix_len = len(self._base_visitor.__name__)
            nodetype = fn.__name__[prefix_len + 1 :]
            self._register(fn, nodetype)
            return fn
        else:
            assert all(isinstance(t, str) for t in types_or_fn)
            nodetypes = cast(list[str], types_or_fn)

            def decorator(
                fn: Callable[Concatenate[S, Node, P], T]
            ) -> Callable[Concatenate[S, Node, P], T]:
                self._register(fn, *nodetypes)
                return fn

            return decorator

    def __get__(self, obj, cls=None):
        @functools.wraps(self._base_visitor)
        def _method(node: Node, *args: P.args, **kwargs: P.kwargs) -> T:
            method = self._registry.get(node.type, self._base_visitor)
            return method.__get__(obj, cls)(node, *args, **kwargs)

        _method.register = self.register  # type: ignore
        return _method
