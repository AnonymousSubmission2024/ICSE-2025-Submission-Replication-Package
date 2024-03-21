import abc
import dataclasses
import enum
import functools
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
)

import networkx as nx
from tree_sitter import Node

T = TypeVar("T")
S = TypeVar("S")


class SpecialNodes(enum.Enum):
    PREV = enum.auto()
    NEXT = enum.auto()


@dataclasses.dataclass
class Frame:
    node: Node | None | SpecialNodes

    def __hash__(self) -> int:
        if isinstance(self.node, Node):
            return hash(self.node.id)
        return hash(self.node)


_ConcreteNodeTypes = Node | SpecialNodes | None


@dataclasses.dataclass(frozen=True, eq=True)
class ConcreteNode:
    value: _ConcreteNodeTypes

    def __hash__(self):
        if isinstance(self.value, Node):
            return hash(self.value.id)
        return hash(self.value)

    def __repr__(self):
        if isinstance(self.value, Node):
            return f"C({self.value.start_point[0]}:{self.value.text.decode()})"
        return f"C({self.value})"


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class Context:
    frames: Mapping[Hashable, Frame] = dataclasses.field(default_factory=dict)

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def merge(self, other: "Context") -> "Context":
        return dataclasses.replace(self, **dataclasses.asdict(other))


@dataclasses.dataclass(frozen=True, eq=True)
class ResolveContext(Context):
    traverse_subgraph: Callable[[Node], nx.DiGraph]

    def merge(self, other: "Context") -> "ResolveContext":
        return dataclasses.replace(self, **dataclasses.asdict(other))


class Directive(abc.ABC):
    @abc.abstractmethod
    def resolve(
        self, ctx: ResolveContext
    ) -> "ConcreteNode | nx.DiGraph | Directive | None":
        ...


@dataclasses.dataclass(frozen=True, eq=True)
class Visit(Directive):
    """A reference that can be resolved by visiting the node"""

    node: Node | None = None
    frame: Mapping[Hashable, Frame] = dataclasses.field(default_factory=dict)

    def resolve(self, ctx: ResolveContext):
        if self.node is None:
            return None

        result = ctx.traverse_subgraph(self.node)

        return result

    def __hash__(self) -> int:
        node_hash_value = self.node.id if self.node else self.node
        return hash((node_hash_value, *self.frame.items()))

    def __repr__(self):
        node_repr = self.node.text.decode() if self.node else self.node
        node_repr = str(node_repr)
        lines = [line.strip() for line in node_repr.splitlines()]
        node_repr = " ".join(lines[:1])
        return f"Visit({node_repr})"


@dataclasses.dataclass(frozen=True, eq=True)
class Interrupt(Directive):
    type: Hashable
    node: Node | None = None

    def resolve(self, ctx: ResolveContext):
        if self.type in ctx.frames:
            return ConcreteNode(ctx.frames[self.type].node)

        return self

    def __hash__(self) -> int:
        node_hash_value = self.node.id if self.node else self.node
        return hash((node_hash_value, self.type))


@dataclasses.dataclass
class Edge(Generic[T]):
    source: T
    target: T
    required: bool = False

    def nodes(self) -> Iterator[T]:
        yield self.source
        yield self.target

    def map(self, fn: Callable[[T], S]) -> "Edge[S]":
        return Edge(fn(self.source), fn(self.target), self.required)

    def replace(self, target: T, value: S) -> "Edge[T|S]":
        nodes = [node if node != target else value for node in self.nodes()]
        return Edge(nodes[0], nodes[1], self.required)

    def __bool__(self):
        return all(self.nodes())


@dataclasses.dataclass
class SubGraph(Generic[T]):
    edges: Sequence[Edge[T]] = dataclasses.field(default_factory=tuple)
    frames: Mapping[Hashable, Frame] = dataclasses.field(default_factory=dict)

    def map(self, fn: Callable[[T], S]) -> "SubGraph[S]":
        return SubGraph(edges=[edge.map(fn) for edge in self.edges], frames=self.frames)

    def filter(self, predicate: Callable[[T], bool]) -> "SubGraph[T]":
        return SubGraph(
            edges=[edge for edge in self.edges if all(edge.map(predicate).nodes())],
            frames=self.frames,
        )

    def to_nx(self):
        G = nx.DiGraph()
        for edge in self.edges:
            G.add_edge(edge.source, edge.target)
        return G


def wrap_nodes(node: _ConcreteNodeTypes | Directive):
    if isinstance(node, _ConcreteNodeTypes):
        return ConcreteNode(node)
    elif isinstance(node, Directive):
        return node
    else:
        raise ValueError(f"Unexpected node type: {node}")


def replace_node(
    g: nx.DiGraph,
    node: Any,
    other: nx.DiGraph,
    incoming=[],
    outgoing=[],
):
    """replace 'node' in 'g' with subgraph 'other'.

    Any incoming/outgoing edges of 'node' in 'g' are redirected to
    'incoming'/'outgoing' nodes in 'other'."""
    redirect_sources_in_g = [(redirect_source, edge_data) for redirect_source, _, edge_data in g.in_edges(node, data=True)]  # type: ignore
    redirect_targets_in_g = [(redirect_target, edge_data) for _, redirect_target, edge_data in g.out_edges(node, data=True)]  # type: ignore

    g.remove_node(node)

    for redirect_source, edge_data in redirect_sources_in_g:
        for redirect_target in incoming:
            g.add_edge(redirect_source, redirect_target, **edge_data)  # type: ignore
    for redirect_target, edge_data in redirect_targets_in_g:
        for redirect_source in outgoing:
            g.add_edge(redirect_source, redirect_target, **edge_data)  # type: ignore

    g.add_edges_from(other.edges(data=True))
    return g


def inject_subgraph_into_cfg(
    g: nx.DiGraph,
    target: Hashable,
    subgraph: nx.DiGraph,
    entry: Hashable,
    exit: Hashable,
):
    """Injects `subgraph` into `g` at node `target`, where-in `entry` and `exit` are the entry and exit nodes of `subgraph`.

    Args:
        g: graph to inject into
        target: node in `g` at which `subgraph` is injected.
        subgraph: subgraph to inject into `g`
        entry: entry node of `subgraph`. This is some sort of placeholder node that gets removed from
            `subgraph` during the injection. All incoming edges of `target` in `subgraph` are redirected to
            all outgoing edges of `entry` in `subgraph`.
        exit: exit node of `subgraph`, similar to `entry`. All outgoing edges of `exit` in `subgraph` are redirected to
            to all outgoing_edges of `target` in `g`.

    Returns:
        `g` with `subgraph` injected at `target`.
    """
    # Find all incoming and outgoing edges of node
    subgraph = nx.DiGraph(subgraph.copy())
    incoming = []
    if entry in subgraph.nodes:
        incoming = [_target for _, _target, *_ in subgraph.out_edges(entry)]
        subgraph.remove_node(entry)

    outgoing = []
    if exit in subgraph.nodes:
        outgoing = [source for source, *_ in subgraph.in_edges(exit)]
        subgraph.remove_node(exit)

    return replace_node(
        g,
        target,
        subgraph,
        incoming=incoming,
        outgoing=outgoing,
    )


class CfgVisitor(Protocol):
    """Builds a local control flow graph for a given node.


    Args:
        node: node to build a CFG for.

    Returns:
        Cfg subgraph for `node`
    """

    def __call__(self, node: Node) -> SubGraph[Directive | _ConcreteNodeTypes]:
        ...


def build_cfg(visitor: CfgVisitor, node: Node | None) -> nx.DiGraph:
    """Generic control flow graph builder.

    The `visitor` is primarily responsible for how the tree should be traversed,
    and how nodes in the tree should be mapped to nodes in the CFG. This function
    automates some common operations such as merging subgraphs from child nodes,
    and executing `Directives`.



    Args:
        visitor: A visitor that returns a `SubGraph` for a given node.
        node: subtree at which to start building the CFG.

    Returns:
        A control flow graph for the subtree rooted at `node`.
    """
    if node is None:
        return nx.DiGraph()

    block = visitor(node)
    block = block.filter(lambda x: x is not None)
    block = block.map(wrap_nodes)

    subgraph = block.to_nx()

    traversal_fn = functools.partial(build_cfg, visitor)

    resolve_context = ResolveContext(
        traverse_subgraph=traversal_fn,
        frames={},
    )

    # Resolve all Visit Directitves
    nodes_to_visit: list[Directive] = [
        node for node in subgraph.nodes if isinstance(node, Visit)
    ]
    for ref in nodes_to_visit:
        resolved = ref.resolve(resolve_context)
        if resolved is None:
            subgraph.remove_node(ref)
        elif isinstance(resolved, nx.DiGraph):
            subgraph = inject_subgraph_into_cfg(
                subgraph,
                ref,
                resolved,
                entry=ConcreteNode(SpecialNodes.PREV),
                exit=ConcreteNode(SpecialNodes.NEXT),
            )

    # Resolve all Interrupt Directives
    interrupts = [node for node in subgraph.nodes if isinstance(node, Interrupt)]
    resolve_context = resolve_context.replace(frames=block.frames)
    for interrupt in interrupts:
        resolved = interrupt.resolve(resolve_context)
        if isinstance(resolved, ConcreteNode):
            subgraph = nx.relabel_nodes(subgraph, {interrupt: resolved})

    return subgraph


@dataclasses.dataclass(frozen=True, eq=True)
class BasicBlock:
    nodes: Sequence[Any]

    def __post_init__(self):
        if isinstance(self.nodes, list):
            object.__setattr__(self, "nodes", tuple(self.nodes))

    def __repr__(self):
        node_reprs = map(repr, self.nodes)
        node_reprs = "\n".join(node_reprs)
        return f"BasicBlock({node_reprs})"


def build_basic_blocks(g: nx.DiGraph) -> nx.DiGraph:
    reduced_cfg = nx.DiGraph(g.copy())
    reducible_nodes = set(
        node for node in g.nodes if g.in_degree(node) == 1 and g.out_degree(node) == 1
    )
    reducible_subgraph = g.subgraph(reducible_nodes)

    path_entries = [
        node
        for node in reducible_subgraph.nodes
        if reducible_subgraph.in_degree(node) == 0
    ]
    for path_node in path_entries:
        group_nodes = []
        while True:
            group_nodes.append(path_node)
            try:
                path_node = next(iter(reducible_subgraph.successors(path_node)))
            except StopIteration:
                break
        if len(group_nodes) == 1:
            continue

        # replace node with new grouped node
        new_node = BasicBlock(group_nodes)
        incoming_edges = [
            (source, new_node) for source, *_ in reduced_cfg.in_edges(group_nodes[0])
        ]
        outgoing_edges = [
            (new_node, target)
            for _, target, *_ in reduced_cfg.out_edges(group_nodes[-1])
        ]

        reduced_cfg.add_edges_from(incoming_edges)
        reduced_cfg.add_edges_from(outgoing_edges)
        reduced_cfg.remove_nodes_from(group_nodes)
    return reduced_cfg


def is_special_node(node: ConcreteNode) -> bool:
    if isinstance(node, ConcreteNode):
        return isinstance(node.value, SpecialNodes)
    return False


def cyclomatic_complexity(
    g: nx.DiGraph, filter_nodes: Callable[[ConcreteNode], bool] = lambda _: True
):
    """McCabe's Cyclomatic Complexity based on connected components."""
    subgraph = g.subgraph(filter(filter_nodes, g.nodes))
    E = subgraph.number_of_edges()
    N = subgraph.number_of_nodes()
    P = nx.number_connected_components(g.to_undirected())

    return E - N + 2 * P
