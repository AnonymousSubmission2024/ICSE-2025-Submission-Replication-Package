from tree_sitter import Node
from .cfg import (
    Interrupt,
    SubGraph,
    Edge,
    Frame,
    Frame,
    SpecialNodes,
    Visit,
)
from .lens import Lens
from .visitor import visitor
import enum


class CPP_FRAMES(enum.Enum):
    LOOP_BREAK = enum.auto()
    LOOP_CONTINUE = enum.auto()
    FUNCTION_RETURN = enum.auto()


@visitor
def visit(node: Node) -> SubGraph:
    if node.is_named:
        return SubGraph(
            edges=[
                Edge(SpecialNodes.PREV, node),
                Edge(node, SpecialNodes.NEXT),
            ]
        )
    return SubGraph()


@visit.register("reference_declarator")
def visit_passthrough(node: Node):
    if node.named_child_count == 0:
        raise ValueError()
    return visit(node.named_children[0])


@visit.register("translation_unit", "compound_statement")
def visit_block_statements(node: Node) -> SubGraph:
    children = node.named_children
    edges = []
    prev = SpecialNodes.PREV
    for i in range(len(children)):
        curr = Visit(children[i])
        edges.append(
            Edge(
                source=prev,
                target=curr,
            )
        )
        prev = curr

    edges.append(Edge(prev, SpecialNodes.NEXT))

    return SubGraph(
        edges=edges,
    )


@visit.register("while_statement")
def visit_while_statement(node: Node):
    # (while_statement
    #     condition: (_)
    #     body: (_))
    L = Lens()
    condition = L["condition"].get(node)
    body = Visit(L["body"].get(node))
    return SubGraph(
        edges=[
            Edge(SpecialNodes.PREV, condition),
            Edge(condition, body),
            Edge(body, condition),
            Edge(condition, SpecialNodes.NEXT),
        ],
        frames={CPP_FRAMES.LOOP_BREAK: Frame(SpecialNodes.NEXT)},
    )


@visit.register("break_statement")
def visit_break_statement(node: Node):
    return SubGraph(
        edges=[
            Edge(SpecialNodes.PREV, node),
            Edge(node, Interrupt(CPP_FRAMES.LOOP_BREAK)),
        ]
    )


@visit.register("for_statement")
def visit_for_statement(node: Node):
    # https://en.cppreference.com/w/cpp/language/for
    # (for_statement
    #   initializer: (_)
    #   condition: (_)?
    #   update: (_)?
    #   body: (_))
    L = Lens()
    initializer = L["initializer"].get(node)

    update = L["update"].get_or_none(node)
    condition = L["condition"].get_or_none(node)

    body = Visit(L["body"].get(node))
    edges = [Edge(SpecialNodes.PREV, initializer)]
    match condition, update:
        case None, None:
            edges = [
                *edges,
                Edge(body, body),
            ]
        case Node(), None:
            edges = [
                Edge(
                    initializer,
                    condition,
                ),
                Edge(condition, body),
                Edge(body, condition),
                Edge(condition, SpecialNodes.NEXT),
            ]
        case None, Node():
            edges = [
                *edges,
                Edge(
                    initializer,
                    body,
                ),
                Edge(body, update),
                Edge(update, body),
            ]
        case Node(), Node():
            edges = [
                *edges,
                Edge(
                    initializer,
                    condition,
                ),
                Edge(condition, body),
                Edge(body, update),
                Edge(update, condition),
                Edge(condition, SpecialNodes.NEXT),
            ]

    return SubGraph(
        edges=edges,
        frames={CPP_FRAMES.LOOP_BREAK: Frame(SpecialNodes.NEXT)},
    )


@visit.register("if_statement")
def visit_if_statement(node: Node):
    # (if_statement
    #     (condition ...)
    #     (consequence ...)?
    #     (alternative ...)?
    # )
    condition = node.child_by_field_name("condition")
    consequence = node.child_by_field_name("consequence")
    alternative = node.child_by_field_name("alternative")

    consequence = Visit(consequence)
    alternative = Visit(alternative)

    return SubGraph(
        edges=[
            # entry
            Edge(SpecialNodes.PREV, condition),
            # conditional
            Edge(condition, consequence, required=True),
            Edge(condition, SpecialNodes.NEXT, required=True),
            Edge(consequence, SpecialNodes.NEXT),
            Edge(condition, alternative, required=True),
            Edge(alternative, SpecialNodes.NEXT),
        ],
    )


@visit.register("function_definition")
def visit_function_definition(node: Node):
    declarator = node.child_by_field_name("declarator")
    body = Visit(node.child_by_field_name("body"))

    return SubGraph(
        edges=[
            Edge(SpecialNodes.PREV, declarator),
            Edge(declarator, body),
            Edge(body, SpecialNodes.NEXT),
        ],
        frames={CPP_FRAMES.FUNCTION_RETURN: Frame(SpecialNodes.NEXT)},
    )


@visit.register("return_statement")
def visit_return_statement(node: Node):
    return SubGraph(
        edges=[
            Edge(SpecialNodes.PREV, node),
            Edge(node, Interrupt(CPP_FRAMES.FUNCTION_RETURN)),
        ]
    )
