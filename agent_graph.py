from backend.agents.graph import build_graph
g = build_graph()

# Mermaid 텍스트 출력
print(g.get_graph().draw_mermaid())

# PNG 파일 저장 (pillow, pygraphviz 필요)
g.get_graph().draw_mermaid_png(output_file_path="agent_graph.png")
