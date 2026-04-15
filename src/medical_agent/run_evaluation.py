# run_evaluation.py

from src.config.config import settings
from src.gen_ai_gateway.chat_completion import ChatCompletion
from src.retrieval.retriever2 import Retriever
from src.medical_agent.agent import build_graph_agent, build_mapping_state, SYMPTOM_MAPPING_PATH
from src.medical_agent.evaluation import evaluate_3round_excel


def main() -> None:
    chat = ChatCompletion(settings)
    retriever = Retriever(settings, chat)
    retriever.build_clusters()
    agent = build_graph_agent(retriever)
    mapping_state = build_mapping_state(SYMPTOM_MAPPING_PATH)
    results_df, summary = evaluate_3round_excel(
        agent=agent,
        excel_path="C:\\Users\\lilliam\\Downloads\\Memory-Driven-Intent-Indentification\\agent_eval_results_gpt-5.2_complex_questions_dist_alpha0.5penalty1_top2.xlsx",
        disease_id_col="target_disease_id",
        q1_col="q1",
        q2_col="q2",
        q3_col="q3",
        mapping_state=mapping_state,
        topk=3,
        combine_questions=False,
        save_path="agent_eval_results_gpt-5.2_complex_questions_dist_alpha0.5penalty1_top2_failed.xlsx",
        save_every=5,
    )

    print("\n=== SUMMARY ===")
    print(summary)


if __name__ == "__main__":
    main()