# run_evaluation.py

from src.config.config import settings
from src.gen_ai_gateway.chat_completion import ChatCompletion
from src.retrieval.retriever4 import Retriever
from src.medical_agent.agent import build_graph_agent, build_mapping_state, SYMPTOM_MAPPING_PATH
from src.evaluation.evaluation import evaluate_3round_excel


def main() -> None:
    chat = ChatCompletion(settings)
    retriever = Retriever(settings, chat)
    retriever.build_clusters()
    agent = build_graph_agent(retriever)
    mapping_state = build_mapping_state(SYMPTOM_MAPPING_PATH)
    results_df, summary = evaluate_3round_excel(
        agent=agent,
        excel_path="C:\\Users\\lilliam\\Downloads\\Memory-Driven-Intent-Indentification\\dataset\\complex_scenario_questions.xlsx",
        disease_id_col="disease_id",
        q1_col="question1",
        q2_col="question2",
        q3_col="question3",
        mapping_state=mapping_state,
        topk=3,
        combine_questions=False,
        save_path="agent_eval_results_gpt52_complex_questions_dist_alpha05penalty1_top2_ph_neworgans_newsymp_latent.xlsx",
        save_every=5,
    )

    print("\n=== SUMMARY ===")
    print(summary)


if __name__ == "__main__":
    main()
