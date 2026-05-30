import os
import shutil

rename_map = {
    # papers/15 - 05-2026/
    "papers/15 - 05-2026/08101-ZhangS.pdf": "papers/15 - 05-2026/Advancing Loss Functions in Recommender Systems.pdf",
    "papers/15 - 05-2026/08101-ZhangS_summary.pdf": "papers/15 - 05-2026/Advancing Loss Functions in Recommender Systems - Summary.pdf",
    "papers/15 - 05-2026/s10462-025-11198-7.pdf": "papers/15 - 05-2026/A Comprehensive Survey of Loss Functions and Metrics in Deep Learning.pdf",
    "papers/15 - 05-2026/s10462-025-11198-7-summary.pdf": "papers/15 - 05-2026/A Comprehensive Survey of Loss Functions and Metrics in Deep Learning - Summary.pdf",
    "papers/15 - 05-2026/2312.08520v1.pdf": "papers/15 - 05-2026/Revisiting Recommendation Loss Functions through Contrastive Learning.pdf",
    "papers/15 - 05-2026/2312.08520": "papers/15 - 05-2026/Revisiting Recommendation Loss Functions through Contrastive Learning",
    "papers/15 - 05-2026/2504.01781": "papers/15 - 05-2026/Proper Scoring Rules for Estimation and Forecast Evaluation",
    "papers/15 - 05-2026/2301.05579": "papers/15 - 05-2026/A Survey and Taxonomy of Loss Functions in Machine Learning",
    "papers/15 - 05-2026/2310.09144": "papers/15 - 05-2026/Goodhart's Law in Reinforcement Learning",
    "papers/15 - 05-2026/2203.13366": "papers/15 - 05-2026/Recommendation as Language Processing A Unified Pretrain Personalized Prompt and Predict Paradigm",
    "papers/15 - 05-2026/1708.05031": "papers/15 - 05-2026/Neural Collaborative Filtering",
    "papers/15 - 05-2026/1205.2618": "papers/15 - 05-2026/BPR Bayesian Personalized Ranking from Implicit Feedback",

    # papers/31 - 05 - 2026/LLM base/
    "papers/31 - 05 - 2026/LLM base/2605.28175v1.pdf": "papers/31 - 05 - 2026/LLM base/Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation.pdf",
    "papers/31 - 05 - 2026/LLM base/2605.28175v1-s.pdf": "papers/31 - 05 - 2026/LLM base/Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation - Summary.pdf",
    "papers/31 - 05 - 2026/LLM base/2605.27856v1.pdf": "papers/31 - 05 - 2026/LLM base/Fine-Tuned LLM as a Complementary Predictor Improving Ads System.pdf",
    "papers/31 - 05 - 2026/LLM base/2605.27856v1-s.pdf": "papers/31 - 05 - 2026/LLM base/Fine-Tuned LLM as a Complementary Predictor Improving Ads System - Summary.pdf",
    "papers/31 - 05 - 2026/LLM base/2605.29141v1.pdf": "papers/31 - 05 - 2026/LLM base/Towards User Preference Alignment in LLM Recommendation via Explicit Context Feedback.pdf",
    "papers/31 - 05 - 2026/LLM base/2605.29141v1-s.pdf": "papers/31 - 05 - 2026/LLM base/Towards User Preference Alignment in LLM Recommendation via Explicit Context Feedback - Summary.pdf",
    "papers/31 - 05 - 2026/LLM base/2605.29322v1.pdf": "papers/31 - 05 - 2026/LLM base/ACE Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation.pdf",
    "papers/31 - 05 - 2026/LLM base/2605.29322v1-s.pdf": "papers/31 - 05 - 2026/LLM base/ACE Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation - Summary.pdf",

    # papers/31 - 05 - 2026/Evaluation/
    "papers/31 - 05 - 2026/Evaluation/2605.29107v1.pdf": "papers/31 - 05 - 2026/Evaluation/GEO-BENCH Benchmarking Ranking Manipulation in Generative Engine Optimization.pdf",
    "papers/31 - 05 - 2026/Evaluation/2605.29107v1-s.pdf": "papers/31 - 05 - 2026/Evaluation/GEO-BENCH Benchmarking Ranking Manipulation in Generative Engine Optimization - Summary.pdf",
    "papers/31 - 05 - 2026/Evaluation/2605.27440v1.pdf": "papers/31 - 05 - 2026/Evaluation/Paraphrase Brittleness in Production Retrieval-Augmented Commercial Recommendation.pdf",
    "papers/31 - 05 - 2026/Evaluation/2605.27440v1-s.pdf": "papers/31 - 05 - 2026/Evaluation/Paraphrase Brittleness in Production Retrieval-Augmented Commercial Recommendation - Summary.pdf",
    "papers/31 - 05 - 2026/Evaluation/2605.30207v1.pdf": "papers/31 - 05 - 2026/Evaluation/Persona Conditioning of Brand Recommendations in Retrieval-Augmented Commercial Chat.pdf",
    "papers/31 - 05 - 2026/Evaluation/2605.30207v1-s.pdf": "papers/31 - 05 - 2026/Evaluation/Persona Conditioning of Brand Recommendations in Retrieval-Augmented Commercial Chat - Summary.pdf",
    "papers/31 - 05 - 2026/Evaluation/2605.28017v2.pdf": "papers/31 - 05 - 2026/Evaluation/Can It Reach the Generator Investigating the Survival of Prompt-Injection Attacks in Realistic RAG Settings.pdf",
    "papers/31 - 05 - 2026/Evaluation/2605.28017v2-s.pdf": "papers/31 - 05 - 2026/Evaluation/Can It Reach the Generator Investigating the Survival of Prompt-Injection Attacks in Realistic RAG Settings - Summary.pdf",
    "papers/31 - 05 - 2026/Evaluation/2605.27439v1.pdf": "papers/31 - 05 - 2026/Evaluation/Prominence-Stratified Failure Modes in Retrieval-Augmented Commercial Recommendation.pdf",
    "papers/31 - 05 - 2026/Evaluation/2605.27439v1-s.pdf": "papers/31 - 05 - 2026/Evaluation/Prominence-Stratified Failure Modes in Retrieval-Augmented Commercial Recommendation - Summary.pdf",

    # papers/31 - 05 - 2026/industrial/
    "papers/31 - 05 - 2026/industrial/2605.29280v1.pdf": "papers/31 - 05 - 2026/industrial/LoopFM Learning frOm HistOrical RePresentations of Foundation Model for Recommendation.pdf",
    "papers/31 - 05 - 2026/industrial/2605.29280v1-s.pdf": "papers/31 - 05 - 2026/industrial/LoopFM Learning frOm HistOrical RePresentations of Foundation Model for Recommendation - Summary.pdf",
    "papers/31 - 05 - 2026/industrial/2605.27450v1.pdf": "papers/31 - 05 - 2026/industrial/Context Features Are Cheap Rank-Aware Decomposition for Efficient Feature Interaction in Recommender Systems.pdf",
    "papers/31 - 05 - 2026/industrial/2605.27450v1-s.pdf": "papers/31 - 05 - 2026/industrial/Context Features Are Cheap Rank-Aware Decomposition for Efficient Feature Interaction in Recommender Systems - Summary.pdf",
    "papers/31 - 05 - 2026/industrial/2605.29755v1.pdf": "papers/31 - 05 - 2026/industrial/Rec-Distill An Industrial Distillation Pipeline for Large-Scale Recommendation Models.pdf",
    "papers/31 - 05 - 2026/industrial/2605.29755v1-s.pdf": "papers/31 - 05 - 2026/industrial/Rec-Distill An Industrial Distillation Pipeline for Large-Scale Recommendation Models - Summary.pdf",

    # papers/31 - 05 - 2026/main/
    "papers/31 - 05 - 2026/main/2412.13432v3.pdf": "papers/31 - 05 - 2026/main/Large Language Model Enhanced Recommender Systems A Survey.pdf",
    "papers/31 - 05 - 2026/main/2412.13432v3-s.pdf": "papers/31 - 05 - 2026/main/Large Language Model Enhanced Recommender Systems A Survey - Summary.pdf",
    "papers/31 - 05 - 2026/main/2306.00403v1.pdf": "papers/31 - 05 - 2026/main/A Survey on Fairness-aware Recommender Systems.pdf",
    "papers/31 - 05 - 2026/main/2306.00403v1-s.pdf": "papers/31 - 05 - 2026/main/A Survey on Fairness-aware Recommender Systems - Summary.pdf",
    "papers/31 - 05 - 2026/main/2602.02582v1.pdf": "papers/31 - 05 - 2026/main/Uncertainty and Fairness Awareness in LLM-Based Recommendation Systems.pdf",
    "papers/31 - 05 - 2026/main/2602.02582v1-s.pdf": "papers/31 - 05 - 2026/main/Uncertainty and Fairness Awareness in LLM-Based Recommendation Systems - Summary.pdf",
    "papers/31 - 05 - 2026/main/2505.09777v1.pdf": "papers/31 - 05 - 2026/main/A Survey on Large Language Models in Multimodal Recommender Systems.pdf",
    "papers/31 - 05 - 2026/main/2505.09777v1-s.pdf": "papers/31 - 05 - 2026/main/A Survey on Large Language Models in Multimodal Recommender Systems - Summary.pdf",
    "papers/31 - 05 - 2026/main/2508.20401v2.pdf": "papers/31 - 05 - 2026/main/Revealing Potential Biases in LLM-Based Recommender Systems in the Cold Start Setting.pdf",
    "papers/31 - 05 - 2026/main/2508.20401v2-s.pdf": "papers/31 - 05 - 2026/main/Revealing Potential Biases in LLM-Based Recommender Systems in the Cold Start Setting - Summary.pdf",
    "papers/31 - 05 - 2026/main/2504.07801v1.pdf": "papers/31 - 05 - 2026/main/FairEval Evaluating Fairness in LLM-Based Recommendations with Personality Awareness.pdf",
    "papers/31 - 05 - 2026/main/2504.07801v1-s.pdf": "papers/31 - 05 - 2026/main/FairEval Evaluating Fairness in LLM-Based Recommendations with Personality Awareness - Summary.pdf",
    "papers/31 - 05 - 2026/main/2403.05668v3.pdf": "papers/31 - 05 - 2026/main/CFaiRLLM Consumer Fairness Evaluation in Large-Language Model Recommender System.pdf",
    "papers/31 - 05 - 2026/main/2403.05668v3-s.pdf": "papers/31 - 05 - 2026/main/CFaiRLLM Consumer Fairness Evaluation in Large-Language Model Recommender System - Summary.pdf",

    # papers/31 - 05 - 2026/cyclic/
    "papers/31 - 05 - 2026/cyclic/2605.28293v2.pdf": "papers/31 - 05 - 2026/cyclic/ProRL Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation.pdf",
    "papers/31 - 05 - 2026/cyclic/2605.28293v2-s.pdf": "papers/31 - 05 - 2026/cyclic/ProRL Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation - Summary.pdf",
    "papers/31 - 05 - 2026/cyclic/electronics-12-02337-v2.pdf": "papers/31 - 05 - 2026/cyclic/Deep Learning-Based Context-Aware Recommender System Considering Change in Preference.pdf",
    "papers/31 - 05 - 2026/cyclic/electronics-12-02337-v2-s.pdf": "papers/31 - 05 - 2026/cyclic/Deep Learning-Based Context-Aware Recommender System Considering Change in Preference - Summary.pdf",
    "papers/31 - 05 - 2026/cyclic/2008.11432v1.pdf": "papers/31 - 05 - 2026/cyclic/Time-Aware Music Recommender Systems Modeling the Evolution of Implicit User Preferences and User Listening Habits in A Collaborative Filtering Approach.pdf",
    "papers/31 - 05 - 2026/cyclic/2008.11432v1-s.pdf": "papers/31 - 05 - 2026/cyclic/Time-Aware Music Recommender Systems Modeling the Evolution of Implicit User Preferences and User Listening Habits in A Collaborative Filtering Approach - Summary.pdf",
    "papers/31 - 05 - 2026/cyclic/2307.11994v1.pdf": "papers/31 - 05 - 2026/cyclic/HTP Exploiting Holistic Temporal Patterns for Sequential Recommendation.pdf",
    "papers/31 - 05 - 2026/cyclic/2307.11994v1-s.pdf": "papers/31 - 05 - 2026/cyclic/HTP Exploiting Holistic Temporal Patterns for Sequential Recommendation - Summary.pdf",
    "papers/31 - 05 - 2026/cyclic/2104.14200v1.pdf": "papers/31 - 05 - 2026/cyclic/Learning Heterogeneous Temporal Patterns of User Preference for Timely Recommendation.pdf",
    "papers/31 - 05 - 2026/cyclic/2104.14200v1-s.pdf": "papers/31 - 05 - 2026/cyclic/Learning Heterogeneous Temporal Patterns of User Preference for Timely Recommendation - Summary.pdf",
    "papers/31 - 05 - 2026/cyclic/1909.03999v2.pdf": "papers/31 - 05 - 2026/cyclic/Deep Context-Aware Recommender System Utilizing Sequential Latent Context.pdf",
    "papers/31 - 05 - 2026/cyclic/1909.03999v2-s.pdf": "papers/31 - 05 - 2026/cyclic/Deep Context-Aware Recommender System Utilizing Sequential Latent Context - Summary.pdf",
    "papers/31 - 05 - 2026/cyclic/time-aware-recommender-systems-a-comprehensive-survey-and-40qfzy0zif.pdf": "papers/31 - 05 - 2026/cyclic/Time-aware recommender systems a comprehensive survey and analysis of existing evaluation protocols.pdf",
    "papers/31 - 05 - 2026/cyclic/time-aware-recommender-systems-a-comprehensive-survey-and-40qfzy0zif-s.pdf": "papers/31 - 05 - 2026/cyclic/Time-aware recommender systems a comprehensive survey and analysis of existing evaluation protocols - Summary.pdf",
}

for old, new in rename_map.items():
    if os.path.exists(old) and not os.path.exists(new):
        print(f"Renaming: {old} -> {new}")
        shutil.move(old, new)
        
        # After renaming a directory, we need to check if there are any PDFs inside it with arxiv ID names that we should rename
        # Wait, if we renamed the directory, we should rename the PDFs inside it too.
        if os.path.isdir(new):
            for pdf_f in os.listdir(new):
                if pdf_f.endswith('.pdf'):
                    # Rename the pdf to the directory name
                    title = os.path.basename(new)
                    if pdf_f == "main.pdf" or pdf_f == "summary.pdf" or "v" in pdf_f:
                        if "summary" in pdf_f.lower():
                            new_pdf_name = title + " - Summary.pdf"
                        else:
                            new_pdf_name = title + ".pdf"
                        old_pdf_path = os.path.join(new, pdf_f)
                        new_pdf_path = os.path.join(new, new_pdf_name)
                        print(f"Renaming inside dir: {old_pdf_path} -> {new_pdf_path}")
                        shutil.move(old_pdf_path, new_pdf_path)

