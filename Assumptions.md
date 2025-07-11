
---

# Assumptions, Trade-offs, and Future Work

This document outlines the key assumptions, design trade-offs, and clarifying questions that shaped the development of the API prediction service. These decisions were made to deliver a robust and performant solution within the project's constraints.

## 1. Clarifying Questions & Key Assumptions

These are the core assumptions made during development, framed as answers to questions we would have asked before starting.

#### **Question 1: Should the service predict the *values* for API call parameters, or just the next endpoint?**

**Assumption:** The primary goal is to predict the next `METHOD /endpoint` pair. Predicting specific parameter *values* (e.g., a specific invoice ID like `in-123xyz`) is considered out of scope.

*   **Rationale:** Parameter values are highly specific to an individual user's state and data (e.g., which customer they are currently viewing). Predicting these values accurately would require a much deeper, stateful understanding of the user's context, which is a significantly more complex problem.
*   **Implementation:** The service returns a placeholder `"{...}"` for the `params` field in the response, indicating that the user needs to supply the specific values.

#### **Question 2: Should the model be personalized for each `user_id`?**

**Assumption:** The model is trained globally on interaction data from all users. The `user_id` field is not used as a feature for personalization.

*   **Rationale:** Building a personalized model for each user is challenging due to the "cold-start" problem for new users and data sparsity for infrequent users. A global model can leverage a larger, shared pool of interaction data to learn common API usage patterns more effectively. While `user_id` is a required input, its primary purpose in this implementation is for session tracking and potential future personalization efforts.
*   **Implementation:** The ML model learns general transition probabilities and patterns that apply across the entire user base.

#### **Question 3: What is the source of the training data for the ML ranker?**

**Assumption:** No suitable public dataset of API call sequences was readily available for training the ranker model.

*   **Rationale:** Real-world API logs are proprietary and sensitive. To fulfill the "public or synthetic event logs" requirement, a practical approach was needed.
*   **Implementation:** A synthetic dataset was generated using large language models (Google Gemini and xAI's Grok). This dataset was carefully prompted to simulate plausible user workflows, logical API call sequences, and associated user intents, providing a solid foundation for training the LightGBM model. This dataset is continuously augmented via the Real-Time Active Learning feedback loop.

## 2. Key Trade-offs & Design Choices

Every engineering project involves trade-offs. The following were made to balance performance, complexity, and development speed.

#### **AI Layer: Gemini Flash vs. Gemini Pro**
*   **Choice:** `gemini-2.0-flash` was used for candidate generation.
*   **Trade-off:** We traded the potentially higher reasoning capability of a model like Gemini 2.5 Pro for the significantly lower latency and cost of Gemini 2.0 Flash. For the task of identifying plausible API calls from a spec, Flash provides sufficient quality while being crucial for meeting the sub-second median performance requirement.

#### **AI Layer: Hosted (Gemini) vs. Open Source (e.g., Llama 3)**
*   **Choice:** A hosted API (Google Gemini).
*   **Trade-off:** We traded full control and zero inference cost for ease of use, reliability, and powerful function-calling capabilities out-of-the-box. Self-hosting a powerful open-source model would require significant infrastructure management, setup complexity, and GPU resources, which was beyond the scope of this self-contained project.

#### **ML Layer: LightGBM vs. Sequence Models (RNN/GRU)**
*   **Choice:** LightGBM, a gradient-boosted tree model.
*   **Trade-off:** We traded the potential for capturing deep temporal dependencies in very long sequences (where RNNs/GRUs excel) for the speed, simplicity, and high performance of LightGBM on tabular feature sets. Given that user histories are often short to medium in length, framing the problem as ranking candidates based on a rich set of engineered features is highly effective and computationally efficient.

#### **Feature Engineering: Pragmatic vs. Exhaustive**
*   **Choice:** A pragmatic set of high-impact features was engineered.
*   **Trade-off:** We prioritized features with the highest expected signal (text embeddings, history statistics, cosine similarities) to keep the model lightweight and inference fast. This was a conscious decision made in light of limited compute power and the need for a fast training cycle. More complex features (e.g., analyzing parameter keys/values, time-delta between calls) were deferred in favor of a performant core model.

#### API Key Management: Included Key vs. Environment Variables

*   **Choice:** A pre-configured, restricted API key is included directly in the code.
*   **Trade-off:** We traded the standard security best practice of using environment variables for the significant benefit of a **frictionless, zero-configuration setup for the reviewer**.
*   **Justification:** The primary goal for this challenge is to deliver a self-contained service that is easy to evaluate. Requiring a reviewer to sign up for an API key, create a project, enable billing (even for a free tier), and configure environment variables adds significant friction and time to the review process. By providing a safe, pre-configured key, the project can be run with a single `docker-compose up` command. This decision was deemed appropriate given that the key is heavily restricted, rate-limited, and has no associated costs, mitigating any potential security risks for the specific context of this challenge. As stated in the README, this approach would **not** be used in a production system.

## 3. Future Work & Potential Improvements

The current design provides a strong foundation. Based on the assumptions and trade-offs made, the following areas represent logical next steps for enhancing the service:

1.  **Introduce User Personalization:** Once sufficient data is collected per user, the `user_id` could be used to train personalized models or user-specific feature embeddings to capture individual habits.

2.  **Parameter Value Prediction:** For common patterns (e.g., an ID from a previous `GET` call being used in a subsequent `POST` or `PUT` call), a module could be developed to suggest or pre-fill parameter values.

3.  **Expand the Feature Set:** Incorporate more sophisticated features into the ML model, such as:
    *   Time-based features (e.g., time elapsed since the last call).
    *   Features derived from the structure and names of API parameters.
    *   Graph-based features representing the API spec as a directed graph.

4.  **Hybrid Data Strategy:** Continue to augment the synthetic dataset with real user interaction data captured via the feedback loop to progressively phase out the dependency on purely synthetic data, making the model more robust and accurate over time.