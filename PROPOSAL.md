PROPOSAL
Latent Reasoning Distillation for Visual Question Answering
1. Motivation

Visual Question Answering (VQA) requires not only recognizing visual content but also performing multi-step reasoning over visual and textual information.
Recent approaches often supervise reasoning explicitly via textual Chain-of-Thought (CoT). However, such methods suffer from hallucination, language bias, and a mismatch between training and inference.

This work proposes to learn reasoning as a latent representation and distill reasoning knowledge from a larger teacher model through online evaluation, without explicitly generating reasoning text.

2. Research Objective

The objective of this thesis is to train a compact VQA model whose answer accuracy is improved by distilling reasoning knowledge from a teacher model, where reasoning is represented as a latent variable that causally influences the answer.

The key research question is:

Can a student VQA model learn better reasoning behaviors by distilling teacher judgments over answers generated from different latent reasoning representations?

3. Problem Definition

Given an image I and a question Q, the model predicts an answer A.
Instead of mapping directly from (I,Q) to A, we introduce an intermediate latent reasoning representation R:

(I,Q)→R→A

The latent variable 
R is not observable and is supervised indirectly through its effect on the answer.

4. Model Architecture

The student model consists of four main components:

Vision Encoder
Extracts visual features from the input image.

Text Encoder
Encodes the question into a textual representation.

Multimodal Fusion Module
Combines visual and textual features into a shared representation.

Latent Reasoning Module
Generates a compact set of latent reasoning tokens 
R
R, which act as an information bottleneck and explicitly mediate the reasoning process.

Answer Decoder
Generates the final answer conditioned on both the fused features and the latent reasoning representation.

The reasoning module is placed between fusion and decoding to ensure that reasoning causally affects the output.

5. Latent Reasoning Representation

Reasoning is represented as a small set of learnable latent tokens:

The number of reasoning tokens controls the capacity of the reasoning space.

A small token count enforces a compact reasoning bottleneck.

The reasoning representation is input-conditioned, attending over fused multimodal features.

To encourage exploration, reasoning is modeled as a stochastic latent variable, allowing multiple plausible reasoning paths for the same input.

6. Teacher Model and Distillation Strategy

A larger teacher model is used only during training to evaluate the quality of student-generated answers.

Key properties of the teacher:

It does not provide reasoning text.

It does not supervise intermediate representations.

It only judges the plausibility or correctness of answers.

For a given 
(I,Q)
(I,Q), the student generates multiple candidate answers, each derived from a different latent reasoning sample.
The teacher compares these answers and expresses a preference for the better one.

This preference signal is used to update the student model, encouraging reasoning representations that lead to better answers.

7. Online Reasoning Distillation

The distillation process is online:

For each training example, the student samples multiple latent reasoning representations.

Each reasoning representation produces a candidate answer.

The teacher evaluates and ranks the candidate answers.

The student is trained using a preference-based objective that favors answers preferred by the teacher.

This process implicitly distills reasoning knowledge without explicitly modeling teacher reasoning.

8. Training Objective

The training objective consists of two components:

Answer Supervision Loss
Encourages the student to match ground-truth answers.

Teacher Preference Loss
Encourages the student to prefer reasoning paths that produce better answers according to the teacher.

Together, these losses guide the model to learn both accurate answering and improved reasoning behaviors.

9. Evaluation Strategy

The effectiveness of latent reasoning distillation is evaluated through:

Answer Accuracy on standard VQA benchmarks.

Reasoning Sensitivity Analysis, by perturbing or removing the latent reasoning representation.

Ablation Studies on the size and stochasticity of the reasoning latent space.

Comparison with models trained without teacher distillation or without latent reasoning.

10. Expected Contributions

This thesis makes the following contributions:

Introduces a framework for learning reasoning as a latent variable in VQA.

Proposes an online distillation method that transfers reasoning knowledge through teacher evaluation rather than explicit explanations.

Demonstrates that latent reasoning distillation improves answer accuracy without generating reasoning text.

Provides empirical evidence that reasoning can be learned and controlled without explicit linguistic supervision.

11. Significance

The proposed approach avoids the pitfalls of textual CoT supervision and offers a scalable, language-agnostic way to study reasoning in multimodal models.
It is particularly suitable for resource-constrained settings where training large reasoning-capable models directly is impractical.

12. Summary

This work frames reasoning as a latent, causally effective representation and shows that reasoning knowledge can be distilled from a teacher model through online evaluation of answers.
By focusing on behavior rather than explanation, the proposed method provides a principled alternative to explicit reasoning supervision in Visual Question Answering.