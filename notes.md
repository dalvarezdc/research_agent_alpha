> I have intention to later, not now, implement the possibility of using multiple models in one query/pipeline. How do we call 
this process btw? 

● Great question! The process of using multiple models in one query/pipeline has several names depending on the approach:

  Common Terms:

  1. Model Chaining - Sequential: Model A's output → Model B's input → Model C's input
  2. Model Ensemble - Parallel: Multiple models process same input, results are combined/voted
  3. Model Orchestration - General term for managing multiple models in a workflow
  4. Model Cascade - Cost optimization: Try cheap/fast model first, escalate to expensive/powerful if needed
  5. Multi-model Pipeline - Descriptive general term
  6. Mixture of Experts (MoE) - Different models specialize in different subtasks

  Your Use Case (Medical Analysis):

  Looking at your medical agents, you likely want:

  Model Chaining examples:
  - Fast model routes → Specialized model analyzes → Quality-check model validates
  - Model A: extracts key info → Model B: deep reasoning → Model C: formats output

  Model Ensemble examples:
  - 3 models analyze same medical query, pick consensus/best answer
  - Use multiple models for critical decisions, majority vote

  Model Cascade examples:
  - Try gpt-4o-mini first (cheap/fast) → if confidence low, escalate to claude-opus (expensive/thorough)

  ---
  How This Affects Router Implementation Choice:

  Given your multi-model pipeline intention, Option 2 (Universal LLM Caller) becomes EVEN MORE valuable because:

  ✅ Any part of your code can easily call any model: call_model("gpt-4o", messages)
  ✅ Easy to implement chaining:
  result1 = call_model("grok-fast", stage1_messages)
  result2 = call_model("claude-opus", stage2_messages)
  ✅ Easy to add ensemble later: call_models_parallel(["gpt-4o", "claude-opus"], messages)
  ✅ Easy to add cascade: call_model_with_fallback("gpt-4o-mini", fallback="gpt-4o", messages)

  Recommendation: Option 2 is perfect for your future multi-model plans.