from groq import Groq
from .exception import log_exception, APIError
from .config import Config
from .logger import get_logger

logger = get_logger(__name__)

class LLM:
    def __init__(self):
        try:
            # Configure and initialize the Generative AI model
            self.llm = Groq(api_key=Config.GROQ_API_KEY)
            logger.info("LLM model initialized successfully.")
        except Exception as e:
            log_exception(e, "Failed to initialize the LLM model in LLM class.")
            raise APIError("Error initializing LLM model. Check API key and model name.") from e

    def prompt_template(self, result):
        # Generates a structured prompt for the LLM based on the provided data
        prompt = f"""You are a Green Finance Advisor specializing in evaluating companies based on sustainability impact areas, ESG scores, and risk probabilities predicted by an ML model. Your task is to generate a detailed and actionable Green Finance Diagnostic Report for investors, based on the provided data.

**Instructions:**
1. Analyze the ESG Score (out of 100) and ML Model's Risk Probability.
2. Use ESG weightage metrics:
   - **Environment**: 50%
   - **Social**: 30%
   - **Governance**: 20%
3. Provide a clear, concise, and investor-friendly report with key insights.
4. Only return the response in markdown compatible plain text without any additional explanations and external sentences or details.

**Input Data**:  
{result}

**Output Format:**

### [Company Name] Green Finance Investment Report

#### 1. **Company Overview**
- **Name**: [Company Name]  
- **Location**: [Location]  
- **Industry**: [Industry]  
- **Products/Services**: [Brief description of the company's offerings, including any sustainability-focused products or initiatives.]

---

#### 2. **ESG Score and Green Finance Readiness**
- **ESG Score (out of 100)**: **[ESG Score]**  
- **Readiness**:  
  - **Good (70-100)**: The company demonstrates strong commitment to sustainability, indicating favorable potential for green finance investments.  
  - **Fair (50-69)**: The company has moderate sustainability practices. While it shows progress, there are areas for improvement to be fully aligned with green finance goals.  
  - **Poor (Below 50)**: The company needs substantial improvement in its sustainability efforts to be considered a viable candidate for green finance investments.

---

#### 3. **ML Risk Probability**
- **Risk Probability**: **[Risk Probability]**  
- **Interpretation**:  
  - **High**: The company faces significant risks, which may include regulatory, environmental, or market challenges. Immediate attention to risk management is recommended.  
  - **Moderate**: There are some identifiable risks, but they are manageable with focused efforts on risk mitigation strategies.  
  - **Low**: The company demonstrates robust risk management practices, with low probability of facing major risks in the near future.

---

#### 4. **Sustainability Impact Areas**
- **Community**:  
  - [Provide insights into the company's community-focused initiatives, such as local impact projects, support for underserved communities, or efforts to enhance social well-being.]
  
- **Environment**:  
  - [Summarize the company’s environmental initiatives, including energy efficiency, carbon footprint reduction, waste management, and use of renewable resources.]
  
- **Customers**:  
  - [Detail how the company engages with customers through sustainable product offerings, ethical sourcing, or customer education on sustainability issues.]

- **Governance**:  
  - [Discuss the company's governance practices, transparency, ethical standards, and policies regarding executive compensation, board diversity, and shareholder rights.]

---

#### 5. **Key Strengths**
- [List the company’s main strengths related to sustainability, such as leadership in environmental practices, strong governance policies, positive social impact, or alignment with global sustainability standards.]
  
---

#### 6. **Areas for Improvement**
- [Provide actionable suggestions for improvement, including potential areas where the company can enhance its sustainability practices, improve risk management, or optimize its ESG score.]

---

#### 7. **Certification Recommendations**
- **Recommended Certifications**:  
  - [List relevant certifications or standards the company should pursue, such as B Corp, ISO 14001, GRI, LEED, or Fair Trade.]
  
- **Timeline**:  
  - [Suggest a feasible timeline for achieving these certifications, such as "Within the next 12 months."]

---

#### 8. **Benefits of Green Finance Alignment**
- [Describe the key benefits for investors, including enhanced marketability, access to sustainable investment funds, alignment with ESG goals, improved risk mitigation, and long-term financial performance.]

---

#### 9. **Next Steps**
- [Actionable next steps for investors to engage with the company in terms of green finance investments, such as "Evaluate the company's sustainability reports for deeper insights" or "Initiate a due diligence process to assess risk exposure."]

    """
        return prompt

    def inference(self, result):
        try:
            # Prepare and send the request to the LLM model
            refined_prompt = self.prompt_template(result)
            chat_completion = self.llm.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": refined_prompt,
                    }
                ],
                model=Config.GROQ_MODEL_NAME,
            )
            
            response = chat_completion.choices[0].message.content

            if response:
                logger.info("LLM inference successful.")
                return response
            else:
                logger.warning("LLM did not return any response.")
                raise APIError("LLM response is empty. Please check input format and prompt.")

        except Exception as e:
            log_exception(e, "Error during LLM inference in LLM class.")
            raise APIError("Error during LLM inference. Check input data and model configuration.") from e
